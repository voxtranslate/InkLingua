"""
InkLinguaNet: CNN -> dual-granularity VQ bottleneck -> style-FiLM Transformer -> CTC, with the connected style task; the losses;
the EMA teacher; the style-equivariant GPU augmentation ([SES]) and the closed-form leakage penalty ([CFL]).
"""
from __future__ import annotations

import copy
import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig, ModelConfig, TrainConfig
from .style import SLANT_RANGE_DEG


# ==============================================================================
# layers
# ==============================================================================
class MaskedBatchNorm2d(nn.Module):
    """BatchNorm whose training statistics are computed over REAL columns only (nn.BatchNorm2d also counts the zero padding of short
    lines, so its statistics depended on how much padding the batch contained).  `m` is a [B,1,1,W] boolean mask of real columns;
    padded columns are returned as exact zeros.  `collect` (AdaBN) accumulates exact sums for test-time adaptation."""

    def __init__(self, c: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.register_buffer("running_mean", torch.zeros(c))
        self.register_buffer("running_var", torch.ones(c))
        self.collect: Optional[Dict[str, Any]] = None

    def forward(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        xf, mf = x.float(), m.to(torch.float32)
        if self.collect is not None and not self.training:
            n = (mf.sum() * x.size(2)).clamp(min=1.0)
            s1, s2 = (xf * mf).sum((0, 2, 3)), ((xf ** 2) * mf).sum((0, 2, 3))
            mean = s1 / n
            var = (s2 / n - mean ** 2).clamp(min=0.0)
            a = self.collect
            a["n"] = a.get("n", 0.0) + float(n)
            a["s1"] = a.get("s1", 0.0) + s1.double().cpu().numpy()
            a["s2"] = a.get("s2", 0.0) + s2.double().cpu().numpy()
        elif self.training:
            n = (mf.sum() * x.size(2)).clamp(min=1.0)
            mean = (xf * mf).sum((0, 2, 3)) / n
            var = (((xf - mean.view(1, -1, 1, 1)) ** 2) * mf).sum((0, 2, 3)) / n
            with torch.no_grad():
                unb = var * n / (n - 1.0).clamp(min=1.0)
                self.running_mean.lerp_(mean.detach(), self.momentum)
                self.running_var.lerp_(unb.detach(), self.momentum)
        else:
            mean, var = self.running_mean, self.running_var
        y = (xf - mean.view(1, -1, 1, 1)) * torch.rsqrt(var.view(1, -1, 1, 1) + self.eps)
        y = y * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return (y * mf).to(x.dtype)


class ConvBNAct(nn.Module):
    def __init__(self, cin: int, cout: int, k: int = 3, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, s, k // 2, bias=False)
        self.bn = MaskedBatchNorm2d(cout)

    def forward(self, x, m):
        return F.relu(self.bn(self.conv(x), m))


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride=(1, 1)):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = MaskedBatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = MaskedBatchNorm2d(cout)
        self.has_short = not (tuple(stride) == (1, 1) and cin == cout)
        if self.has_short:
            self.sc = nn.Conv2d(cin, cout, 1, stride, bias=False)
            self.sb = MaskedBatchNorm2d(cout)

    def forward(self, x, m):
        y = F.relu(self.b1(self.c1(x), m))
        y = self.b2(self.c2(y), m)
        s = self.sb(self.sc(x), m) if self.has_short else x
        return F.relu(y + s) * m.to(y.dtype)


class CNNBackbone(nn.Module):
    """ResNet-style. Width is downsampled 4x (T = ceil(W/4) frames), height 16x.  Returns a LOW-level map (stroke level, H/4) and a
    HIGH-level map (component level, H/16).  Every layer sees a column mask, so a line's output does not depend on the batch padding."""

    def __init__(self, ch=(32, 64, 128, 256)):
        super().__init__()
        self.stem = ConvBNAct(1, ch[0])
        self.l1 = nn.ModuleList([BasicBlock(ch[0], ch[1], (2, 2)), BasicBlock(ch[1], ch[1])])
        self.l2 = nn.ModuleList([BasicBlock(ch[1], ch[2], (2, 2)), BasicBlock(ch[2], ch[2])])
        self.l3 = nn.ModuleList([BasicBlock(ch[2], ch[3], (2, 1)), BasicBlock(ch[3], ch[3])])
        self.l4 = nn.ModuleList([BasicBlock(ch[3], ch[3], (2, 1)), BasicBlock(ch[3], ch[3])])

    def forward(self, x, widths):
        W = x.size(-1)

        def mk(f):
            lim = torch.ceil(widths.float() / f).long()
            return (torch.arange(-(-W // f), device=x.device)[None] < lim[:, None])[:, None, None, :]

        m1, m2, m4 = mk(1), mk(2), mk(4)
        x = self.stem(x, m1)
        for blk in self.l1:
            x = blk(x, m2)
        for blk in self.l2:
            x = blk(x, m4)
        low = x
        for blk in list(self.l3) + list(self.l4):
            x = blk(x, m4)
        return low, x


class VectorQuantizerEMA(nn.Module):
    """VQ-VAE codebook with EMA updates, data-dependent init and dead-code restarts.  Input [B,T,D]; `valid` [B,T] marks real frames."""

    def __init__(self, num_codes: int, dim: int, decay: float = 0.99, eps: float = 1e-5, dead_thresh: float = 0.5):
        super().__init__()
        self.K, self.D, self.decay, self.eps, self.dead = num_codes, dim, decay, eps, dead_thresh
        self.register_buffer("embed", torch.randn(num_codes, dim) * 0.1)
        self.register_buffer("embed_avg", self.embed.clone())
        self.register_buffer("cluster_size", torch.ones(num_codes))
        self.register_buffer("initted", torch.zeros(1))
        self._init_py: Optional[bool] = None

    @torch.no_grad()
    def _update(self, flat: torch.Tensor, idx: torch.Tensor) -> None:
        oh = F.one_hot(idx, self.K).to(flat.dtype)
        n, s = oh.sum(0), oh.t() @ flat
        self.cluster_size.mul_(self.decay).add_(n, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(s, alpha=1 - self.decay)
        tot = self.cluster_size.sum()
        cs = (self.cluster_size + self.eps) / (tot + self.K * self.eps) * tot
        self.embed.copy_(self.embed_avg / cs.unsqueeze(1))
        dead = self.cluster_size < self.dead
        if flat.size(0) > 0 and bool(dead.any()):
            r = flat[torch.randint(0, flat.size(0), (int(dead.sum()),), device=flat.device)]
            self.embed[dead] = r
            self.embed_avg[dead] = r
            self.cluster_size[dead] = self.dead * 2

    def forward(self, z: torch.Tensor, valid: torch.Tensor):
        with torch.autocast(device_type=z.device.type, enabled=False):
            zf = z.float()
            B, T, D = zf.shape
            flat = zf.reshape(-1, D)
            vflat = valid.reshape(-1)
            if self._init_py is None:
                self._init_py = bool(float(self.initted) > 0)
            if self.training and not self._init_py and int(vflat.sum()) >= self.K:
                with torch.no_grad():
                    fv = flat[vflat].detach()
                    pick = fv[torch.randperm(fv.size(0), device=z.device)[:self.K]]
                    self.embed.copy_(pick)
                    self.embed_avg.copy_(pick)
                    self.initted.fill_(1)
                    self._init_py = True
            d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.t() + self.embed.pow(2).sum(1)[None]
            idx = d.argmin(1)
            q = F.embedding(idx, self.embed).view(B, T, D)
            if self.training:
                self._update(flat[vflat].detach(), idx[vflat])
            vm = valid.unsqueeze(-1).float()
            commit = (((zf - q.detach()) ** 2) * vm).sum() / (vm.sum() * D).clamp(min=1)
            q_st = zf + (q - zf).detach()
            with torch.no_grad():
                p = torch.bincount(idx[vflat], minlength=self.K).float()
                p = p / p.sum().clamp(min=1)
                ppl = torch.exp(-(p * torch.log(p + 1e-10)).sum())
        return q_st.to(z.dtype), q.to(z.dtype), idx.view(B, T), commit, ppl

    def _load_from_state_dict(self, *a, **kw):
        self._init_py = None
        super()._load_from_state_dict(*a, **kw)


class PositionalEncoding(nn.Module):
    def __init__(self, d: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div[: d // 2])
        self.max_len = max_len
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, T: int) -> torch.Tensor:
        if T > self.max_len:
            raise ValueError(f"{T} frames exceed the positional-encoding table ({self.max_len}); reduce cfg.data.max_img_width")
        return self.pe[:, :T]


class FiLM(nn.Module):
    """Feature-wise affine modulation from the style embedding (identity at init).  gamma / beta are BOUNDED (|.| <= max_scale, tanh)
    and have NO bias (an unconstrained, biased FiLM learned a constant per-channel offset that carried no style)."""

    def __init__(self, sdim: int, d: int, max_scale: float = 0.5):
        super().__init__()
        self.lin = nn.Linear(sdim, 2 * d, bias=False)
        nn.init.zeros_(self.lin.weight)
        self.max_scale = float(max_scale)

    def forward(self, x, s):
        g, b = (self.max_scale * torch.tanh(self.lin(s))).chunk(2, dim=-1)
        return x * (1 + g.unsqueeze(1)) + b.unsqueeze(1), g, b


class ConvModule(nn.Module):
    def __init__(self, d: int, k: int = 7):
        super().__init__()
        self.ln, self.pw1 = nn.LayerNorm(d), nn.Linear(d, 2 * d)
        self.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d)
        self.pw2 = nn.Linear(d, d)

    def forward(self, x, valid):
        y = F.glu(self.pw1(self.ln(x)), dim=-1).masked_fill(~valid.unsqueeze(-1), 0.0)
        y = F.gelu(self.dw(y.transpose(1, 2))).transpose(1, 2)
        return x + self.pw2(y)


class EncoderLayer(nn.Module):
    def __init__(self, d, heads, ffn, drop, k):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.conv = ConvModule(d, k)
        self.ff = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Dropout(drop), nn.Linear(ffn, d))
        self.drop = nn.Dropout(drop)

    def forward(self, x, pad_mask, valid, need_attn=False):
        h = self.n1(x)
        a, w = self.attn(h, h, h, key_padding_mask=pad_mask, need_weights=need_attn, average_attn_weights=False)
        x = x + self.drop(a)
        x = self.conv(x, valid)
        x = x + self.drop(self.ff(self.n2(x)))
        return x, w


class _GradScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.a = a
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * ctx.a, None


def grad_scale(x: torch.Tensor, a: float) -> torch.Tensor:
    return x if a == 1.0 else _GradScale.apply(x, a)


def masked_mean(x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    w = valid.unsqueeze(-1).to(x.dtype)
    return (x * w).sum(1) / w.sum(1).clamp(min=1)


class StyleEncoder(nn.Module):
    """Style stream. Input: quantisation residuals of both codebook levels (what the discrete content codes cannot express).
    Output: style embedding + descriptor regression + style class + the attention over frames (which frames define the style)."""

    def __init__(self, d_in: int, style_dim: int, n_desc: int, n_cls: int):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(d_in, 256), nn.GELU(), nn.LayerNorm(256))
        self.conv = nn.Conv1d(256, 256, 5, padding=2)
        self.att = nn.Linear(256, 1)
        self.out = nn.Sequential(nn.Linear(256, style_dim), nn.LayerNorm(style_dim))
        self.reg = nn.Sequential(nn.Linear(style_dim, 128), nn.GELU(), nn.Linear(128, n_desc))
        self.cls = nn.Linear(style_dim, n_cls)

    def forward(self, feat, valid):
        h = self.proj(feat).masked_fill(~valid.unsqueeze(-1), 0.0)
        h = F.gelu(self.conv(h.transpose(1, 2))).transpose(1, 2)
        a = self.att(h).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)          # [B,T]
        s = self.out((h * a.unsqueeze(-1)).sum(1))
        return s, self.reg(s), self.cls(s), a


# ==============================================================================
# the network
# ==============================================================================
class InkLinguaNet(nn.Module):
    def __init__(self, mc: ModelConfig, num_classes: int, num_desc: int, num_clusters: int, img_h: int):
        super().__init__()
        self.mc, self.num_classes = mc, num_classes
        ch, d = mc.channels, mc.d_model
        self.backbone = CNNBackbone(ch)
        self.low_proj = nn.Sequential(nn.Linear(ch[2] * (img_h // 4), mc.low_dim), nn.LayerNorm(mc.low_dim))
        self.high_proj = nn.Sequential(nn.Linear(ch[3] * (img_h // 16), d), nn.LayerNorm(d))
        self.vq_low = VectorQuantizerEMA(mc.codes_low, mc.low_dim, mc.vq_decay, dead_thresh=mc.vq_dead_thresh)
        self.vq_high = VectorQuantizerEMA(mc.codes_high, d, mc.vq_decay, dead_thresh=mc.vq_dead_thresh)
        self.fuse = nn.Sequential(nn.Linear(mc.low_dim + d, d), nn.LayerNorm(d))
        self.aux_head = nn.Linear(d, num_classes)                      # inter-CTC on the quantised content
        self.mask_token = nn.Parameter(torch.randn(d) * 0.02)
        self.pos = PositionalEncoding(d)
        self.lstm = nn.LSTM(d, d // 2, batch_first=True, bidirectional=True) if mc.lstm_prefix else None
        self.lstm_ln = nn.LayerNorm(d) if mc.lstm_prefix else None
        self.film_in = FiLM(mc.style_dim, d, mc.film_max) if mc.film else None
        self.film_out = FiLM(mc.style_dim, d, mc.film_max) if mc.film else None
        self.layers = nn.ModuleList([EncoderLayer(d, mc.heads, mc.ffn, mc.dropout, mc.conv_kernel) for _ in range(mc.enc_layers)])
        self.final_ln = nn.LayerNorm(d)
        self.ctc_head = nn.Linear(d, num_classes)
        self.mask_head = nn.Linear(d, mc.codes_high)
        self.style_enc = StyleEncoder(mc.low_dim + d, mc.style_dim, num_desc, num_clusters)

    def _span_mask(self, valid: torch.Tensor, p: float, span: int) -> torch.Tensor:
        starts = torch.rand(valid.shape, device=valid.device) < (p / max(span, 1))
        m = F.max_pool1d(starts.float().unsqueeze(1), span, 1, span // 2).squeeze(1)[:, :valid.size(1)] > 0
        return m & valid

    def forward(self, x, widths, vq_lambda: float = 1.0, mask_prob: float = 0.0, extras: bool = False, need_attn: bool = False,
                keep_hidden: bool = False) -> Dict[str, Any]:
        """keep_hidden=True also returns the encoder state after every layer (`hidden`: list of [B,T,d]) for the layer-wise probes."""
        low, high = self.backbone(x, widths)
        T = high.size(-1)
        lens = torch.clamp(torch.ceil(widths.float() / 4).long(), min=1, max=T)
        valid = torch.arange(T, device=x.device)[None] < lens[:, None]
        zl = self.low_proj(low.permute(0, 3, 1, 2).flatten(2))
        zh = self.high_proj(high.permute(0, 3, 1, 2).flatten(2))
        ql_st, ql, il, c_l, ppl_l = self.vq_low(zl, valid)
        qh_st, qh, ih, c_h, ppl_h = self.vq_high(zh, valid)

        cl = zl + vq_lambda * (ql_st - zl)                      # content stream: annealed quantisation (continuous -> discrete)
        ch = zh + vq_lambda * (qh_st - zh)
        content = self.fuse(torch.cat([cl, ch], -1))
        aux_lp = F.log_softmax(self.aux_head(content).float(), -1)
        c_pool = masked_mean(content, valid)

        if self.mc.style_input == "residual":                   # style stream: what the codebooks cannot express
            sfeat = torch.cat([zl - ql.detach(), zh - qh.detach()], -1)
        else:
            sfeat = torch.cat([zl, zh], -1)
        style, style_reg, style_cls, style_att = self.style_enc(grad_scale(sfeat, self.mc.style_grad_scale), valid)

        mask = None
        h = content
        if self.training and mask_prob > 0:                     # masked-code modelling (span masking of the content stream)
            mask = self._span_mask(valid, mask_prob, self.mc.mask_span)
            h = torch.where(mask.unsqueeze(-1), self.mask_token.to(content.dtype).expand_as(content), content)
        if self.lstm is not None:
            pk = nn.utils.rnn.pack_padded_sequence(self.lstm_ln(h), lens.cpu(), batch_first=True, enforce_sorted=False)
            r, _ = self.lstm(pk)
            r, _ = nn.utils.rnn.pad_packed_sequence(r, batch_first=True, total_length=T)
            h = h + r.to(h.dtype)
        h = h + self.pos(T).to(h.dtype)
        cond = style.detach() if self.mc.film_detach_style else style
        g_in = g_out = None
        if self.film_in is not None:
            h, g_in, _ = self.film_in(h, cond)
        pad = ~valid
        attns, hidden = [], [h] if keep_hidden else []
        nl = len(self.layers)
        for li, layer in enumerate(self.layers):
            if (self.training and self.mc.layer_drop > 0 and not need_attn           # stochastic depth: the residual path alone
                    and float(torch.rand(())) < self.mc.layer_drop * (li + 1) / nl):
                continue
            h, w = layer(h, pad, valid, need_attn)
            if need_attn:
                attns.append(w)
            if keep_hidden:
                hidden.append(h)
        h = self.final_ln(h)
        if self.film_out is not None:
            h, g_out, _ = self.film_out(h, cond)
        log_probs = F.log_softmax(self.ctc_head(h).float(), -1)

        out: Dict[str, Any] = {"log_probs": log_probs, "out_lens": lens, "aux_log_probs": aux_lp, "style_reg": style_reg.float(),
                               "style_cls": style_cls.float(), "style_emb": style.float(), "content_pool": c_pool.float(),
                               "vq_commit": c_l + c_h, "ppl_low": ppl_l, "ppl_high": ppl_h}
        if mask is not None and bool(mask.any()):
            out["mask_logits"] = self.mask_head(h[mask]).float()
            out["mask_target"] = ih[mask]
        if extras:
            out.update({"idx_low": il, "idx_high": ih, "valid": valid, "attn": attns, "style_att": style_att.float(),
                        "film_gamma": g_in if g_in is not None else g_out})
        if keep_hidden:
            out["hidden"] = hidden
        return out


# ==============================================================================
# losses
# ==============================================================================
def ctc_per_sample(log_probs, out_lens, targets, target_lens) -> torch.Tensor:
    loss = F.ctc_loss(log_probs.transpose(0, 1), targets, out_lens, target_lens, blank=0, reduction="none", zero_infinity=True)
    return loss / target_lens.clamp(min=1).float()


def cross_cov_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.size(0) < 4:
        return a.new_zeros(())
    a = (a - a.mean(0)) / (a.std(0) + 1e-5)
    b = (b - b.mean(0)) / (b.std(0) + 1e-5)
    return ((a.t() @ b / (a.size(0) - 1)) ** 2).mean()


def frame_consistency(student_lp: torch.Tensor, teacher_lp: torch.Tensor, lens: torch.Tensor, rows: Optional[torch.Tensor] = None) -> torch.Tensor:
    """KL(teacher || student) of the per-frame label posteriors, averaged over real frames (of the selected `rows`).  The teacher (EMA)
    reads the weak view, the student the strong view of the same geometry, so frame t describes the same pixels in both (CR-CTC-style
    view consistency, applied to EVERY line -- the unlabeled lines teach invariances without any label)."""
    T = student_lp.size(1)
    valid = (torch.arange(T, device=student_lp.device)[None] < lens[:, None]).float()
    if rows is not None:
        valid = valid * rows[:, None].float()
    kl = (teacher_lp.exp() * (teacher_lp - student_lp)).sum(-1)
    return (kl * valid).sum() / valid.sum().clamp(min=1.0)


def fragment_distillation(student_lp: torch.Tensor, teacher_lp: torch.Tensor, lens: torch.Tensor, trust: torch.Tensor,
                          rows: torch.Tensor, tau: float, anchor: float) -> torch.Tensor:
    """[FRAG-RC] soft self-distillation restricted to trusted fragments.  On trusted frames the target is the SHARPENED teacher
    posterior softmax(log p / tau) (tau < 1: the student becomes more decisive exactly where the teacher is reliable); on the other
    frames of the same lines the target is the UNsharpened teacher posterior with weight `anchor`.  Averaged over all frames of the
    selected rows, so the gradient scales with the trusted share.  rows: [B] bool; trust: [B, T] bool."""
    T = student_lp.size(1)
    valid = (torch.arange(T, device=student_lp.device)[None] < lens[:, None]) & rows[:, None]
    t = teacher_lp.detach()
    sharp = F.log_softmax(t / max(tau, 1e-3), dim=-1)
    kl_s = (sharp.exp() * (sharp - student_lp)).sum(-1)
    kl_a = (t.exp() * (t - student_lp)).sum(-1)
    w = torch.where(trust, torch.ones_like(kl_s), torch.full_like(kl_s, float(anchor))) * valid.float()
    kl = torch.where(trust, kl_s, kl_a)
    return (kl * w).sum() / valid.float().sum().clamp(min=1.0)


class ClosedFormLeakage:
    """[CFL] Closed-form leakage penalty on the content stream, with a transcription-explained floor.  A queue holds recent (content
    embedding, standardised style target, measured-mask) triples; every step a ridge regression is fitted on the queue in closed form
    and applied to the CURRENT batch; the encoder minimises  mean_j relu( E[yhat_j^2] - floor_j - slack )  over the active descriptors
    (targets are z-scores, so E[yhat^2] estimates the explained variance).  floor_j = R2 the TRANSCRIPTION alone achieves."""

    def __init__(self, dim: int, n_desc: int, size: int, ridge: float, slack: float, active: np.ndarray, floor: np.ndarray, device: torch.device):
        self.size, self.ridge, self.slack = int(size), float(ridge), float(slack)
        self.active = [int(j) for j in np.flatnonzero(np.asarray(active, bool))]
        self.floor = torch.as_tensor(np.asarray(floor, np.float32), device=device)
        self.C = torch.zeros(self.size, dim, device=device)
        self.Y = torch.zeros(self.size, n_desc, device=device)
        self.M = torch.zeros(self.size, n_desc, device=device)
        self.n, self.ptr = 0, 0
        self.last: Dict[str, float] = {}

    @torch.no_grad()
    def _push(self, c: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> None:
        for k in range(c.size(0)):
            self.C[self.ptr], self.Y[self.ptr], self.M[self.ptr] = c[k], y[k], m[k]
            self.ptr = (self.ptr + 1) % self.size
            self.n = min(self.n + 1, self.size)

    def __call__(self, c: torch.Tensor, y: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        c, y, m = c.float(), y.detach().float(), m.detach().float()
        zero = c.new_zeros(())
        if not self.active or self.n < max(64, self.size // 4):
            self._push(c.detach(), y, m)
            return zero
        with torch.no_grad():
            Cq, Yq, Mq = self.C[:self.n], self.Y[:self.n], self.M[:self.n]
            mu, sd = Cq.mean(0), Cq.std(0) + 1e-4
            A = (Cq - mu) / sd
            W = torch.zeros(A.size(1), Yq.size(1), device=c.device)
            eye = torch.eye(A.size(1), device=c.device)
            for j in self.active:
                mj = Mq[:, j]
                nj = mj.sum()
                if float(nj) < 32:
                    continue
                Aj = A * mj[:, None]
                W[:, j] = torch.linalg.solve(Aj.t() @ A / nj + self.ridge * eye, Aj.t() @ Yq[:, j] / nj)
        pred = ((c - mu) / sd) @ W
        leak = (pred ** 2 * m).sum(0) / m.sum(0).clamp(min=1.0)
        act = torch.zeros_like(leak, dtype=torch.bool)
        act[self.active] = True
        pen = torch.relu(leak - self.floor - self.slack)[act].mean()
        self.last = {"leak_r2_" + str(j): float(leak[j].detach()) for j in self.active}
        self._push(c.detach(), y, m)
        return pen


def compute_losses(out: Dict[str, Any], batch: Dict[str, Any], lc: LossConfig, style_t: torch.Tensor, style_mask: torch.Tensor,
                   cluster_ok: torch.Tensor, training: bool, aux_scale: float = 1.0, vq_scale: float = 1.0,
                   teacher_lp: Optional[torch.Tensor] = None, leak: Optional[ClosedFormLeakage] = None,
                   frag: Optional[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
    """All loss terms.  Recognition losses are weighted per sample (labeled = 1, pseudo-label = risk weight, none = 0).  Style losses use
    the (augmentation-transformed) targets `style_t` [B,5] on the entries flagged in `style_mask`.  `teacher_lp`: EMA-teacher posteriors
    (weak view) -> consistency loss; `frag`: trust masks of the unlabeled rows -> fragment distillation."""
    dev = out["log_probs"].device
    w = batch["weight"].to(dev).float()
    tgt, tl = batch["target"].to(dev), batch["target_len"].to(dev)
    L: Dict[str, torch.Tensor] = {}
    ctc = ctc_per_sample(out["log_probs"], out["out_lens"], tgt, tl)
    L["ctc"] = (ctc * w).sum() / w.sum().clamp(min=1.0)
    aux = ctc_per_sample(out["aux_log_probs"], out["out_lens"], tgt, tl)
    L["aux_ctc"] = (aux * w).sum() / w.sum().clamp(min=1.0)
    L["_ctc_ps"] = ctc.detach()
    if training and lc.ctc_smooth > 0:
        # [CTC-LS] label smoothing on EMITTING frames only: -mean_{c != blank} log p_t(c) where the arg-max is a character; the blank frames
        # stay sharp (a plain entropy bonus would also flatten the blanks and break the alignment)
        lpf = out["log_probs"]
        Tn = lpf.size(1)
        vm = (torch.arange(Tn, device=lpf.device)[None] < out["out_lens"][:, None]) & (lpf.argmax(-1) != 0)
        ls = (-lpf[..., 1:].mean(-1) * vm).sum(1) / vm.sum(1).clamp(min=1)
        L["ctc_ls"] = (ls * w).sum() / w.sum().clamp(min=1.0)
    else:
        L["ctc_ls"] = out["log_probs"].new_zeros(())
    sm = style_mask.float()
    cl = batch["cluster"].to(dev)
    L["style_reg"] = (F.smooth_l1_loss(out["style_reg"], style_t, reduction="none") * sm).sum() / sm.sum().clamp(min=1.0)
    cmask = cluster_ok.float() * (cl >= 0).float()
    cls_ps = F.cross_entropy(out["style_cls"], cl, ignore_index=-100, reduction="none", label_smoothing=lc.style_cls_smoothing)
    L["style_cls"] = (cls_ps * cmask).sum() / cmask.sum().clamp(min=1.0)
    if training:
        zero = out["log_probs"].new_zeros(())
        L["leak"] = leak(out["content_pool"], style_t, sm) if (leak is not None and lc.leak > 0) else zero
        use_frag = frag is not None and teacher_lp is not None and lc.frag > 0
        L["frag"] = (fragment_distillation(out["log_probs"], teacher_lp, out["out_lens"], frag["trust"], frag["rows"], frag["tau"], frag["anchor"])
                     if use_frag else zero)
        L["cons"] = (frame_consistency(out["log_probs"], teacher_lp, out["out_lens"], (~frag["rows"]) if use_frag else None)
                     if (teacher_lp is not None and lc.consistency > 0) else zero)
        L["decor"] = cross_cov_loss(out["style_emb"], out["content_pool"])
        L["vq"] = out["vq_commit"]
        L["mask"] = F.cross_entropy(out["mask_logits"], out["mask_target"]) if "mask_logits" in out else zero
        L["total"] = (L["ctc"] + lc.ctc_smooth * L["ctc_ls"] + vq_scale * lc.vq_commit * L["vq"]
                      + aux_scale * (lc.aux_ctc * L["aux_ctc"] + lc.style_reg * L["style_reg"] + lc.style_cls * L["style_cls"]
                                     + lc.leak * L["leak"] + lc.consistency * L["cons"] + lc.frag * L["frag"]
                                     + lc.decorrelate * L["decor"] + lc.masked_code * L["mask"]))
    else:   # validation "task loss": comparable between train-subset and val
        L["total"] = L["ctc"] + lc.style_reg * L["style_reg"] + lc.style_cls * L["style_cls"]
    return L


# ==============================================================================
# EMA teacher
# ==============================================================================
class ModelEMA:
    """Exponential moving average of the weights AND of the float buffers (VQ codebooks, BN statistics)."""

    def __init__(self, model: nn.Module, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for pe, pm in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1 - d)
        for (n, be), (_, bm) in zip(self.module.named_buffers(), model.named_buffers()):
            if be.dtype.is_floating_point and not n.endswith("initted"):
                be.mul_(d).add_(bm.detach(), alpha=1 - d)
            else:
                be.copy_(bm)


# ==============================================================================
# [SES] geometry <-> style relations and the GPU augmentation
# ==============================================================================
def transform_slant_deg(slant_deg: torch.Tensor, shear: torch.Tensor, rot: torch.Tensor, vscale: torch.Tensor,
                        inv_hscale: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Closed-form effect of the sampling map  x_in = a * x_out + (shear - rot) * y_out,  y_in = rot * x_out + vscale * y_out  on the
    slant (deg, >0 = leaning right):  tan(slant') = (vscale * tan(slant) + shear - rot) / (a + rot * tan(slant))."""
    a = 1.0 if inv_hscale is None else inv_hscale
    t = torch.tan(torch.deg2rad(slant_deg))
    t2 = (vscale * t + (shear - rot)) / (a + rot * t)
    return torch.rad2deg(torch.atan(t2))


def body_ratio_t(x: torch.Tensor, thr: float = 0.45) -> torch.Tensor:
    """Body ratio of [B,1,H,W] images, measured exactly like style.body_ratio_from_rows; 0 for blank images."""
    H = x.size(2)
    rows = (x > thr).float().sum(3)
    s = F.conv1d(rows, torch.ones(1, 1, 3, device=x.device), padding=1)[:, 0]
    mx = s.amax(1, keepdim=True)
    return ((s >= 0.5 * mx).float().sum(1) / H) * (mx[:, 0] > 0).float()


class GPUAugment:
    """Batch augmentation on [B,1,H,W] ink images (ink = 1).  [SES] Style-Equivariant Supervision: geometric augmentations change the
    style descriptors in KNOWN ways, so the targets are TRANSFORMED instead of discarded (slant: closed form; body ratio: re-measured;
    stroke / density / gap: masked for altered lines).  Two views are returned: STRONG (geometry + morphology + photometric + blur +
    noise) for the student and WEAK (the same geometry only) for the EMA teacher -- frame t covers the same pixels in both."""

    def __init__(self, tc: TrainConfig, style_mean: np.ndarray, style_std: np.ndarray):
        self.tc = tc
        self.mean = torch.as_tensor(style_mean, dtype=torch.float32)
        self.std = torch.as_tensor(style_std, dtype=torch.float32)

    def __call__(self, x: torch.Tensor, style_z: torch.Tensor, style_ok: torch.Tensor, widths: torch.Tensor,
                 min_width: Optional[torch.Tensor] = None
                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tc, B, dev = self.tc, x.size(0), x.device
        H, Wc = x.shape[-2:]
        mean, std = self.mean.to(dev), self.std.to(dev)
        rnd = lambda lo, hi: torch.empty(B, 1, 1, 1, device=dev).uniform_(lo, hi)  # noqa: E731
        st = style_z.to(dev).float().clone()
        sm = style_ok.to(dev).float().clone()
        wd = widths.to(dev).clone()
        cok = torch.ones(B, dtype=torch.bool, device=dev)

        def colmask(w):
            return (torch.arange(Wc, device=dev)[None] < w[:, None]).float()[:, None, None, :]

        if tc.strong_aug:
            body0 = body_ratio_t(x)
            geo = torch.rand(B, device=dev) < tc.aug_geom_prob
            hs = torch.rand(B, device=dev) < tc.aug_hscale_prob
            ela = torch.rand(B, device=dev) < tc.aug_elastic_prob
            shear = torch.empty(B, device=dev).uniform_(-tc.aug_shear_max, tc.aug_shear_max) * geo
            rot = torch.empty(B, device=dev).uniform_(-1, 1) * (tc.aug_tilt_px / (Wc / 2.0)) * geo
            vsc = 1.0 + (torch.empty(B, device=dev).uniform_(*tc.aug_vscale) - 1.0) * geo
            s_h = torch.empty(B, device=dev).uniform_(*tc.aug_hscale)        # horizontal stretch bounded by the canvas and by CTC
            w_f = wd.float().clamp(min=1.0)
            hi_ = Wc / w_f
            lo_ = (min_width.to(dev).float() / w_f) if min_width is not None else torch.zeros_like(w_f)
            s_h = torch.minimum(torch.maximum(s_h, lo_), hi_)
            s_h = torch.where((lo_ > hi_) | ~hs, torch.ones_like(s_h), s_h)
            wd_new = torch.clamp(torch.ceil(w_f * s_h), max=float(Wc)).long()
            s_eff = wd_new.float() / w_f
            hs = hs & (wd_new != wd)
            a_inv = 1.0 / s_eff
            ar = Wc / H
            theta = torch.zeros(B, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = a_inv, (shear - rot) / ar, a_inv - 1.0   # left edge stays put
            theta[:, 1, 0], theta[:, 1, 1] = rot * ar, vsc
            grid = F.affine_grid(theta, list(x.shape), align_corners=False)
            if bool(ela.any()):
                flow = F.interpolate(torch.randn(B, 2, 3, 12, device=dev), size=(H, Wc), mode="bilinear", align_corners=True)
                dx = flow[:, 0] * tc.aug_elastic_px * (2.0 / Wc)
                dy = flow[:, 1] * 0.5 * tc.aug_elastic_px * (2.0 / H)
                grid = grid + torch.stack([dx, dy], -1) * ela.view(B, 1, 1, 1).float()
            if bool((geo | hs | ela).any()):
                x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
            wd = torch.where(hs, wd_new, wd)
            moved = geo | hs
            slant = st[:, 0] * std[0] + mean[0]
            new = torch.where(moved, transform_slant_deg(slant, shear, rot, vsc, a_inv), slant)
            inside = new.abs() < (SLANT_RANGE_DEG - 1.0)
            st[:, 0] = ((new.clamp(-SLANT_RANGE_DEG, SLANT_RANGE_DEG) - mean[0]) / std[0]).clamp(-3.0, 3.0)
            x_weak = x * colmask(wd)
            m = torch.rand(B, device=dev)
            dil = m < tc.aug_morph_prob / 2
            ero = (m >= tc.aug_morph_prob / 2) & (m < tc.aug_morph_prob)
            morph = dil | ero
            if bool(morph.any()):
                d_, e_ = F.max_pool2d(x, 3, 1, 1), -F.max_pool2d(-x, 3, 1, 1)
                x = torch.where(dil.view(B, 1, 1, 1), d_, torch.where(ero.view(B, 1, 1, 1), e_, x))
            altered = moved | ela | morph
            if tc.style_equivariant:
                st[:, 4] = (st[:, 4] + (body_ratio_t(x) - body0) / std[4]).clamp(-3.0, 3.0)
                sm[:, 0] = sm[:, 0] * (inside | ~moved).float()
                sm[:, 1:4] = sm[:, 1:4] * (~altered).float().unsqueeze(1)
            else:
                sm[:, 0] = sm[:, 0] * (~moved).float()
                sm[:, 1:] = sm[:, 1:] * (~altered).float().unsqueeze(1)
            cok = ~altered
        else:
            x_weak = x
        x = (x * rnd(0.75, 1.25) + rnd(-0.05, 0.05)).clamp(0, 1)
        x = x.clamp(1e-4, 1) ** rnd(0.75, 1.3)
        if tc.strong_aug:
            k = torch.rand(B, device=dev) < tc.aug_blur_prob
            if bool(k.any()):
                bl = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)
                x = torch.where(k.view(B, 1, 1, 1), bl, x)
            x = x + torch.randn_like(x) * rnd(0.0, 0.05) * (x.amax((1, 2, 3), keepdim=True) > 0)
        col = colmask(wd)
        return (x * col).clamp(0, 1), (x_weak * col).clamp(0, 1), wd, st, sm, cok
