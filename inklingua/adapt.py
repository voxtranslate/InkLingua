"""
Model adaptability to new handwriting:

  * `run_finetune` (mode "finetune"): continue ANY InkLingua checkpoint on a new corpus / corpus mix with LABELS (a registry key, a custom
    DatasetSpec or a local parquet folder).  The alphabet grows function-preservingly ([ALPHA-SPLIT]), every component of the normal training
    loop (FRAG-RC, MixLine, protocol, suite-compatible outputs) applies.  This is the path for "a new dataset with transcriptions".
  * `ZeroShotAdapter` (mode "adapt"): a FOLDER of line images in a language / orthography the model never saw, with few or no labels: alphabet
    growth + [LEX-TIE] language-prior resolution of tied variants + [XL-RC] transfer-calibrated risk control of the pseudo-labels + [TIE-MARGINAL]
    training.  Translation into any language.  This is the path for "handwriting without transcriptions".
  * writer self-distillation (inklingua/posthoc.py) is the transductive, label-free path at inference time.
"""
from __future__ import annotations

import copy
import csv
import json
import os
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .alphabet import CharGroups, extend_codec, extend_ctc_heads, resolve_variants
from .config import Config, ModelConfig, dc_from_dict
from .data import BucketBatchSampler, HandwritingDataModule, preprocess_line, to_u8
from .decoding import greedy_decode, seq_confidence
from .llm import Hypothesis, LLMEngine
from .model import GPUAugment, InkLinguaNet, MaskedBatchNorm2d, ModelEMA, VectorQuantizerEMA, ctc_per_sample
from .pseudo import PERTURBATIONS, RELIABILITY_FEATURES, SelectionRule, apply_perturbation, reliability_features, reliability_views
from .riskcontrol import transfer_risk_control
from .style import NUM_DESC
from .text import CharCodec, cer, ctc_min_frames, edit_distance, grapheme_clusters, normalize_text, to_logical, to_visual
from .utils import LOG, atomic_save, json_safe, resolve_device, set_seed, setup_logging, torch_load

IMAGE_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp")


# ==============================================================================
# fine-tuning on a labeled corpus
# ==============================================================================
def run_finetune(cfg: Config) -> Optional[Dict[str, Any]]:
    """Warm start from cfg.train.init_from on the corpora of cfg.data.sources.  The model config is taken from the checkpoint (shapes must
    match), the data module builds the alphabet as an EXTENSION of the checkpoint's, the Trainer grows the output layers, then the normal
    train -> evaluate path runs (resumable, its own run directory)."""
    from .main import run_single
    p = cfg.train.init_from
    if not p or not os.path.exists(p):
        raise FileNotFoundError(f"mode 'finetune' needs cfg.train.init_from = path of an InkLingua checkpoint (got {p!r})")
    st = torch_load(p)
    if "cfg" in st:
        cfg.model = dc_from_dict(ModelConfig, st["cfg"]["model"])
        cfg.data.img_height = st["cfg"]["data"].get("img_height", cfg.data.img_height)
    units = list(st.get("units") or grapheme_clusters(st["chars"]))
    LOG.info("[FINETUNE] from %s (epoch %s, %d units) onto %s", p, st.get("epoch"), len(units), cfg.data.sources)
    return run_single(cfg, base_units=units)


# ==============================================================================
# target-language data (a folder of line images)
# ==============================================================================
def list_line_images(folder: str, limit: Optional[int] = None) -> List[str]:
    if not folder or not os.path.isdir(folder):
        raise FileNotFoundError(f"cfg.adapt.lines_dir='{folder}' is not a directory")
    names = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMAGE_EXT))
    if not names:
        raise FileNotFoundError(f"no line images ({', '.join(IMAGE_EXT)}) in {folder}")
    return names[:limit] if limit else names


def read_label_file(path: str, rtl: bool = False) -> Dict[str, str]:
    """UTF-8 'filename<TAB>text' per line -> {filename: visual-order text}, also indexed by file stem.  Text is NFC-normalised; RTL text (logical
    order in the file) is converted to visual order."""
    out: Dict[str, str] = {}
    if not path:
        return out
    with open(path, encoding="utf-8-sig") as f:
        for ln in f:
            ln = ln.rstrip("\r\n")
            if not ln.strip():
                continue
            name, sep, txt = ln.partition("\t")
            if not sep:
                name, _, txt = ln.partition(" ")
            txt = to_visual(unicodedata.normalize("NFC", normalize_text(txt)), rtl)
            name = os.path.basename(name.strip())
            out[name] = txt
            out.setdefault(os.path.splitext(name)[0], txt)
    return out


class LineSet(Dataset):
    """Preprocessed target-language lines (uint8 ink, exactly the preprocessing of training) + the text they are trained on."""

    def __init__(self, folder: str, names: List[str], img_h: int, max_w: int, min_w: int, pad_x: int):
        self.names, self.img_h = names, img_h
        self.ink = []
        for n in names:
            with Image.open(os.path.join(folder, n)) as im:
                self.ink.append(to_u8(preprocess_line(im, img_h, max_w, min_w, pad_x)))
        self.width = np.array([a.shape[1] for a in self.ink])
        self.text = [""] * len(names)
        self.weight = np.zeros(len(names), np.float32)
        self.tied = [False] * len(names)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ink = self.ink[i].astype(np.float32) / 255.0
        return {"index": i, "image": torch.from_numpy(ink)[None], "width": ink.shape[1], "text": self.text[i], "weight": float(self.weight[i]), "tied": bool(self.tied[i])}


def collate_lines(items: List[Dict[str, Any]], codec, pad_multiple: int = 4) -> Dict[str, Any]:
    B, H = len(items), items[0]["image"].shape[1]
    Wm = int(np.ceil(max(it["width"] for it in items) / pad_multiple) * pad_multiple)
    imgs = torch.zeros(B, 1, H, Wm)
    tgt, tlen, minw = [], [], []
    for b, it in enumerate(items):
        imgs[b, :, :, :it["width"]] = it["image"]
        ids = codec.encode(it["text"]) if it["weight"] > 0 else []
        tgt += ids
        tlen.append(len(ids))
        minw.append(4 * ctc_min_frames(ids) if ids else 0)
    return {"image": imgs, "width": torch.tensor([it["width"] for it in items]), "target": torch.tensor(tgt, dtype=torch.long),
            "target_len": torch.tensor(tlen, dtype=torch.long), "weight": torch.tensor([it["weight"] if tl > 0 else 0.0 for it, tl in zip(items, tlen)]),
            "tied": torch.tensor([bool(it.get("tied", False)) for it in items]), "min_width": torch.tensor(minw, dtype=torch.long),
            "index": torch.tensor([it["index"] for it in items], dtype=torch.long)}


def resolve_ckpt(cfg: Config, name: str) -> str:
    d = cfg.ckpt_dir
    if name == "best":
        p = os.path.join(d, "best.pt")
        return p if os.path.exists(p) else os.path.join(d, "last.pt")
    if name == "last":
        return os.path.join(d, "last.pt")
    return name


def _set_adapt_mode(model: nn.Module) -> None:
    """Dropout on, but the batch-norm statistics and the VQ codebooks stay those of the source model."""
    model.train()
    for m in model.modules():
        if isinstance(m, (MaskedBatchNorm2d, VectorQuantizerEMA)):
            m.eval()


def load_source_calibration(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    z = np.load(path, allow_pickle=False)
    return {"X": z["X"], "ed": z["ed"], "rl": z["rl"], "epoch": int(z["epoch"]), "chars": str(z["chars"])}


# ==============================================================================
# the zero-shot adapter
# ==============================================================================
class ZeroShotAdapter:
    """Adapt a trained checkpoint to the handwriting in `cfg.adapt.lines_dir`, then transcribe and translate it into every target language.
    0. [ALPHA-SPLIT] extend the alphabet; 1. the EMA teacher reads every line, the SOURCE calibration set gives the risk score;
    2. [LEX-TIE] ambiguous variants decided by vision + language model; 3. [XL-RC] threshold transferred to the target (weights + anchor audit);
    4. the student trains on the selected pseudo-labels (+ training anchors); the teacher follows by EMA; repeat.  `eval_file` is only measured."""

    def __init__(self, cfg: Config, device: torch.device, llm: Optional[LLMEngine] = None):
        self.cfg, self.ac, self.device = cfg, cfg.adapt, device
        self.pc = cfg.train.pseudo
        self.llm = llm
        self.log: Dict[str, Any] = {"rounds": []}
        self.tied: set = set()

    def load_source(self) -> None:
        p = resolve_ckpt(self.cfg, self.ac.checkpoint)
        if not os.path.exists(p):
            raise FileNotFoundError(f"no source checkpoint at {p}: train first (cfg.run.do_train) or set cfg.adapt.checkpoint")
        st = torch_load(p)
        weights = st["weights"] if "weights" in st else st["ema"]
        ckc = st.get("cfg") or self.cfg.to_dict()
        mcfg = dc_from_dict(ModelConfig, ckc["model"])
        self.dcfg = ckc["data"]
        self.src_units = list(st.get("units") or grapheme_clusters(st["chars"]))
        model = InkLinguaNet(mcfg, len(self.src_units) + 1, NUM_DESC, st.get("n_clusters", 1), self.dcfg["img_height"])
        model.load_state_dict(weights)
        self.model0, self.src_codec = model, CharCodec(self.src_units)
        self.ckpt_path = p
        LOG.info("[ZERO-SHOT] source model %s: %d units, epoch %s", p, len(self.src_units), st.get("epoch"))

    def load_lines(self) -> None:
        ac = self.ac
        names = list_line_images(ac.lines_dir, ac.max_lines)
        self.labels = read_label_file(ac.anchors_file, ac.rtl)
        self.eval_labels = read_label_file(ac.eval_file, ac.rtl)
        lab = lambda n, d: d.get(n, d.get(os.path.splitext(n)[0]))                     # noqa: E731
        d = self.dcfg
        self.ds = LineSet(ac.lines_dir, names, d["img_height"], d["max_img_width"], d["min_img_width"], d["pad_x"])
        anchors = [i for i, n in enumerate(names) if lab(n, self.labels) is not None]
        ev = [i for i, n in enumerate(names) if lab(n, self.eval_labels) is not None and i not in set(anchors)]
        rs = np.random.RandomState(self.cfg.run.seed)
        anchors = list(rs.permutation(anchors)) if anchors else []
        if ac.anchor_limit is not None:
            anchors = anchors[:ac.anchor_limit]
        self.anchor_ref = {int(i): lab(names[i], self.labels) for i in anchors}
        self.eval_ref = {int(i): lab(names[i], self.eval_labels) for i in ev}
        n_tr = int(round(ac.anchor_train_share * len(anchors))) if len(anchors) >= ac.min_anchors_to_train else 0
        self.anchors_train, self.anchors_cal = [int(i) for i in anchors[:n_tr]], [int(i) for i in anchors[n_tr:]]
        self.target_idx = np.array([i for i in range(len(names)) if i not in self.eval_ref and i not in set(self.anchors_train)], np.int64)
        LOG.info("[ZERO-SHOT] %d target lines | anchors: %d train + %d calibration | evaluation lines (measure only): %d", len(names), len(self.anchors_train),
                 len(self.anchors_cal), len(self.eval_ref))

    def extend_alphabet(self) -> None:
        extra_text = self.ac.charset + "".join(self.anchor_ref.values())
        self.codec, self.groups = extend_codec(self.src_codec, extra_text, self.ac.visual_base)
        self.model = extend_ctc_heads(self.model0, self.groups).to(self.device)
        g = self.groups
        LOG.info("[ALPHA-SPLIT] +%d units (%d variants of %d known letters, %d without visual base: %s)", g.V - g.n_old, g.V - g.n_old - len(g.orphans),
                 g.n_variant_groups, len(g.orphans), "".join(g.units[i - 1] for i in g.orphans) or "-")

    def _loader(self, idxs: Sequence[int], bs: int, shuffle: bool = False, epoch: int = 0) -> DataLoader:
        s = BucketBatchSampler(self.ds.width, bs, shuffle, 50, seed=self.cfg.run.seed + epoch, indices=list(idxs))
        return DataLoader(self.ds, batch_sampler=s, collate_fn=lambda it: collate_lines(it, self.codec), num_workers=0)

    @torch.no_grad()
    def read(self, model: nn.Module, idxs: Sequence[int]) -> Dict[str, Any]:
        model.eval()
        pc, g = self.pc, self.groups
        views = reliability_views(pc)
        n_gate = len(pc.cons_views) if pc.consistency else 0
        r: Dict[str, Any] = {"idx": [], "text": [], "ids": [], "frames": [], "conf": [], "cons": [], "X": [], "lp": {}}
        for batch in self._loader(idxs, self.ac.batch_size):
            x, w = batch["image"].to(self.device), batch["width"].to(self.device)
            o = model(x, w)
            lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
            B = lp.shape[0]
            reads = []
            for b in range(B):
                lpc = g.collapse(lp[b, :lens[b]])
                ids, cf, fr = greedy_decode(lpc)
                reads.append((lpc, ids, cf, fr, self.codec.decode_ids(ids)))
            cons_v = np.zeros((B, len(views)))
            for vi, v in enumerate(views):
                ov = model(apply_perturbation(PERTURBATIONS[v], x, w), w)
                lpv, lnv = ov["log_probs"].cpu().numpy(), ov["out_lens"].cpu().tolist()
                for b in range(B):
                    cons_v[b, vi] = cer(reads[b][4], self.codec.decode_ids(greedy_decode(g.collapse(lpv[b, :lnv[b]]))[0]))
            for b in range(B):
                lpc, ids, cf, fr, text = reads[b]
                i = int(batch["index"][b])
                r["idx"].append(i), r["text"].append(text), r["ids"].append(ids), r["frames"].append(fr)
                r["conf"].append(seq_confidence(cf))
                r["cons"].append(float(cons_v[b, :n_gate].mean()) if n_gate else float("nan"))
                r["X"].append(reliability_features(lpc, ids, cf, fr, cons_v[b]))
                r["lp"][i] = lp[b, :lens[b]].copy()
        r["X"] = np.array(r["X"]) if r["X"] else np.zeros((0, len(RELIABILITY_FEATURES)))
        return r

    def _scorer(self):
        if self.llm is not None and self.llm.can_score():
            rtl = self.ac.rtl
            base = self.llm.score_texts
            return (lambda texts: base([to_logical(t, True) for t in texts])) if rtl else base
        return None

    def resolve(self, R: Dict[str, Any], rows: Sequence[int], use_lm: bool = True) -> Dict[int, Tuple[str, Dict[str, float]]]:
        ac, sc = self.ac, (self._scorer() if use_lm else None)
        out = {}
        for k in rows:
            i = R["idx"][k]
            out[i] = resolve_variants(R["ids"][k], R["frames"][k], R["lp"][i], self.groups, sc, beam=ac.lex_beam, vis_w=ac.lex_vis_weight, lm_w=ac.lex_lm_weight,
                                      rho=ac.lex_rho, decided_p=ac.lex_decided_prob, max_options=ac.lex_max_options)
        return out

    def measure(self, model: nn.Module, label: str) -> Dict[str, float]:
        if not self.eval_ref:
            return {}
        idxs = sorted(self.eval_ref)
        R = self.read(model, idxs)
        rows = list(range(len(R["idx"])))
        res_lm, res_v = self.resolve(R, rows, True), self.resolve(R, rows, False)
        ref = [self.eval_ref[i] for i in R["idx"]]
        out = {}
        for tag, res in (("lex_tie", res_lm), ("vision_only", res_v)):
            ed = sum(edit_distance(a, res[i][0]) for a, i in zip(ref, R["idx"]))
            out[f"CER_{tag}"] = ed / max(sum(len(a) for a in ref), 1)
        LOG.info("[ZERO-SHOT] %s | eval CER %.2f %% with language-prior tie-breaking, %.2f %% vision only (%d lines)", label, 100 * out["CER_lex_tie"],
                 100 * out["CER_vision_only"], len(ref))
        return out

    def select(self, R: Dict[str, Any], rule: Optional[SelectionRule], src: Optional[Dict[str, Any]], resolved: Dict[int, Tuple[str, Dict]]
               ) -> Tuple[Dict[int, Tuple[str, float]], Dict[str, Any]]:
        ac, pc = self.ac, self.pc
        pos = {i: k for k, i in enumerate(R["idx"])}
        score_all = rule.predict(R["X"]) if rule is not None else np.full(len(R["idx"]), np.nan)
        tgt_rows = [pos[int(i)] for i in self.target_idx if int(i) in pos]
        cal_rows = [pos[i] for i in self.anchors_cal if i in pos and i in resolved]
        info: Dict[str, Any] = {"mode": "none", "k": 0, "theta": float("-inf")}
        if src is None or rule is None:
            LOG.warning("[XL-RC] no source calibration (source_calibration.npz): NO pseudo-labels are selected -- only the training anchors are learned from. "
                        "Mode 'source_calib' rebuilds the file for a checkpoint.")
            return {}, info
        X_s, ed_s, rl_s, score_s = src["X"], src["ed"], src["rl"], rule.oof
        sa = ea = la = None
        if cal_rows:
            sa = score_all[cal_rows]
            ea = np.array([edit_distance(self.anchor_ref[R["idx"][k]], resolved[R["idx"][k]][0]) for k in cal_rows], np.float64)
            la = np.array([len(self.anchor_ref[R["idx"][k]]) for k in cal_rows], np.float64)
        res = transfer_risk_control(X_s, score_s, ed_s, rl_s, R["X"][tgt_rows], score_all[tgt_rows], ac.target_cer, ac.risk_z, ac.min_selected, ac.min_ess,
                                    score_a=sa, ed_a=ea, rl_a=la, zero_anchor_inflation=ac.zero_anchor_inflation, prior_edits=ac.anchor_prior_edits, seed=self.cfg.run.seed)
        info.update(res)
        theta = res["theta"]
        sel: Dict[int, Tuple[str, float]] = {}
        if res["k"] > 0:
            lo = float(np.min(score_all[tgt_rows])) if tgt_rows else 0.0
            pool = list(dict.fromkeys(tgt_rows + cal_rows))
            for k in pool:
                i = R["idx"][k]
                if i not in resolved or not np.isfinite(score_all[k]) or score_all[k] > theta:
                    continue
                if pc.consistency and not (np.isfinite(R["cons"][k]) and R["cons"][k] <= pc.tau_cons):
                    continue
                text = resolved[i][0]
                ids_ = self.codec.encode(text)
                if len(text) < pc.min_chars or ctc_min_frames(ids_) > len(R["lp"][i]):
                    continue
                rel = float(np.clip((score_all[k] - lo) / max(theta - lo, 1e-9), 0.0, 1.0))
                sel[i] = (text, 1.0 - 0.5 * rel)
        info["selected"] = len(sel)
        info["coverage_lines"] = len(sel) / max(len(tgt_rows), 1)
        return sel, info

    def train_round(self, student: nn.Module, ema: ModelEMA, opt, sel: Dict[int, Tuple[str, float]], rnd: int) -> Dict[str, float]:
        """[TIE-MARGINAL] a pseudo-label whose variants the language model could not settle is trained with the loss of the MARGINAL label
        (CTC on group-summed posteriors with the base-letter text): no gradient between the copies, so the model is not taught its own coin flips."""
        ac, ds = self.ac, self.ds
        for i in range(len(ds)):
            ds.text[i], ds.weight[i] = "", 0.0
        for i in self.anchors_train:
            ds.text[i], ds.weight[i] = self.anchor_ref[i], 1.0
        for i, (t, w) in sel.items():
            ds.text[i], ds.weight[i] = t, w
        ds.tied = [(i in sel and i in self.tied) for i in range(len(ds))]
        idx = [i for i in range(len(ds)) if ds.weight[i] > 0]
        if not idx:
            return {"n_train": 0, "loss": float("nan")}
        _set_adapt_mode(student)
        tot, cnt = 0.0, 0
        for ep in range(ac.epochs_per_round):
            for batch in self._loader(idx, ac.batch_size, True, rnd * 100 + ep):
                x, wd = batch["image"].to(self.device), batch["width"].to(self.device)
                B = x.size(0)
                zs = torch.zeros(B, NUM_DESC, device=self.device)
                x_s, _, wd, _, _, _ = self.aug(x, zs, zs, wd, batch["min_width"])
                out = student(x_s, wd)
                tgt, tl = batch["target"].to(self.device), batch["target_len"].to(self.device)
                per = ctc_per_sample(out["log_probs"], out["out_lens"], tgt, tl)
                tied = batch["tied"].to(self.device)
                if bool(tied.any()):
                    tb = torch.tensor(self.groups.base_ids(tgt.tolist()), dtype=torch.long, device=self.device)
                    per_m = ctc_per_sample(self.groups.collapse_t(out["log_probs"]), out["out_lens"], tb, tl)
                    per = torch.where(tied, per_m, per)
                wt = batch["weight"].to(self.device)
                loss = (per * wt).sum() / wt.sum().clamp(min=1e-6)
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for g_ in opt.param_groups for p in g_["params"]], self.cfg.train.grad_clip)
                opt.step()
                self.step += 1
                ema.update(student, self.step)
                tot, cnt = tot + float(loss), cnt + 1
        student.eval()
        return {"n_train": len(idx), "loss": tot / max(cnt, 1)}

    def _optimizer(self, student: nn.Module):
        for p in student.parameters():
            p.requires_grad_(False)
        train = [student.ctc_head, student.final_ln] + ([student.film_out] if student.film_out is not None else [])
        train += list(student.layers)[len(student.layers) - max(0, self.ac.unfreeze_layers):] if self.ac.unfreeze_layers > 0 else []
        params = []
        for m in train:
            for p in m.parameters():
                p.requires_grad_(True)
                params.append(p)
        return torch.optim.AdamW(params, lr=self.ac.lr, weight_decay=1e-2, betas=(0.9, 0.98))

    def run(self) -> Dict[str, Any]:
        ac, cfg = self.ac, self.cfg
        out_dir = ac.out_dir or os.path.join(cfg.run_dir, "adapt")
        os.makedirs(out_dir, exist_ok=True)
        self.load_source()
        self.load_lines()
        self.extend_alphabet()
        src = load_source_calibration(ac.source_calibration or os.path.join(cfg.ckpt_dir, "source_calibration.npz"))
        if src is None:
            LOG.warning("[XL-RC] source_calibration.npz not found (written by training with the legacy pseudo-labels; mode 'source_calib' rebuilds it)")
        rule = SelectionRule.fit(src["X"], src["ed"], src["rl"], self.pc, seed=cfg.run.seed) if src is not None else None
        student = self.model
        ema = ModelEMA(student, ac.ema_decay)
        self.aug = GPUAugment(cfg.train, np.zeros(NUM_DESC, np.float32), np.ones(NUM_DESC, np.float32))
        opt = self._optimizer(student)
        self.step = 0
        metrics: Dict[str, Any] = {"zero_shot_start": self.measure(ema.module, "round 0 (before any training)")}
        all_idx = list(range(len(self.ds)))
        for rnd in range(ac.rounds):
            t0 = time.time()
            read_idx = [i for i in all_idx if i not in self.eval_ref]
            R = self.read(ema.module, read_idx)
            score_all = rule.predict(R["X"]) if rule is not None else np.zeros(len(R["idx"]))
            pos = {i: k for k, i in enumerate(R["idx"])}
            order = [pos[int(i)] for i in self.target_idx if int(i) in pos]
            order = sorted(order, key=lambda k: score_all[k])[:ac.lex_max_lines]
            rows = list(dict.fromkeys(order + [pos[i] for i in self.anchors_cal if i in pos]))
            resolved = self.resolve(R, rows, True)
            sel, info = self.select(R, rule, src, resolved)
            self.tied = {i for i, (_, inf_) in resolved.items() if inf_.get("n_ambiguous", 0) > 0 and inf_.get("margin", float("inf")) < ac.lex_tie_margin}
            info["tied_lines"] = len([i for i in sel if i in self.tied])
            tr = self.train_round(student, ema, opt, sel, rnd)
            rec = {"round": rnd + 1, "seconds": round(time.time() - t0, 1), **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in info.items()}, **tr,
                   "eval": self.measure(ema.module, f"after round {rnd + 1}")}
            self.log["rounds"].append(rec)
            LOG.info("[ZERO-SHOT] round %d/%d | %s | threshold covers %.1f %% of the target lines | factor %.2f (%d anchors) | selected %d lines | domain AUROC %.2f | "
                     "train loss %.3f", rnd + 1, ac.rounds, info.get("mode"), 100 * info.get("coverage_target", 0.0), info.get("factor", float("nan")),
                     info.get("n_anchors", 0), info.get("selected", 0), info.get("domain_auroc", float("nan")), tr.get("loss", float("nan")))
        metrics["rounds"] = self.log["rounds"]
        metrics["zero_shot_end"] = self.measure(ema.module, "final")
        self._finish(ema.module, out_dir, metrics)
        return metrics

    def _finish(self, model: nn.Module, out_dir: str, metrics: Dict[str, Any]) -> None:
        ac, cfg = self.ac, self.cfg
        R = self.read(model, [i for i in range(len(self.ds))])
        src = load_source_calibration(ac.source_calibration or os.path.join(cfg.ckpt_dir, "source_calibration.npz"))
        rule = SelectionRule.fit(src["X"], src["ed"], src["rl"], self.pc, seed=cfg.run.seed) if src is not None else None
        score = rule.predict(R["X"]) if rule is not None else np.full(len(R["idx"]), np.nan)
        rows = sorted(range(len(R["idx"])), key=lambda k: (not np.isfinite(score[k]), score[k]))[:ac.lex_final_lines]
        resolved = self.resolve(R, rows, True)
        rowsd = []
        for k in range(len(R["idx"])):
            i = R["idx"][k]
            text = resolved[i][0] if i in resolved else self.resolve(R, [k], False)[i][0]
            shown = to_logical(text, ac.rtl)
            langs = {}
            if self.llm is not None and k < ac.translate_max:
                hyp = Hypothesis(text=shown, char_conf=[], logprob=0.0, post=1.0)
                for lg in ac.target_languages:
                    try:
                        langs[lg] = self.llm.translate([hyp], uncertain=False, target_language=lg)["translation"]
                    except Exception as e:
                        langs[lg] = f"<translation failed: {e}>"
            ref = self.eval_ref.get(i, self.anchor_ref.get(i, ""))
            rowsd.append({"file": self.ds.names[i], "transcription": shown, "risk": float(score[k]) if np.isfinite(score[k]) else "",
                          "split": "eval" if i in self.eval_ref else ("anchor" if i in self.anchor_ref else "target"), "reference": to_logical(ref, ac.rtl),
                          **{f"translation_{lg}": langs.get(lg, "") for lg in ac.target_languages}})
        with open(os.path.join(out_dir, "adapt_results.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rowsd[0].keys()))
            w.writeheader()
            w.writerows(rowsd)
        atomic_save({"weights": model.state_dict(), "chars": self.codec.chars, "units": self.codec.units, "cfg": cfg.to_dict(), "from": "adapt", "n_clusters": 1,
                     "epoch": self.step}, os.path.join(out_dir, "adapted.pt"))
        with open(os.path.join(out_dir, "adapt_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe({**metrics, "new_units": self.codec.units[len(self.src_units):], "n_lines": len(self.ds), "n_anchors_train": len(self.anchors_train),
                                 "n_anchors_calibration": len(self.anchors_cal), "source_checkpoint": self.ckpt_path, "steps": self.step}), f, indent=2, ensure_ascii=False)
        LOG.info("[ZERO-SHOT] wrote %s", out_dir)


def run_adapt(cfg: Optional[Config] = None) -> Dict[str, Any]:
    from . import __version__
    cfg = cfg or Config()
    cfg.make_dirs()
    setup_logging(cfg)
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    LOG.info("InkLingua v%s | zero-shot adaptation | device %s", __version__, device)
    llm = LLMEngine(cfg.llm, device)
    return ZeroShotAdapter(cfg, device, llm).run()


def run_adapt_curve(cfg: Optional[Config] = None) -> Dict[str, Any]:
    """CER on the evaluation lines as a function of the number of LABELED target lines (cfg.adapt.anchor_counts)."""
    cfg = cfg or Config()
    cfg.make_dirs()
    setup_logging(cfg)
    device = resolve_device(cfg.run.device)
    llm = LLMEngine(cfg.llm, device)
    curve = {}
    for k in cfg.adapt.anchor_counts:
        c = copy.deepcopy(cfg)
        c.adapt.anchor_limit = int(k)
        c.adapt.out_dir = os.path.join(cfg.adapt.out_dir or os.path.join(cfg.run_dir, "adapt"), f"anchors_{k}")
        set_seed(cfg.run.seed)
        m = ZeroShotAdapter(c, device, llm).run()
        curve[int(k)] = {"start": m.get("zero_shot_start"), "end": m.get("zero_shot_end"), "last_round": (m.get("rounds") or [{}])[-1].get("selected")}
    base = cfg.adapt.out_dir or os.path.join(cfg.run_dir, "adapt")
    with open(os.path.join(base, "adapt_curve.json"), "w") as f:
        json.dump(json_safe(curve), f, indent=2)
    try:
        ks = sorted(curve)
        fig, ax = plt.subplots(figsize=(5, 3.4))
        for key, lab in (("CER_lex_tie", "language prior + vision"), ("CER_vision_only", "vision only")):
            ax.plot(ks, [100 * (curve[k]["end"] or {}).get(key, float("nan")) for k in ks], "o-", label=f"adapted, {lab}")
            ax.plot(ks, [100 * (curve[k]["start"] or {}).get(key, float("nan")) for k in ks], "x--", label=f"before, {lab}")
        ax.set_xlabel("labeled target lines (anchors)"), ax.set_ylabel("CER on evaluation lines (%)"), ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(base, "adapt_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as e:
        LOG.warning("curve plot failed: %s", e)
    return curve


def run_source_calib(cfg: Optional[Config] = None) -> str:
    """Rebuild source_calibration.npz for an existing checkpoint (needs the SOURCE data: same cfg.data / seed / run.name)."""
    from .inferencer import Inferencer
    from .trainer import Trainer
    cfg = cfg or Config()
    cfg.make_dirs()
    setup_logging(cfg)
    set_seed(cfg.run.seed)
    device = resolve_device(cfg.run.device)
    data = HandwritingDataModule(cfg).prepare()
    llm = LLMEngine(cfg.llm, device)
    inf = Inferencer(cfg, data, llm, device)
    inf.load(resolve_ckpt(cfg, cfg.adapt.checkpoint))
    tr = Trainer(cfg, data, inf.model, llm, device)
    rc = tr._teacher_read(data.calib_idx, desc="source calibration lines")
    ref = [data.train_ds.reference(i) for i in rc["idx"]]
    ed = np.array([edit_distance(a, b) for a, b in zip(ref, rc["text"])], np.float64)
    rl = np.array([len(a) for a in ref], np.float64)
    tr._save_source_calibration(rc["X"], ed, rl, -1)
    p = os.path.join(cfg.ckpt_dir, "source_calibration.npz")
    LOG.info("source calibration: %d lines, CER %.2f %% -> %s", len(ed), 100 * ed.sum() / max(rl.sum(), 1), p)
    return p
