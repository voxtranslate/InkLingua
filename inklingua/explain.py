"""
Explanation helpers used by the Inferencer: per-sample panels (input, CTC posterior, character confidence, saliency, integrated gradients,
occlusion, per-character attribution, attention rollout, VQ codes, style attention), calibration / selective-prediction statistics,
codebook statistics and the batch-norm helpers of the test-time adaptation.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .model import MaskedBatchNorm2d

SPACE = "␣"
ALIGN_CAP = 1000          # max #lines used for the O(n*m) character alignments (confusions, calibration)


def chlabel(c: str) -> str:
    return SPACE if c == " " else c


# ---------------------------------------------------------------- calibration / selective prediction
def adaptive_calibration(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> Dict[str, Any]:
    """Equal-MASS bins (a CTC model puts >90 % of its characters into the top fixed-width bin).  ECE = sum_b (n_b / N) |acc_b - conf_b|."""
    conf, correct = np.asarray(conf, np.float64), np.asarray(correct, np.float64)
    if len(conf) < bins:
        return {"ece": float("nan"), "nll": float("nan"), "brier": float("nan"), "mean_conf": float("nan"), "accuracy": float("nan"), "points": [], "n": int(len(conf))}
    o = np.argsort(conf, kind="stable")
    c, a = conf[o], correct[o]
    pts, ece = [], 0.0
    for p in np.array_split(np.arange(len(c)), bins):
        pts.append((float(c[p].mean()), float(a[p].mean()), len(p)))
        ece += len(p) / len(c) * abs(pts[-1][1] - pts[-1][0])
    cc = np.clip(c, 1e-6, 1 - 1e-6)
    return {"ece": float(ece), "nll": float(-(a * np.log(cc) + (1 - a) * np.log(1 - cc)).mean()), "brier": float(((c - a) ** 2).mean()),
            "mean_conf": float(c.mean()), "accuracy": float(a.mean()), "points": pts, "n": int(len(c))}


def risk_coverage(score: np.ndarray, ed: np.ndarray, ref_len: np.ndarray) -> Dict[str, Any]:
    """Selective prediction: keep the lines with the smallest `score`; CER of the kept lines vs the fraction kept.  AURC = area (lower is better)."""
    n = len(score)
    o = np.argsort(score, kind="stable")
    cov = np.arange(1, n + 1) / n
    risk = np.cumsum(np.asarray(ed, np.float64)[o]) / np.maximum(np.cumsum(np.asarray(ref_len, np.float64)[o]), 1.0)
    aurc = float(((risk[1:] + risk[:-1]) * 0.5 * np.diff(cov)).sum()) if n > 1 else float(risk[-1])
    return {"coverage": cov, "risk": risk, "aurc": aurc}


def coverage_at_risk(rc: Dict[str, Any], target: float) -> float:
    ok = np.flatnonzero(rc["risk"] <= target)
    return float(rc["coverage"][ok[-1]]) if ok.size else 0.0


def risk_at_coverage(rc: Dict[str, Any], cov: float) -> float:
    k = int(np.clip(np.searchsorted(rc["coverage"], cov), 0, len(rc["risk"]) - 1))
    return float(rc["risk"][k])


# ---------------------------------------------------------------- codebook statistics
def _entropy(p: np.ndarray) -> float:
    p = np.asarray(p, np.float64)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _purity_nmi(c: np.ndarray) -> Tuple[float, float]:
    n = c.sum()
    pur = float(c.max(1).sum() / n)
    pk, pc = c.sum(1) / n, c.sum(0) / n
    mi = _entropy(pc) + _entropy(pk) - _entropy(c / n)
    return pur, float(2 * mi / max(_entropy(pk) + _entropy(pc), 1e-12))


def code_char_stats(cc: np.ndarray, min_count: int = 20, n_null: int = 20, seed: int = 0) -> Dict[str, float]:
    """How well does a discrete code predict the character emitted on its frame?  Purity / NMI of the codes with >= min_count emissions,
    compared with a multinomial null that keeps the emission counts; *_gain = (x - null) / (1 - null)."""
    cc = np.asarray(cc, np.float64)
    tot = cc.sum(1)
    keep = tot >= min_count
    nan = float("nan")
    out = {"component_code_min_count": float(min_count), "component_codes_with_support": float(keep.sum()),
           "component_code_emission_share_with_support": float(tot[keep].sum() / max(tot.sum(), 1))}
    if keep.sum() < 2 or tot[keep].sum() <= 0:
        out.update(component_code_purity=nan, component_code_purity_chance=nan, component_code_purity_gain=nan, component_code_nmi=nan,
                   component_code_nmi_chance=nan, component_code_nmi_gain=nan)
        return out
    sub = cc[keep]
    pur, nmi = _purity_nmi(sub)
    marg, rng = sub.sum(0) / sub.sum(), np.random.RandomState(seed)
    nulls = [_purity_nmi(np.stack([rng.multinomial(int(t), marg) for t in sub.sum(1)]).astype(np.float64)) for _ in range(n_null)]
    pur0, nmi0 = np.mean(nulls, 0)
    out.update(component_code_purity=pur, component_code_purity_chance=float(pur0), component_code_purity_gain=float((pur - pur0) / max(1 - pur0, 1e-9)),
               component_code_nmi=nmi, component_code_nmi_chance=float(nmi0), component_code_nmi_gain=float((nmi - nmi0) / max(1 - nmi0, 1e-9)))
    return out


def pca2(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:2].T


def contrast_rollout(rows: np.ndarray) -> Tuple[np.ndarray, float]:
    """Sink-removed attention rollout: subtract the component shared by all emissions (attention sinks) to keep what is specific to each
    character.  rows: (K, T), each row sums to 1.  Returns (positive residual, mean mass on the 3 most shared frames)."""
    rows = np.asarray(rows, np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2:
        return rows.copy(), float("nan")
    shared = rows.mean(0)
    top = np.argsort(-shared)[:3]
    return np.maximum(rows - shared[None], 0.0), float(rows[:, top].sum(1).mean())


# ---------------------------------------------------------------- figures
def save_fig(fig, path: str, dpi: int = 120) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _show_img(ax, ctx, alpha: float = 1.0) -> None:
    ax.imshow(1.0 - ctx["img"], cmap="gray", vmin=0, vmax=1, extent=[0, ctx["W"], ctx["H"], 0], aspect="auto", alpha=alpha)


def _smooth_norm(m: np.ndarray, sigma: float = 2.0, signed: bool = False) -> np.ndarray:
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), sigma)
    s = np.percentile(np.abs(m), 99) + 1e-9
    return np.clip(m / s, -1, 1) if signed else np.clip(m / s, 0, 1)


def viz_input(ax, ctx):
    _show_img(ax, ctx)
    sp, st = ctx["style_pred"], ctx["style_true"]
    tag = f"[{ctx['tag']}] " if ctx.get("tag") else ""
    ax.set_title(f"{tag}GT  : {ctx['gt_display']}\nPRED: {ctx['pred_display']}   (CER {ctx['cer']:.3f}, conf {ctx['conf']:.2f})\n"
                 f"style pred/measured  slant {sp[0]:+.0f}/{st[0]:+.0f} deg | stroke {sp[1]:.3f}/{st[1]:.3f} | gap {sp[3]:.2f}/{st[3]:.2f} | "
                 f"body {sp[4]:.2f}/{st[4]:.2f}   (nan = not measurable)", fontsize=8, loc="left")
    ax.set_yticks([])


def viz_posterior(ax, ctx):
    P, units = ctx["probs"], ctx["units"]
    keep = [v for v in range(P.shape[1]) if P[:, v].max() >= 0.05]
    keep = [0] + sorted([v for v in keep if v != 0], key=lambda v: int(P[:, v].argmax()))
    ax.imshow(P[:, keep].T, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[0, ctx["T"] * 4, len(keep), 0])
    ax.set_yticks(np.arange(len(keep)) + 0.5)
    ax.set_yticklabels(["blank"] + [chlabel(units[v - 1]) for v in keep[1:]], fontsize=6)
    ax.set_title("CTC posterior  p(char | frame)", fontsize=8, loc="left")


def viz_charconf(ax, ctx):
    H = ctx["H"]
    _show_img(ax, ctx, alpha=0.35)
    ops = {o[2]: o[0] for o in ctx["ops"] if o[2] is not None}
    cmap = plt.get_cmap("RdYlGn")
    for k, (f, c) in enumerate(zip(ctx["frames"], ctx["char_conf"])):
        bad = ops.get(k, "match") in ("sub", "ins")
        ax.bar(f * 4 + 2, -c * H, width=4, bottom=H, color=cmap(c), edgecolor="red" if bad else "none", linewidth=1.2, alpha=0.9)
        ax.text(f * 4 + 2, -0.03 * H, chlabel(ctx["pred_units"][k]), ha="center", va="bottom", fontsize=6, color="red" if bad else "black")
    ax.set_ylim(H, -0.35 * H)
    ax.set_xlim(0, ctx["W"])
    ax.set_yticks([])
    ax.set_title("per-character confidence (bar height/colour); red = wrong vs ground truth", fontsize=8, loc="left")


def _overlay(ax, ctx, m, title, cmap="inferno", signed=False):
    _show_img(ax, ctx, alpha=0.9)
    if signed:
        rgba = plt.get_cmap("seismic")((m + 1.0) / 2.0)
        rgba[..., 3] = np.clip(np.abs(m), 0, 1) ** 0.8 * 0.9
    else:
        rgba = plt.get_cmap(cmap)(np.clip(m, 0, 1))
        rgba[..., 3] = np.clip(m, 0, 1) ** 0.8 * 0.9
    ax.imshow(rgba, extent=[0, ctx["W"], ctx["H"], 0], aspect="auto")
    ax.set_yticks([])
    ax.set_title(title, fontsize=8, loc="left")


def viz_gradxinput(ax, ctx):
    _overlay(ax, ctx, _smooth_norm(ctx["gradxin"], 1.5), "Gradient x Input saliency (CTC loss of the prediction)")


def viz_ig(ax, ctx):
    _overlay(ax, ctx, _smooth_norm(ctx["ig"], 1.5, signed=True), "Integrated Gradients (red = supports prediction, blue = contradicts)", signed=True)


def viz_occlusion(ax, ctx):
    _show_img(ax, ctx, alpha=0.35)
    ax2 = ax.twinx()
    ax2.plot(ctx["occ_x"], ctx["occ_d"], color="crimson", lw=1.6)
    ax2.fill_between(ctx["occ_x"], 0, ctx["occ_d"], color="crimson", alpha=0.2)
    ax2.set_ylabel("loss increase", fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(0, ctx["W"])
    ax.set_title("occlusion sensitivity (sliding window)", fontsize=8, loc="left")


def viz_char_attr(ax, ctx):
    A = ctx["char_attr"]
    if A.size == 0:
        ax.axis("off")
        return
    ax.imshow(A, aspect="auto", cmap="magma", extent=[0, ctx["W"], A.shape[0], 0], vmin=0, vmax=1)
    ax.set_yticks(np.arange(A.shape[0]) + 0.5)
    ax.set_yticklabels([chlabel(c) for c in ctx["pred_units"][:A.shape[0]]], fontsize=6)
    ax.set_title("per-character attribution: which columns drive each predicted character (Grad x Input)", fontsize=8, loc="left")


def viz_attention(ax, ctx):
    R = ctx["attn_roll"]
    if R is None or len(ctx["frames"]) == 0:
        ax.axis("off")
        return
    rows = np.stack([R[f] / max(R[f].sum(), 1e-9) for f in ctx["frames"]])
    res, sink = contrast_rollout(rows)
    for r, f in enumerate(ctx["frames"][:res.shape[0]]):
        res[r, f] = 0.0
    res = res / (res.max(1, keepdims=True) + 1e-12)
    ax.imshow(res, aspect="auto", cmap="hot", extent=[0, ctx["T"] * 4, res.shape[0], 0], vmin=0, vmax=1)
    ax.scatter([4 * f + 2 for f in ctx["frames"][:res.shape[0]]], np.arange(res.shape[0]) + 0.5, s=10, c="cyan", marker="o", linewidths=0)
    ax.set_yticks(np.arange(res.shape[0]) + 0.5)
    ax.set_yticklabels([chlabel(c) for c in ctx["pred_units"][:res.shape[0]]], fontsize=6)
    tail = "" if not np.isfinite(sink) else f"; sinks held {100 * sink:.0f}% of the raw mass"
    ax.set_title("attention rollout: context of each emission (cyan = emitting frame, shared sink removed" + tail + ")", fontsize=8, loc="left")


def viz_vq(ax, ctx):
    cm = plt.get_cmap("tab20")
    for r, (name, idx) in enumerate([("stroke codes", ctx["idx_low"]), ("component codes", ctx["idx_high"])]):
        ax.imshow(cm(idx % 20)[None], extent=[0, ctx["T"] * 4, 2 - r, 1 - r], aspect="auto")
        if ctx["T"] <= 110:
            for t, v in enumerate(idx):
                ax.text(t * 4 + 2, 1.5 - r, str(int(v)), ha="center", va="center", fontsize=4.5)
    ax.set_ylim(0, 2)
    ax.set_yticks([1.5, 0.5])
    ax.set_yticklabels(["stroke codes", "component codes"], fontsize=7)
    ax.set_xlim(0, ctx["T"] * 4)
    ax.set_title("discrete VQ codes per frame (colour = code id mod 20)", fontsize=8, loc="left")


def viz_style_attention(ax, ctx):
    """Which frames the style encoder attends to when it summarises the writer's style (uniform = 1/T)."""
    a = ctx.get("style_att")
    if a is None or len(a) == 0:
        ax.axis("off")
        return
    _show_img(ax, ctx, alpha=0.35)
    ax2 = ax.twinx()
    x = np.arange(len(a)) * 4 + 2
    ax2.plot(x, a * len(a), color="purple", lw=1.4)
    ax2.axhline(1.0, color="gray", ls=":", lw=1)
    ax2.set_ylabel("attention x T (1 = uniform)", fontsize=7)
    ax.set_yticks([])
    ax.set_xlim(0, ctx["W"])
    ax.set_title("style attention: which frames define the predicted style (dotted = uniform)", fontsize=8, loc="left")


SAMPLE_VIZ = [("input", viz_input, 1.2), ("ctc_posterior", viz_posterior, 2.0), ("char_confidence", viz_charconf, 1.4),
              ("grad_x_input", viz_gradxinput, 1.2), ("integrated_gradients", viz_ig, 1.2), ("occlusion", viz_occlusion, 1.2),
              ("char_attribution", viz_char_attr, 2.4), ("attention_rollout", viz_attention, 2.4), ("vq_codes", viz_vq, 1.0),
              ("style_attention", viz_style_attention, 1.2)]


# ---------------------------------------------------------------- batch-norm helpers (test-time adaptation)
def bn_layers(model: nn.Module) -> List[Tuple[str, MaskedBatchNorm2d]]:
    return [(n, m) for n, m in model.named_modules() if isinstance(m, MaskedBatchNorm2d)]


def snapshot_bn(model: nn.Module) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    return {n: (m.running_mean.detach().cpu().numpy().astype(np.float64), m.running_var.detach().cpu().numpy().astype(np.float64)) for n, m in bn_layers(model)}


def load_bn(model: nn.Module, state: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    with torch.no_grad():
        for n, m in bn_layers(model):
            mu, var = state[n]
            m.running_mean.copy_(torch.as_tensor(mu, dtype=m.running_mean.dtype, device=m.running_mean.device))
            m.running_var.copy_(torch.as_tensor(var, dtype=m.running_var.dtype, device=m.running_var.device))


def adapt_bn_state(src: Dict[str, Tuple[np.ndarray, np.ndarray]], acc: Dict[str, Dict[str, Any]], n_lines: int, tau: float) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Source statistics shrunk toward the statistics of a group of unlabeled lines (AdaBN with lam = n / (n + tau)); the adapted variance
    is the variance of the two-component mixture."""
    lam = n_lines / (n_lines + tau) if (n_lines + tau) > 0 else 0.0
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for n, (mu_s, v_s) in src.items():
        a = acc.get(n) or {}
        if lam <= 0.0 or a.get("n", 0.0) <= 0.0:
            out[n] = (mu_s, v_s)
            continue
        mu_c = a["s1"] / a["n"]
        v_c = np.maximum(a["s2"] / a["n"] - mu_c ** 2, 0.0)
        mu = lam * mu_c + (1.0 - lam) * mu_s
        out[n] = (mu, lam * (v_c + (mu_c - mu) ** 2) + (1.0 - lam) * (v_s + (mu_s - mu) ** 2))
    return out
