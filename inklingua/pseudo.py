"""
Legacy (v3-v9) LINE-level pseudo-labelling, kept as the `legacy_pl` baseline of the suite: perturbation views, the label-free
reliability features of a reading, the cross-fitted reliability model with risk-controlled thresholds ([RC-PL v2], [SHADOW],
[GUARD-REF]) and the pseudo-label bank with its gates ([PCG]).  FRAG-RC (inklingua/frag.py) replaces it by default.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .config import PseudoConfig
from .decoding import seq_confidence
from .riskcontrol import auroc, risk_control_threshold, risk_curve, theta_at_count
from .text import cer, ctc_feasible, edit_distance


# ---------------------------------------------------------------- perturbation views
def eval_noise(x: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Deterministic noise + blur perturbation (noise-consistency metric, [PCG] view 'gauss noise')."""
    g = torch.Generator().manual_seed(seed)
    n = torch.randn(x.shape, generator=g).to(x.device) * 0.06
    x = (x + n).clamp(0, 1)
    return F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="replicate"), 3, 1)


def shear_batch(x: torch.Tensor, s: float) -> torch.Tensor:
    """Horizontal shear of s pixels per pixel of height (s > 0: the writing leans further to the right)."""
    B, _, H, W = x.shape
    theta = torch.zeros(B, 2, 3, device=x.device)
    theta[:, 0, 0], theta[:, 1, 1], theta[:, 0, 1] = 1.0, 1.0, s * H / W
    return F.grid_sample(x, F.affine_grid(theta, list(x.shape), align_corners=False), align_corners=False)


PERTURBATIONS = {
    "clean": lambda x: x,
    "gauss noise": lambda x: eval_noise(x, 1),
    "blur": lambda x: F.avg_pool2d(F.pad(x, (2, 2, 2, 2), mode="replicate"), 5, 1),
    "low contrast": lambda x: x * 0.45,
    "thicker ink": lambda x: F.max_pool2d(x, 3, 1, 1),
    "thinner ink": lambda x: -F.max_pool2d(-x, 3, 1, 1),
    "shear +0.08": lambda x: shear_batch(x, 0.08),
    "shear -0.08": lambda x: shear_batch(x, -0.08),
    "shear +0.15": lambda x: shear_batch(x, 0.15),
    "shear -0.15": lambda x: shear_batch(x, -0.15),
}


def apply_perturbation(fn, x: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
    """Apply a perturbation and keep the padded columns of the batch at exactly zero."""
    col = (torch.arange(x.size(-1), device=x.device)[None] < widths.to(x.device)[:, None]).to(x.dtype)
    return fn(x) * col[:, None, None, :]


# ---------------------------------------------------------------- reliability of a reading
RELIABILITY_FEATURES = ["log mean char uncertainty", "log max char uncertainty", "share of chars below 0.9", "log mean runner-up prob.",
                        "log near-miss blank frames", "log length", "log frames per char", "log sequence uncertainty",
                        "sqrt mean view disagreement", "sqrt max view disagreement"]


def reliability_views(pc: PseudoConfig) -> Tuple[str, ...]:
    """Views the teacher reads for every line: the [PCG] gate views first, then the extra reliability views."""
    if not pc.consistency and not pc.rel_views:
        return ()
    v = list(pc.cons_views if pc.consistency else ())
    for x in pc.rel_views:
        if x not in v:
            v.append(x)
    return tuple(v)


def reliability_features(lp: np.ndarray, ids: Sequence[int], cf: Sequence[float], frames: Sequence[int], cons: Sequence[float]) -> np.ndarray:
    """Label-free description of one reading from the recogniser's own frame posteriors lp [T,V] and the CERs between the clean reading
    and perturbed readings (`cons`): where a peaky CTC model still leaks doubt."""
    T, K = lp.shape[0], len(ids)
    c = np.asarray(cf, np.float64) if K else np.ones(1)
    u = 1.0 - c
    if K:
        P = np.exp(lp[list(frames)])
        p2 = -np.sort(-P, axis=1)[:, 1]
    else:
        p2 = np.zeros(1)
    p = np.exp(lp)
    near = int(((p.argmax(1) == 0) & (p[:, 0] < 0.95)).sum())
    cons = np.asarray(list(cons), np.float64) if len(cons) else np.zeros(1)
    return np.array([np.log(u.mean() + 1e-4), np.log(u.max() + 1e-4), float((c < 0.9).mean()), np.log(p2.mean() + 1e-5), np.log1p(near),
                     np.log1p(K), np.log((T + 1.0) / (K + 1.0)), np.log(1.0 - seq_confidence(cf) + 1e-4) if K else 0.0,
                     float(np.sqrt(cons.mean())), float(np.sqrt(cons.max()))], np.float64)


class ReliabilityModel:
    """Ridge regression  reading features -> sqrt(line CER).  The output is a RISK (small = reliable); only its RANKING matters."""

    def __init__(self, lam: float = 0.1):
        self.lam = lam
        self.mu = self.sd = self.w = None
        self.b = 0.0

    def fit(self, X: np.ndarray, cer_line: np.ndarray) -> "ReliabilityModel":
        y = np.sqrt(np.clip(np.asarray(cer_line, np.float64), 0.0, 1.0))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        A = (X - self.mu) / self.sd
        self.b = float(y.mean())
        self.w = np.linalg.solve(A.T @ A + self.lam * len(A) * np.eye(A.shape[1]), A.T @ (y - self.b))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.w + self.b


class SelectionRule:
    """Reliability model(s) + thresholds fitted on the calibration lines (K-fold cross-fitting, coverage guard)."""

    def __init__(self):
        self.models: List[ReliabilityModel] = []
        self.theta: Optional[float] = None
        self.theta_llm: Optional[float] = None
        self.oof: np.ndarray = np.zeros(0)
        self.info: Dict[str, float] = {}

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([m.predict(X) for m in self.models], axis=0)

    @classmethod
    def fit(cls, X: np.ndarray, ed: np.ndarray, ref_len: np.ndarray, pc: PseudoConfig, seed: int = 0,
            prev_cov: Optional[float] = None, ref_cov: Optional[float] = None) -> "SelectionRule":
        r = cls()
        n = len(X)
        K = int(max(2, min(pc.risk_folds, n // 8)))
        cer_l = ed / np.maximum(ref_len, 1.0)
        fold = np.random.RandomState(seed).permutation(n) % K
        oof = np.zeros(n)
        for f in range(K):
            m = ReliabilityModel().fit(X[fold != f], cer_l[fold != f])
            oof[fold == f] = m.predict(X[fold == f])
            r.models.append(m)
        theta, k, c_at = risk_control_threshold(oof, ed, ref_len, pc.target_cer, pc.risk_z, pc.min_selected_calib, pc.risk_smoothing)
        theta_l, k_l, _ = risk_control_threshold(oof, ed, ref_len, pc.llm_target_cer, pc.risk_z, pc.min_selected_calib, pc.risk_smoothing)
        k_raw, guard = k, 0.0
        if pc.cov_guard and prev_cov is not None and np.isfinite(prev_cov):
            hi = int(math.floor((prev_cov + pc.max_cov_growth) * n))
            base = prev_cov if (ref_cov is None or not np.isfinite(ref_cov)) else max(prev_cov, ref_cov)
            lo = int(math.ceil(pc.min_cov_keep * base * n))
            k = int(np.clip(k, min(lo, hi), hi))
            if k > k_raw:                       # the shrink floor raised k: never above guard_cer_factor x target in calibration CER
                raw_curve = risk_curve(oof, ed, ref_len, False)["raw"]
                okk = np.flatnonzero(raw_curve[:k] <= pc.guard_cer_factor * pc.target_cer)
                k = max(k_raw, int(okk[-1]) + 1 if okk.size else 0)
            if k != k_raw:
                guard = 1.0
                theta = theta_at_count(oof, k)
                c_at = float(risk_curve(oof, ed, ref_len, False)["raw"][max(k, 1) - 1]) if k > 0 else float("nan")
        if np.isfinite(theta) and np.isfinite(theta_l):
            theta_l = max(theta_l, theta)
        elif np.isfinite(theta):
            theta_l = theta
        r.theta, r.theta_llm, r.oof = theta, theta_l, oof
        r.info = {"calib_n": float(n), "calib_cer": float(ed.sum() / max(ref_len.sum(), 1.0)), "calib_selected": float(k),
                  "calib_selected_cer": c_at, "calib_cov_raw": k_raw / max(n, 1), "calib_guard_binding": guard,
                  "calib_auroc_error_detection": auroc(oof, ed > 0), "calib_auroc_seq_conf": auroc(X[:, 7], ed > 0),
                  "calib_coverage": k / max(n, 1)}
        return r


class PseudoLabelBank:
    """Book-keeping for the unlabeled lines.  `hidden_reference` (the held-back ground truth) is read ONLY by `diagnostics()`."""

    def __init__(self, n: int, unlabeled: np.ndarray, hidden_reference: List[str], frames: np.ndarray, calib: Optional[np.ndarray] = None):
        self.n, self.unl, self.hidden = n, np.asarray(unlabeled, bool), list(hidden_reference)
        self.cal = np.zeros(n, bool) if calib is None else (np.asarray(calib, bool) & self.unl)
        self.frames = np.asarray(frames, np.int64)
        self.text = [""] * n
        self.conf = np.zeros(n, np.float32)
        self.stab = np.zeros(n, np.int32)
        self.has = np.zeros(n, bool)
        self.cons = np.full(n, np.nan, np.float32)
        self.risk = np.full(n, np.nan, np.float32)
        self.loss_ema = np.full(n, np.nan, np.float32)
        self.llm_text: List[Optional[str]] = [None] * n
        self.llm_ok = np.zeros(n, bool)
        self.sel = np.zeros(n, bool)
        self.weight = np.zeros(n, np.float32)
        self.train_text = [""] * n
        self.source = np.zeros(n, np.int8)            # 0 none, 1 recogniser, 2 LLM-rescued
        self.rounds = 0
        self._pcg_rejected = np.zeros(n, bool)
        self._pcg_accepted = np.zeros(n, bool)

    def begin_round(self) -> None:
        self.llm_text, self.llm_ok = [None] * self.n, np.zeros(self.n, bool)
        self.rounds += 1

    def update_predictions(self, idxs: Sequence[int], texts: Sequence[str], confs: Sequence[float], cons: Optional[Sequence[float]] = None,
                           risk: Optional[Sequence[float]] = None) -> None:
        for k, (i, t, c) in enumerate(zip(idxs, texts, confs)):
            if self.has[i] and self.text[i] == t:
                self.stab[i] += 1
            else:
                self.stab[i] = 0
                self.loss_ema[i] = np.nan
            self.text[i], self.conf[i], self.has[i] = t, c, True
            if cons is not None:
                self.cons[i] = cons[k]
            if risk is not None:
                self.risk[i] = risk[k]

    def update_loss(self, idxs: Sequence[int], losses: Sequence[float], alpha: float = 0.5) -> None:
        for i, l in zip(idxs, losses):
            self.loss_ema[i] = l if np.isnan(self.loss_ema[i]) else (1 - alpha) * self.loss_ema[i] + alpha * l

    def _usable(self, i: int, text: str, min_chars: int) -> bool:
        return len(text.strip()) >= min_chars and ctc_feasible(text, int(self.frames[i]))

    def select(self, pc: PseudoConfig, theta: Optional[float], theta_llm: Optional[float] = None) -> None:
        """theta = largest predicted risk that is accepted; theta_llm = the looser level up to which an LLM-rescued reading may still be
        accepted; theta None = no calibration lines -> fallback to the sequence confidence.  A line must also be stable across refreshes
        and pass the perturbation-consistency gate."""
        ok = self.unl & self.has
        usable = np.array([self._usable(i, self.text[i], pc.min_chars) if ok[i] else False for i in range(self.n)])
        gate = (np.isfinite(self.cons) & (self.cons <= pc.tau_cons)) if pc.consistency else np.ones(self.n, bool)
        if theta is None:
            reliable = self.conf >= pc.fallback_tau_conf
            rel_risk = np.clip((1.0 - self.conf) / max(1.0 - pc.fallback_tau_conf, 1e-6), 0, 1)
        else:
            reliable = np.isfinite(self.risk) & (self.risk <= theta)
            rel_risk = np.clip(np.nan_to_num(self.risk, nan=1.0) / max(theta, 1e-6), 0, 1) if np.isfinite(theta) else np.ones(self.n)
        conf_ok = ok & usable & (self.stab >= pc.min_stability) & reliable
        cand = conf_ok & gate
        self._pcg_rejected, self._pcg_accepted = conf_ok & ~gate, cand.copy()
        if pc.lops_drop > 0 and cand.sum() > 10:
            score = np.where(np.isnan(self.loss_ema), -np.log(np.clip(self.conf, 1e-6, 1)), self.loss_ema)
            ids = np.flatnonzero(cand)
            k = int(round(pc.lops_drop * len(ids)))
            if k > 0:
                cand[ids[np.argsort(-score[ids])[:k]]] = False
        llm_usable = np.array([bool(self.llm_ok[i]) and self._usable(i, self.llm_text[i] or "", pc.min_chars) for i in range(self.n)])
        llm_floor = np.zeros(self.n, bool) if theta_llm is None else (np.isfinite(self.risk) & (self.risk <= theta_llm))
        llm = ok & llm_usable & ~cand & gate & (self.stab >= pc.min_stability) & llm_floor
        self.sel = cand | llm
        w = np.clip(1.0 - 0.5 * rel_risk, pc.min_weight, 1.0)
        self.weight = np.where(llm, np.clip(w * pc.llm_weight_scale, pc.min_weight * 0.5, 1.0), w).astype(np.float32)
        self.weight[~self.sel] = 0.0
        self.source = np.where(cand, 1, np.where(llm, 2, 0)).astype(np.int8)
        self.train_text = [self.text[i] if cand[i] else (self.llm_text[i] or "") if llm[i] else "" for i in range(self.n)]

    def _corpus_cer(self, idx: Sequence[int], texts: Sequence[str]) -> float:
        e = sum(edit_distance(self.hidden[i], t) for i, t in zip(idx, texts))
        return float(e / max(sum(len(self.hidden[i]) for i in idx), 1))

    def _group_diag(self, ids: np.ndarray, prefix: str) -> Dict[str, float]:
        d: Dict[str, float] = {}
        if len(ids) == 0:
            return d
        d[prefix + "_all_cer"] = self._corpus_cer(ids, [self.text[i] for i in ids])
        d[prefix + "_stable_frac"] = float(np.mean(self.stab[ids] >= 1))
        d[prefix + "_coverage"] = float(self.sel[ids].mean())
        s = ids[self.sel[ids]]
        d[prefix + "_selected"] = float(len(s))
        if len(s):
            d[prefix + "_sel_cer"] = self._corpus_cer(s, [self.train_text[i] for i in s])
            d[prefix + "_sel_cer_line_mean"] = float(np.mean([cer(self.hidden[i], self.train_text[i]) for i in s]))
            d[prefix + "_sel_exact"] = float(np.mean([self.hidden[i] == self.train_text[i] for i in s]))
            for tag, code in (("rc", 1), ("llm", 2)):
                sub = s[self.source[s] == code]
                if len(sub):
                    d[f"{prefix}_{tag}_sel_cer"] = self._corpus_cer(sub, [self.train_text[i] for i in sub])
        return d

    def diagnostics(self) -> Dict[str, float]:
        """Pseudo-label quality against the hidden GT.  pl_* = the TRUE unlabeled lines; shadow_* = the calibration lines."""
        ok_all = self.unl & self.has
        ok = np.flatnonzero(ok_all & ~self.cal)
        if len(ok) == 0:
            return {}
        d = self._group_diag(ok, "pl")
        d["pl_pcg_refused_n"] = float(self._pcg_rejected[ok].sum())
        d["pl_llm_selected"] = float((self.source[ok] == 2).sum())
        d.update(self._group_diag(np.flatnonzero(ok_all & self.cal), "shadow"))
        if np.isfinite(self.cons[ok]).any():
            d["pl_cons_mean"] = float(np.nanmean(self.cons[ok]))
        if np.isfinite(self.risk[ok]).any():
            cer_all = np.array([cer(self.hidden[i], self.text[i]) for i in ok])
            m = np.isfinite(self.risk[ok])
            d["pl_auroc_error_detection"] = auroc(self.risk[ok][m], cer_all[m] > 0)
            conf_i = np.array([1.0 - self.conf[i] for i in ok])
            d["pl_auroc_seq_conf"] = auroc(conf_i[m], cer_all[m] > 0)
        rej, acc = np.flatnonzero(self._pcg_rejected & ~self.cal), np.flatnonzero(self._pcg_accepted & ~self.cal)
        d["pl_pcg_rejected"] = float(len(rej))
        if len(rej):
            d["pl_pcg_rejected_cer"] = self._corpus_cer(rej, [self.text[i] for i in rej])
        if len(acc):
            d["pl_pcg_accepted_cer"] = self._corpus_cer(acc, [self.text[i] for i in acc])
        return d

    def state_dict(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in ("text", "conf", "stab", "has", "cons", "risk", "loss_ema", "sel", "weight", "train_text", "source", "rounds")}

    def load_state_dict(self, sd: Dict[str, Any]) -> None:
        for k, v in sd.items():
            setattr(self, k, v)
