"""
[FRAG-RC] Fragment-level, style-stratified, risk-controlled self-distillation (replaces line-level pseudo-labels).

Why (v9 IAM run): the line-level rule kept ~10 % of the unlabeled lines, namely those the teacher already read (CER 1.2 % vs 6.4 %
overall); the thin / tightly written hands that carry most of the errors never got in; and the rule was a knife edge because it was
calibrated on ~190 LINES.  FRAG-RC changes the unit of trust:
  * a CHARACTER, not a line: ~8,000 calibration characters instead of 194 lines -> a stable threshold;
  * a FRAGMENT of a line: the reliable stretches of a hard line still teach, the rest is left alone;
  * a Mondrian cell (corpus x hardness stratum; thin stroke / tight gaps = hard) instead of one global threshold: every cell gets its own
    threshold at the same character-error target, so hard hands and small corpora are represented instead of silently excluded;
  * soft frame targets (sharpened EMA-teacher posteriors) with an anchor on the untrusted frames, not a hard label sequence.
Evidence for a character comes from sources that fail differently: the teacher's posterior, the inter-CTC head that reads the quantised
content stream, and a second (strongly augmented) view of the same pixels read by the same teacher.

Closest precedents (all different in the unit or the guarantee): Seq-UPS, SoftCTC, Star Temporal Classification, conformal risk control
/ Mondrian conformal prediction, Selective Conformal Risk Control (2512.12844), frame-level self-distillation for CTC in speech
(2406.07909), CR-CTC.  No 2025-2026 paper found (2026-11-04) selects characters of a CTC reading with a risk-controlled stratified rule.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .riskcontrol import char_risk_threshold, fit_logistic, sigmoid
from .text import align_chars

FRAG_FEATURES = ["logit peak prob", "margin to runner-up", "inter-CTC agreement", "second-view agreement",
                 "min logit peak in the 3-char window", "log run length", "logit min blank prob in the adjacent gaps"]


def _logit_clip(p, eps: float = 1e-4):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def char_evidence(lp_w: np.ndarray, lp_s: np.ndarray, lp_aux: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray, np.ndarray]:
    """Per EMITTED character of the weak-view reading: (ids, first frame, end frame (exclusive), features [K, len(FRAG_FEATURES)]).
    lp_w / lp_s / lp_aux: [T, V] log-posteriors of the same line (valid frames only): main head on the weak view, main head on the strong
    view, inter-CTC head on the weak view.  Label-free."""
    T = lp_w.shape[0]
    nf = len(FRAG_FEATURES)
    if T == 0:
        return [], np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, nf))
    pw = np.exp(lp_w)
    idx, mp = pw.argmax(1), pw.max(1)
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], T]
    runs = [k for k in range(len(starts)) if idx[starts[k]] != 0]
    ia, is_ = np.exp(lp_aux).argmax(1), np.exp(lp_s).argmax(1)
    pb = pw[:, 0]
    ids: List[int] = []
    st, en, rows = [], [], []
    for n_, k in enumerate(runs):
        s, e = int(starts[k]), int(ends[k])
        c = int(idx[s])
        f_peak = int(s + mp[s:e].argmax())
        peak = float(mp[f_peak])
        srt = np.sort(pw[f_peak])[::-1]
        run2 = float(srt[1]) if len(srt) > 1 else 0.0
        a_prev = int(ends[runs[n_ - 1]]) if n_ > 0 else 0
        a_next = int(starts[runs[n_ + 1]]) if n_ + 1 < len(runs) else T
        gaps = [pb[a_prev:s], pb[e:a_next]]
        mb = min([float(g.min()) for g in gaps if g.size] or [1.0])
        lo, hi = max(0, s - 1), min(T, e + 1)
        rows.append([_logit_clip(peak), peak - run2, float((ia[s:e] == c).mean()), float((is_[lo:hi] == c).any()), 0.0,
                     float(np.log(max(e - s, 1))), _logit_clip(mb)])
        ids.append(c), st.append(s), en.append(e)
    Fm = np.array(rows, np.float64).reshape(-1, nf)
    if len(Fm):
        lg = Fm[:, 0]
        pad = np.r_[lg[:1], lg, lg[-1:]]
        Fm[:, 4] = np.minimum(np.minimum(pad[:-2], pad[1:-1]), pad[2:])
    return ids, np.array(st, np.int64), np.array(en, np.int64), Fm


def char_wrong_labels(ref: str, hyp: str) -> np.ndarray:
    """1 for every character of the reading that is wrong OR sits next to an omission (a reference character the reading lacks): a
    fragment bordering a missing character is not safe to learn from.  Same length as `hyp`."""
    ops = align_chars(ref, hyp)
    y = np.zeros(len(hyp), np.int8)
    jj = [o[2] for o in ops]
    for k, (op, _i, j) in enumerate(ops):
        if op in ("sub", "ins"):
            y[j] = 1
        elif op == "del":
            for q in range(k - 1, -1, -1):
                if jj[q] is not None:
                    y[jj[q]] = 1
                    break
            for q in range(k + 1, len(ops)):
                if jj[q] is not None:
                    y[jj[q]] = 1
                    break
    return y


class FragModel:
    """Ridge-logistic model  character evidence -> P(character is wrong or borders an omission).  Plain arrays only (checkpoint-safe)."""

    def __init__(self, mu: np.ndarray, sd: np.ndarray, b: np.ndarray):
        self.mu, self.sd, self.b = np.asarray(mu, np.float64), np.asarray(sd, np.float64), np.asarray(b, np.float64)

    @staticmethod
    def fit(X: np.ndarray, y: np.ndarray, l2: float = 3.0) -> "FragModel":
        mu, sd = X.mean(0), X.std(0) + 1e-6
        return FragModel(mu, sd, fit_logistic((X - mu) / sd, np.asarray(y, np.float64), l2))

    def risk(self, X: np.ndarray) -> np.ndarray:
        if len(X) == 0:
            return np.zeros(0)
        return sigmoid(self.b[0] + ((X - self.mu) / self.sd) @ self.b[1:])

    def state(self) -> Dict[str, Any]:
        return {"mu": self.mu, "sd": self.sd, "b": self.b}

    @staticmethod
    def from_state(d: Dict[str, Any]) -> "FragModel":
        return FragModel(d["mu"], d["sd"], d["b"])


def fragment_mask(T: int, st: np.ndarray, en: np.ndarray, risky: np.ndarray, guard: int = 1) -> np.ndarray:
    """Frames of the TRUSTED fragments of one line: the frames of every reliable character, the blank frames between two consecutive
    reliable characters, and the margins before the first / after the last character when it is reliable.  Frames within `guard` frames
    of an untrusted character are never trusted.  A line without any emitted character is not trusted at all."""
    K = len(st)
    trust = np.zeros(T, bool)
    if K == 0:
        return trust
    rel = ~np.asarray(risky, bool)
    for k in range(K):
        if rel[k]:
            trust[st[k]:en[k]] = True
    for k in range(K - 1):
        if rel[k] and rel[k + 1]:
            trust[en[k]:st[k + 1]] = True
    if rel[0]:
        trust[:st[0]] = True
    if rel[-1]:
        trust[en[-1]:] = True
    bad = np.zeros(T, bool)
    for k in range(K):
        if not rel[k]:
            bad[st[k]:en[k]] = True
    if guard > 0 and bad.any():
        bad = np.convolve(bad.astype(np.float64), np.ones(2 * guard + 1), mode="same") > 0
    return trust & ~bad


def cell_theta(theta: Sequence[float], cell: int) -> float:
    """Threshold of a Mondrian cell; indices beyond the rule's cells are clipped (a checkpoint applied to data with more cells)."""
    if len(theta) == 0:
        return float("-inf")
    return float(theta[int(np.clip(cell, 0, len(theta) - 1))])


def frag_trust_batch(lp_w: np.ndarray, aux_w: np.ndarray, lp_s: np.ndarray, lens: np.ndarray, strat: np.ndarray, rows: np.ndarray,
                     model: FragModel, theta: Sequence[float], guard: int) -> Tuple[np.ndarray, int, int]:
    """Trust masks [B, T] of a batch ([B, T, V] numpy log-posteriors; only the lines flagged in `rows` are processed).  Also returns the
    number of emitted characters seen and the number that were trusted."""
    B, T = lp_w.shape[:2]
    trust = np.zeros((B, T), bool)
    n_ch = n_tr = 0
    for b in range(B):
        if not rows[b]:
            continue
        L = int(lens[b])
        ids, st, en, Fm = char_evidence(lp_w[b, :L], lp_s[b, :L], aux_w[b, :L])
        if not len(ids):
            continue
        risky = ~(model.risk(Fm) <= cell_theta(theta, int(strat[b])))
        trust[b, :L] = fragment_mask(L, st, en, risky, guard)
        n_ch += len(ids)
        n_tr += int((~risky).sum())
    return trust, n_ch, n_tr


def fit_trust_rule(X: np.ndarray, y: np.ndarray, line: np.ndarray, cell: np.ndarray, n_cells: int, target: float, z: float, min_chars: int,
                   folds: int, seed: int) -> Tuple[FragModel, List[float], np.ndarray, Dict[str, float]]:
    """Cross-fitted (by line) reliability model + one threshold per Mondrian cell.  Returns (model, theta per cell, out-of-fold risks, info)."""
    rng = np.random.RandomState(seed)
    lines = np.unique(line)
    fold = dict(zip(lines.tolist(), (rng.permutation(len(lines)) % max(folds, 2)).tolist()))
    fo = np.array([fold[int(i)] for i in line])
    oof = np.zeros(len(y))
    for f_ in range(max(folds, 2)):
        te = fo == f_
        if te.all() or not te.any():
            oof[te] = 0.5
            continue
        oof[te] = FragModel.fit(X[~te], y[~te]).risk(X[te])
    model = FragModel.fit(X, y)
    theta, info = [], {}
    for s in range(n_cells):
        m_ = cell == s
        th, k, err = char_risk_threshold(oof[m_], y[m_], line[m_], target, z, min_chars)
        theta.append(th)
        info[f"frag_theta_s{s}"] = th if np.isfinite(th) else float("nan")
        info[f"frag_cal_cov_s{s}"] = float(k / max(int(m_.sum()), 1))
        info[f"frag_cal_err_sel_s{s}"] = err
        info[f"frag_cal_err_all_s{s}"] = float(y[m_].mean()) if m_.any() else float("nan")
        info[f"frag_cal_chars_s{s}"] = float(m_.sum())
    return model, theta, oof, info
