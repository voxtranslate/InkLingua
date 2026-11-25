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
import torch

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
    if guard > 0 and bad.any():                              # dilate by `guard` frames, same length (np.convolve 'same' would return
        idx = np.flatnonzero(bad)                            # max(T, 2*guard+1) samples and break lines shorter than the kernel)
        lo, hi = np.clip(idx - guard, 0, T), np.clip(idx + guard + 1, 0, T)
        d = np.zeros(T + 1, np.int64)
        np.add.at(d, lo, 1)
        np.add.at(d, hi, -1)
        bad = np.cumsum(d)[:T] > 0
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


def frag_trust_batch_t(lp_w: torch.Tensor, aux_w: torch.Tensor, lp_s: torch.Tensor, lens: torch.Tensor, strat: torch.Tensor, rows: torch.Tensor,
                       model: FragModel, theta: Sequence[float], guard: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """frag_trust_batch on the device, without a Python loop over lines or characters and without a host synchronisation: the same
    evidence (char_evidence), the same risk model and per-cell thresholds, the same fragments (fragment_mask).

    Why: the per-line NumPy version copied three [B, T, V] posteriors to the host on every training step and walked every emitted
    character in Python; on the 3-corpus Kaggle run an epoch took ~1,500 s for 1,327 steps.  Here every quantity is a run-level
    reduction over the frames of the batch (a run = maximal stretch of frames with the same arg-max symbol; an emitted character = a
    non-blank run).  Inputs: [B, T, V] log-posteriors (weak view, inter-CTC head on the weak view, strong view), lens [B], strat [B],
    rows [B] bool.  Returns (trust [B, T] bool, emitted characters, trusted characters) as device tensors."""
    dev = lp_w.device
    B, T, _V = lp_w.shape
    tt = torch.arange(T, device=dev)
    lens = lens.to(dev).long()
    valid = (tt[None, :] < lens[:, None]) & rows.to(dev).bool()[:, None]                     # [B, T]
    pw = lp_w.float().exp()
    top = pw.topk(2, dim=-1)
    mp, idx, sec = top.values[..., 0], top.indices[..., 0], top.values[..., 1]
    ia, is_, pb = aux_w.argmax(-1), lp_s.argmax(-1), pw[..., 0]
    prev = torch.cat([torch.full((B, 1), -1, dtype=idx.dtype, device=dev), idx[:, :-1]], 1)
    new_run = valid & (prev != idx)
    vf = valid.reshape(-1)
    pos = vf.nonzero().squeeze(1)                                                            # flat positions of the valid frames
    zero_b = torch.zeros(B, T, dtype=torch.bool, device=dev)
    if pos.numel() == 0:
        z = torch.zeros((), dtype=torch.long, device=dev)
        return zero_b, z, z
    rid = (new_run.reshape(-1).long().cumsum(0) - 1)[pos]                                    # run id of every valid frame
    nR = pos.numel()                                         # an upper bound of the number of runs (padding runs have length 0)
    b_f, t_f = pos // T, pos % T
    idx_f, mp_f = idx.reshape(-1)[pos], mp.reshape(-1)[pos]
    ia_f, is_f, pb_f = ia.reshape(-1)[pos], is_.reshape(-1)[pos], pb.reshape(-1)[pos]

    def red(src: torch.Tensor, how: str, init: float) -> torch.Tensor:
        out = torch.full((nR,), init, dtype=src.dtype, device=dev)
        return out.scatter_reduce(0, rid, src, reduce=how, include_self=True)
    big = torch.iinfo(torch.long).max
    start = red(t_f, "amin", big)                                                            # first frame of the run
    length = torch.zeros(nR, dtype=torch.long, device=dev).scatter_add(0, rid, torch.ones_like(rid))
    exists = length > 0                                                                      # runs beyond the last one are padding
    start = torch.where(exists, start, torch.zeros_like(start))
    line = red(b_f, "amin", big).clamp(max=B - 1)
    char = torch.zeros(nR, dtype=idx_f.dtype, device=dev).scatter(0, rid, idx_f)
    end = start + length                                                                     # exclusive
    peak = red(mp_f, "amax", -1.0)
    at_peak = mp_f == peak[rid]
    pk_t = red(torch.where(at_peak, t_f, torch.full_like(t_f, big)), "amin", big).clamp(max=T - 1)
    sec_pk = sec.reshape(-1)[line * T + pk_t]
    aux_agree = torch.zeros(nR, dtype=torch.float32, device=dev).scatter_add(0, rid, (ia_f == idx_f).float()) / length.clamp(min=1)
    inside = red((is_f == idx_f).long(), "amax", 0) > 0
    L_line = lens[line]
    left_ok = (start > 0) & (is_.reshape(-1)[line * T + (start - 1).clamp(min=0)] == char)
    right_ok = (end < L_line) & (is_.reshape(-1)[line * T + end.clamp(max=T - 1)] == char)
    view_agree = (inside | left_ok | right_ok).float()
    min_pb = red(pb_f, "amin", 2.0)
    ar = torch.arange(nR, device=dev)
    same_prev = (ar > 0) & exists & torch.cat([exists[:1] & False, exists[:-1]]) & (line == torch.cat([line[:1], line[:-1]]))
    same_next = (ar < nR - 1) & exists & torch.cat([exists[1:], exists[-1:] & False]) & (line == torch.cat([line[1:], line[-1:]]))
    emitted = exists & (char != 0)
    blank_prev = same_prev & (torch.cat([char[:1], char[:-1]]) == 0)
    blank_next = same_next & (torch.cat([char[1:], char[-1:]]) == 0)
    gap_prev = torch.where(blank_prev, torch.cat([min_pb[:1], min_pb[:-1]]), torch.full_like(min_pb, 2.0))
    gap_next = torch.where(blank_next, torch.cat([min_pb[1:], min_pb[-1:]]), torch.full_like(min_pb, 2.0))
    mb = torch.minimum(gap_prev, gap_next)
    mb = torch.where(mb > 1.5, torch.ones_like(mb), mb)                                      # no adjacent gap -> 1.0 (as in char_evidence)

    def logit(p: torch.Tensor) -> torch.Tensor:
        p = p.double().clamp(1e-4, 1.0 - 1e-4)
        return torch.log(p / (1.0 - p))
    # features of the emitted characters, in reading order (runs are ordered line by line, frame by frame)
    E = emitted.nonzero().squeeze(1)
    f0 = logit(peak[E])
    eline = line[E]
    pe = torch.cat([E[:1], E[:-1]]) if E.numel() else E
    ne = torch.cat([E[1:], E[-1:]]) if E.numel() else E
    f0_prev = torch.where(line[pe] == eline, logit(peak[pe]), f0) if E.numel() else f0
    f0_next = torch.where(line[ne] == eline, logit(peak[ne]), f0) if E.numel() else f0
    X = torch.stack([f0, (peak[E] - sec_pk[E]).double(), aux_agree[E].double(), view_agree[E].double(),
                     torch.minimum(torch.minimum(f0_prev, f0), f0_next), torch.log(length[E].clamp(min=1).double()), logit(mb[E])], 1)
    mu, sd, bb = (torch.as_tensor(a, dtype=torch.float64, device=dev) for a in (model.mu, model.sd, model.b))
    risk = torch.sigmoid(bb[0] + ((X - mu) / sd) @ bb[1:])
    if len(theta) == 0:
        th = torch.full_like(risk, float("-inf"))
    else:
        th_t = torch.as_tensor(np.asarray(theta, np.float64), device=dev)
        th = th_t[strat.to(dev).long()[eline].clamp(0, len(theta) - 1)]
    rel_e = risk <= th                                                                      # NaN risk -> not reliable
    rel = torch.zeros(nR, dtype=torch.bool, device=dev)
    rel[E] = rel_e
    rel_prev, rel_next = torch.cat([rel[:1] & False, rel[:-1]]), torch.cat([rel[1:], rel[-1:] & False])
    blank_trust = (same_prev & same_next & rel_prev & rel_next) | (~same_prev & same_next & rel_next) | (same_prev & ~same_next & rel_prev)
    run_trust = torch.where(emitted, rel, exists & (char == 0) & blank_trust)
    trust = torch.zeros(B * T, dtype=torch.bool, device=dev)
    trust[pos] = run_trust[rid]
    bad = torch.zeros(B * T, dtype=torch.float32, device=dev)
    bad[pos] = (emitted & ~rel)[rid].float()
    bad = bad.view(B, 1, T)
    if guard > 0:
        bad = torch.nn.functional.max_pool1d(bad, 2 * guard + 1, stride=1, padding=guard)
    trust = trust.view(B, T) & ~(bad.view(B, T) > 0) & valid
    return trust, emitted.sum(), rel_e.sum()


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
