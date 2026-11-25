"""CTC decoding: greedy with per-character confidences, prefix beam search N-best, the word-gap (space) bonus."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def greedy_decode(logp: np.ndarray) -> Tuple[List[int], List[float], List[int]]:
    """logp: [T, V] log-probabilities (valid frames only). Returns ids, per-char confidence (peak probability inside the
    emitting run) and the peak frame of every char."""
    if logp.shape[0] == 0:
        return [], [], []
    p = np.exp(logp)
    idx, mp = p.argmax(1), p.max(1)
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    run_ids = idx[starts]
    run_max = np.maximum.reduceat(mp, starts)
    ids, confs, frames = [], [], []
    for k in np.flatnonzero(run_ids != 0):
        ids.append(int(run_ids[k]))
        confs.append(float(run_max[k]))
        frames.append(int(starts[k] + mp[starts[k]:ends[k]].argmax()))
    return ids, confs, frames


def greedy_runs(logp: np.ndarray) -> Tuple[List[int], np.ndarray, np.ndarray]:
    """Emitted ids with the first frame and the end frame (exclusive) of every emitting run (the CTC alignment)."""
    if logp.shape[0] == 0:
        return [], np.zeros(0, np.int64), np.zeros(0, np.int64)
    idx = logp.argmax(1)
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], len(idx)]
    keep = idx[starts] != 0
    return [int(v) for v in idx[starts][keep]], starts[keep].astype(np.int64), ends[keep].astype(np.int64)


def seq_confidence(confs: Sequence[float]) -> float:
    if len(confs) == 0:
        return 0.0
    return float(np.exp(np.mean(np.log(np.clip(confs, 1e-6, 1.0)))))


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b <= -1e29:
        return a
    return a + math.log1p(math.exp(b - a))


def ctc_prefix_beam_search(logp: np.ndarray, beam: int = 5, prune: float = -7.0) -> List[Tuple[Tuple[int, ...], float]]:
    """Standard CTC prefix beam search. Returns [(ids, log_prob)] sorted best-first.  (A 5-best prefix beam, not an exact N-best.)"""
    NEG = -1e30
    T, V = logp.shape
    beams: Dict[Tuple[int, ...], Tuple[float, float]] = {(): (0.0, NEG)}
    for t in range(T):
        row = logp[t].tolist()
        cand = [c for c in range(V) if row[c] > prune] or [int(np.argmax(row))]
        nxt: Dict[Tuple[int, ...], List[float]] = {}

        def slot(k):
            if k not in nxt:
                nxt[k] = [NEG, NEG]
            return nxt[k]

        for prefix, (pb, pnb) in beams.items():
            ptot = _lse(pb, pnb)
            last = prefix[-1] if prefix else None
            for c in cand:
                p = row[c]
                if c == 0:
                    s = slot(prefix)
                    s[0] = _lse(s[0], ptot + p)
                elif c == last:
                    s = slot(prefix + (c,))
                    s[1] = _lse(s[1], pb + p)
                    s2 = slot(prefix)
                    s2[1] = _lse(s2[1], pnb + p)
                else:
                    s = slot(prefix + (c,))
                    s[1] = _lse(s[1], ptot + p)
        top = sorted(nxt.items(), key=lambda kv: -_lse(kv[1][0], kv[1][1]))[:beam]
        beams = {k: (v[0], v[1]) for k, v in top}
    out = [(k, _lse(v[0], v[1])) for k, v in beams.items()]
    out.sort(key=lambda x: -x[1])
    return out


def decode_with_space_bonus(lp: np.ndarray, space_id: Optional[int], delta: float) -> List[int]:
    """Greedy CTC decoding after adding `delta` (nats) to the space class: a one-parameter correction of the recogniser's bias
    against word gaps (v3 test run: 637 deleted vs 231 inserted spaces).  delta = 0 is plain greedy."""
    if space_id is not None and delta != 0.0:
        lp = lp.copy()
        lp[:, space_id] += delta
    return greedy_decode(lp)[0]


def log_softmax_np(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))
