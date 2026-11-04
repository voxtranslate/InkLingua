"""
Empirical risk control shared by the legacy line-level pseudo-labels, FRAG-RC and the zero-shot adapter: ranking quality (AUROC),
isotonic (PAVA) risk curves, the first-violation threshold rule, the character-level rule with a line-clustered standard error,
ridge-logistic regression, and the transfer calibration of [XL-RC] (importance weights, weighted curves, anchor audit).

Everything here is a finite-sample HEURISTIC in the spirit of conformal risk control / Learn-then-Test (Angelopoulos et al.) and of
Selective Conformal Risk Control (arXiv 2512.12844), not a guarantee: the thresholds are data-dependent and the calibration sets are small.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------- ranking quality
def _ranks(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="mergesort")
    xs, r, i = x[o], np.empty(len(x)), 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return r


def auroc(score: np.ndarray, bad: np.ndarray) -> float:
    """P(score of a bad item > score of a good item); 0.5 = chance.  `score` is a risk (high = suspicious)."""
    score, bad = np.asarray(score, np.float64), np.asarray(bad, bool)
    n1, n0 = int(bad.sum()), int((~bad).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _ranks(score)
    return float((r[bad].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


# ---------------------------------------------------------------- isotonic regression and risk curves
def pava_increasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted isotonic (non-decreasing) regression, pool-adjacent-violators."""
    vals: List[float] = []
    wts: List[float] = []
    cnt: List[int] = []
    for yi, wi in zip(np.asarray(y, np.float64), np.asarray(w, np.float64)):
        vals.append(float(yi)), wts.append(float(max(wi, 1e-12))), cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            ww = wts[-2] + wts[-1]
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / ww
            c = cnt[-2] + cnt[-1]
            vals.pop(), wts.pop(), cnt.pop()
            vals[-1], wts[-1], cnt[-1] = v, ww, c
    return np.repeat(vals, cnt) if vals else np.zeros(0)


def risk_curve(score: np.ndarray, ed: np.ndarray, ref_len: np.ndarray, smooth: bool = True) -> Dict[str, np.ndarray]:
    """Cumulative corpus CER of the calibration lines taken in order of increasing predicted risk; `smooth` = after an isotonic fit of
    the local error rate along the ranking (a better score must not mean MORE errors); `se` = s.e. of the ratio estimator."""
    o = np.argsort(score, kind="stable")
    e, l = np.asarray(ed, np.float64)[o], np.asarray(ref_len, np.float64)[o]
    k = np.arange(1, len(e) + 1, dtype=np.float64)
    ce, cl = np.cumsum(e), np.cumsum(l)
    R = ce / np.maximum(cl, 1.0)
    ss = np.cumsum(e ** 2) - 2.0 * R * np.cumsum(e * l) + R ** 2 * np.cumsum(l ** 2)
    se = np.sqrt(np.maximum(ss, 0.0) * k / np.maximum(k - 1.0, 1.0)) / np.maximum(cl, 1.0)
    if smooth and len(e) > 1:
        rate = pava_increasing(e / np.maximum(l, 1.0), l)
        Rs = np.cumsum(rate * l) / np.maximum(cl, 1.0)
    else:
        Rs = R
    return {"order": o, "raw": R, "smooth": Rs, "se": se}


def risk_control_threshold(score: np.ndarray, ed: np.ndarray, ref_len: np.ndarray, alpha: float, z: float, min_n: int,
                           smooth: bool = True) -> Tuple[float, int, float]:
    """Largest set of calibration lines, in order of increasing predicted risk, whose corpus CER plus z standard errors stays <= alpha.
    Returns (theta, k, cer_at_theta); k = 0 -> theta = -inf.  smooth=True: isotonic curve and the rule stops at the FIRST k (>= min_n) at
    which the bound is violated (no winner's curse)."""
    c = risk_curve(score, ed, ref_len, smooth)
    n = len(c["raw"])
    if n < min_n or n == 0:
        return float("-inf"), 0, float("nan")
    bound = c["smooth"] + z * c["se"]
    k = np.arange(1, n + 1)
    if smooth:
        bad = np.flatnonzero((bound > alpha) & (k >= min_n))
        kk = int(bad[0]) if bad.size else n
        if kk < min_n:
            return float("-inf"), 0, float("nan")
    else:
        ok = np.flatnonzero((c["raw"] + z * c["se"] <= alpha) & (k >= min_n))
        if ok.size == 0:
            return float("-inf"), 0, float("nan")
        kk = int(ok[-1]) + 1
    return float(np.asarray(score)[c["order"]][kk - 1]), kk, float(c["raw"][kk - 1])


def theta_at_count(score: np.ndarray, k: int) -> float:
    """Risk threshold that admits the k lowest-risk calibration lines (ties admit more)."""
    if k <= 0:
        return float("-inf")
    return float(np.sort(np.asarray(score))[min(k, len(score)) - 1])


# ---------------------------------------------------------------- character-level rule (FRAG-RC)
def char_risk_threshold(risk: np.ndarray, wrong: np.ndarray, cluster: np.ndarray, alpha: float, z: float, min_n: int,
                        n_grid: int = 200) -> Tuple[float, int, float]:
    """Empirical risk control of ONE Mondrian cell at the CHARACTER level.  Characters are taken in order of increasing predicted risk;
    the local error rate is made monotone (PAVA); the bound is  smoothed rate + z x s.e.  with a CLUSTER-robust s.e. (clusters = lines:
    the errors of one line are correlated); the threshold is the last grid point BEFORE THE FIRST violation of  bound <= alpha.
    Returns (theta, k, raw error rate of the k characters); k = 0 -> theta = -inf."""
    n = len(risk)
    if n < max(min_n, 2):
        return float("-inf"), 0, float("nan")
    o = np.argsort(risk, kind="stable")
    r, y = np.asarray(risk)[o], np.asarray(wrong, np.float64)[o]
    _, gi = np.unique(np.asarray(cluster)[o], return_inverse=True)
    rate = pava_increasing(y, np.ones(n))
    cum_s = np.cumsum(rate) / np.arange(1, n + 1)
    ks = np.unique(np.linspace(min_n, n, n_grid).astype(int))
    ecl, ncl = np.zeros(gi.max() + 1), np.zeros(gi.max() + 1)
    prev, ok_k = 0, 0
    for k in ks:
        np.add.at(ecl, gi[prev:k], y[prev:k])
        np.add.at(ncl, gi[prev:k], 1.0)
        prev = k
        R = float(ecl.sum() / k)
        act = ncl > 0
        G = int(act.sum())
        var = (float(((ecl[act] - R * ncl[act]) ** 2).sum()) * G / (G - 1) / k ** 2) if G > 1 else R * (1 - R) / k
        if cum_s[k - 1] + z * math.sqrt(max(var, 0.0)) > alpha:
            break
        ok_k = int(k)
    if ok_k == 0:
        return float("-inf"), 0, float("nan")
    return float(r[ok_k - 1]), ok_k, float(y[:ok_k].mean())


# ---------------------------------------------------------------- logistic regression (IRLS)
def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def fit_logistic(Z: np.ndarray, y: np.ndarray, l2: float = 1.0, iters: int = 30) -> np.ndarray:
    """Ridge-penalised logistic regression by IRLS (intercept unpenalised); Z is standardised.  Returns [intercept, weights]."""
    n, d = Z.shape
    A = np.c_[np.ones(n), Z]
    b, R = np.zeros(d + 1), l2 * np.eye(d + 1)
    R[0, 0] = 0.0
    for _ in range(iters):
        p = sigmoid(A @ b)
        W = np.maximum(p * (1 - p), 1e-6)
        step = np.linalg.solve((A * W[:, None]).T @ A + R, A.T @ (p - y) + R @ b)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


# ---------------------------------------------------------------- [XL-RC] transfer calibration
def domain_shift(Xs: np.ndarray, Xt: np.ndarray, l2: float = 3.0, clip: float = 10.0, folds: int = 3, seed: int = 0
                 ) -> Tuple[np.ndarray, float, float]:
    """Importance weights w = p_target(x) / p_source(x) of the SOURCE calibration lines (mean 1, clipped) from a cross-fitted domain
    classifier, the AUROC of that classifier (0.5 = no detectable shift) and the effective sample size of the weights.  Valid under
    COVARIATE shift only -- the anchor audit checks that assumption."""
    ns, nt = len(Xs), len(Xt)
    X = np.r_[Xs, Xt]
    Z = (X - X.mean(0)) / (X.std(0) + 1e-6)
    y = np.r_[np.zeros(ns), np.ones(nt)]
    fold = np.random.RandomState(seed).permutation(ns + nt) % folds
    p = np.zeros(ns + nt)
    for f in range(folds):
        b = fit_logistic(Z[fold != f], y[fold != f], l2)
        p[fold == f] = sigmoid(np.c_[np.ones((fold == f).sum()), Z[fold == f]] @ b)
    p = np.clip(p, 1e-3, 1 - 1e-3)
    w = (p[:ns] / (1 - p[:ns])) / (nt / ns)
    w = np.minimum(w, clip * w.mean())
    w = w / w.mean()
    return w, auroc(p, y > 0), float(w.sum() ** 2 / np.sum(w ** 2))


def weighted_risk_curve(score: np.ndarray, ed: np.ndarray, ref_len: np.ndarray, w: np.ndarray) -> Dict[str, np.ndarray]:
    """risk_curve with importance weights: cumulative weighted CER, its isotonic version, the sandwich s.e. and the effective sample size."""
    o = np.argsort(score, kind="stable")
    e, l, ww = np.asarray(ed, np.float64)[o], np.asarray(ref_len, np.float64)[o], np.asarray(w, np.float64)[o]
    k = np.arange(1, len(e) + 1.0)
    den = np.maximum(np.cumsum(ww * l), 1e-9)
    R = np.cumsum(ww * e) / den
    rate = pava_increasing(e / np.maximum(l, 1.0), ww * l)
    Rs = np.cumsum(rate * ww * l) / den
    ss = np.cumsum(ww ** 2 * e ** 2) - 2 * R * np.cumsum(ww ** 2 * e * l) + R ** 2 * np.cumsum(ww ** 2 * l ** 2)
    se = np.sqrt(np.maximum(ss, 0.0) * k / np.maximum(k - 1.0, 1.0)) / den
    ess = np.cumsum(ww) ** 2 / np.maximum(np.cumsum(ww ** 2), 1e-12)
    return {"order": o, "raw": R, "smooth": Rs, "se": se, "ess": ess, "rate": rate}


def _transfer_theta(score_s, ed_s, rl_s, w, alpha, z, min_n, min_ess, c):
    cv = weighted_risk_curve(score_s, ed_s, rl_s, w)
    n = len(cv["raw"])
    if n < min_n:
        return float("-inf"), 0, cv
    k = np.arange(1, n + 1)
    bound = c * (cv["smooth"] + z * cv["se"])
    bad = np.flatnonzero(((bound > alpha) | (cv["ess"] < min_ess)) & (k >= min_n))
    kk = int(bad[0]) if bad.size else n
    if kk < min_n:
        return float("-inf"), 0, cv
    return float(np.asarray(score_s)[cv["order"]][kk - 1]), kk, cv


def anchor_audit(score_s: np.ndarray, cv: Dict[str, np.ndarray], score_a: np.ndarray, ed_a: np.ndarray, rl_a: np.ndarray,
                 prior_edits: float = 20.0, lo: float = 0.5, hi: float = 4.0) -> Tuple[float, float, float]:
    """Observed / predicted number of edits on the LABELED target lines (anchors), the prediction being the source error rate at the
    anchors' risk scores.  Poisson-Gamma shrinkage with `prior_edits` pseudo-edits.  > 1: the target is harder than the transferred
    calibration says (concept shift)."""
    ss = np.asarray(score_s)[cv["order"]]
    j = np.clip(np.searchsorted(ss, np.asarray(score_a), side="right") - 1, 0, len(ss) - 1)
    pred = float((cv["rate"][j] * np.asarray(rl_a, np.float64)).sum())
    obs = float(np.sum(ed_a))
    return float(np.clip((obs + prior_edits) / (pred + prior_edits), lo, hi)), pred, obs


def weighted_ks(a: np.ndarray, b: np.ndarray, wa: Optional[np.ndarray] = None) -> float:
    """Two-sample Kolmogorov-Smirnov distance (weights on the first sample)."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    wa = np.ones(len(a)) if wa is None else np.asarray(wa, np.float64)
    pts = np.sort(np.r_[a, b])
    oa = np.argsort(a)
    ca = np.cumsum(wa[oa]) / wa.sum()
    ia = np.searchsorted(a[oa], pts, side="right")
    fa = np.where(ia > 0, ca[np.maximum(ia - 1, 0)], 0.0)
    fb = np.searchsorted(np.sort(b), pts, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def transfer_risk_control(X_s: np.ndarray, score_s: np.ndarray, ed_s: np.ndarray, rl_s: np.ndarray, X_t: np.ndarray, score_t: np.ndarray,
                          alpha: float, z: float, min_n: int = 15, min_ess: float = 10.0, *, score_a: Optional[np.ndarray] = None,
                          ed_a: Optional[np.ndarray] = None, rl_a: Optional[np.ndarray] = None, zero_anchor_inflation: float = 1.5,
                          prior_edits: float = 20.0, clip: float = 10.0, l2: float = 3.0, seed: int = 0) -> Dict[str, Any]:
    """[XL-RC] Which TARGET-language lines may be pseudo-labelled when only SOURCE-language lines carry labels?  (1) covariate-shift weights
    of the source lines toward the target; (2) the weighted isotonic source curve, first violation of c*(CER + z*se) <= alpha or of the
    effective sample size; (3) the factor c from the labeled anchors (audit of the covariate-shift assumption; detects concept shift) or,
    without anchors, c = zero_anchor_inflation (a heuristic)."""
    X_s, X_t = np.asarray(X_s, np.float64), np.asarray(X_t, np.float64)
    if len(X_t) >= 10 and len(X_s) >= 20:
        w, dom_auc, ess_all = domain_shift(X_s, X_t, l2, clip, seed=seed)
    else:
        w, dom_auc, ess_all = np.ones(len(X_s)), float("nan"), float(len(X_s))
    th1, k1, cv = _transfer_theta(score_s, ed_s, rl_s, w, alpha, z, min_n, min_ess, 1.0)
    na = 0 if score_a is None else len(score_a)
    pred = obs = float("nan")
    if na > 0:
        c, pred, obs = anchor_audit(score_s, cv, score_a, ed_a, rl_a, prior_edits, lo=0.7 if na >= 60 else 1.0)
        mode = "transfer+anchors"
    else:
        c, mode = float(zero_anchor_inflation), "transfer-uncalibrated"
    th, k, cv = _transfer_theta(score_s, ed_s, rl_s, w, alpha, z, min_n, min_ess, c)
    pred_cer = float(c * cv["smooth"][k - 1]) if k > 0 else float("nan")
    return {"theta": th, "k": k, "factor": float(c), "mode": mode, "n_anchors": na, "anchor_observed_edits": obs,
            "anchor_predicted_edits": pred, "domain_auroc": dom_auc, "ess_all": ess_all, "ess_at_k": float(cv["ess"][k - 1]) if k > 0 else 0.0,
            "ks_before": weighted_ks(score_s, score_t), "ks_after": weighted_ks(score_s, score_t, w),
            "predicted_cer_selected": pred_cer, "coverage_target": float(np.mean(np.asarray(score_t) <= th)) if k > 0 else 0.0}
