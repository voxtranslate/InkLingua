"""
Classical style descriptors: the targets of the connected "style" task.  They are MEASUREMENTS on the line image (never
invented): a descriptor that cannot be measured on a line is NaN and is masked out of every loss and metric.  They are
script-agnostic (slant, stroke width, ink density, word gap, body ratio exist for every handwriting).
"""
from __future__ import annotations

import math

import cv2
import numpy as np

STYLE_NAMES = ["slant_deg", "stroke_width", "ink_density", "gap_ratio", "body_ratio"]
NUM_DESC = len(STYLE_NAMES)
SLANT_RANGE_DEG = 45.0
STYLE_FLOORS = np.array([1.0, 0.005, 0.01, 0.01, 0.02], np.float32)   # lower bounds of the per-descriptor std


def _runs(b: np.ndarray) -> np.ndarray:
    d = np.diff(np.r_[0, b.astype(np.int8), 0])
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def estimate_slant_deg(mask: np.ndarray, max_deg: float = SLANT_RANGE_DEG, step: float = 1.0,
                       min_prominence: float = 0.01) -> float:
    """Slant (deg, > 0 = leaning right) of a binary ink mask: the shear that makes the vertical projection profile sharpest (sum of
    squared column counts).  Every candidate angle uses the SAME resampling (rows moved by whole pixels), so 0 deg enjoys no
    interpolation advantage; the optimum is refined by a parabola through the three best angles.  NaN at the border of the search
    range or when the profile energy is flat."""
    H = mask.shape[0]
    ys, xs = np.nonzero(mask)
    if ys.size < 30:
        return float("nan")
    dy = ys.astype(np.float64) - (H - 1) / 2.0
    xs = xs.astype(np.int64)
    angles = np.arange(-max_deg, max_deg + 1e-9, step)
    obj = np.empty(len(angles))
    for k, th in enumerate(angles):
        xp = xs + np.rint(math.tan(math.radians(th)) * dy).astype(np.int64)
        prof = np.bincount(xp - xp.min())
        obj[k] = float((prof.astype(np.float64) ** 2).sum())
    k = int(obj.argmax())
    med = float(np.median(obj))
    if k == 0 or k == len(angles) - 1 or (obj[k] - med) / max(med, 1e-9) < min_prominence:
        return float("nan")
    den = obj[k - 1] - 2.0 * obj[k] + obj[k + 1]
    delta = 0.0 if den >= -1e-12 else 0.5 * (obj[k - 1] - obj[k + 1]) / den
    return float(angles[k] + float(np.clip(delta, -0.5, 0.5)) * step)


def body_ratio_from_rows(rows: np.ndarray, H: int) -> float:
    """Height of the dense 'body' band (rows whose smoothed ink count reaches half of the maximum) / H."""
    s = np.convolve(rows.astype(np.float64), np.ones(3), mode="same")
    mx = s.max()
    if mx <= 0:
        return float("nan")
    return float((s >= 0.5 * mx).sum() / H)


def compute_style_descriptors(ink: np.ndarray) -> np.ndarray:
    """ink: float [H, W] in [0,1], background 0.  Returns float32 [5]: slant (deg) | stroke width / H | ink density in the text
    box | mean inter-word gap / H | body ratio.  NaN = not measurable on this line."""
    H, W = ink.shape
    out = np.full(NUM_DESC, np.nan, np.float32)
    mask = (ink > 0.45).astype(np.uint8)
    if int(mask.sum()) < 30:
        return out
    cols, rows = mask.sum(0), mask.sum(1)
    xs, ys = np.flatnonzero(cols), np.flatnonzero(rows)
    x0, x1, y0, y1 = int(xs[0]), int(xs[-1]) + 1, int(ys[0]), int(ys[-1]) + 1
    out[2] = float(mask[y0:y1, x0:x1].mean())
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    if ridge.any():
        out[1] = float(2.0 * dt[ridge].mean() / H)
    out[0] = estimate_slant_deg(mask)
    gaps = _runs(cols[x0:x1] == 0)
    gaps = gaps[gaps >= max(3, int(0.12 * H))]
    if gaps.size:
        out[3] = float(gaps.mean() / H)
    out[4] = body_ratio_from_rows(rows, H)
    return out


def kmeans_np(X: np.ndarray, k: int, iters: int = 60, seed: int = 0) -> np.ndarray:
    """Tiny k-means++ (numpy only). Returns centroids [k, D] ordered by their first coordinate (slant) -> reproducible labels."""
    rng = np.random.RandomState(seed)
    k = min(k, len(X))
    C = [X[rng.randint(len(X))]]
    for _ in range(1, k):
        d2 = np.min(((X[:, None, :] - np.stack(C)[None]) ** 2).sum(-1), axis=1)
        pr = d2 / d2.sum() if d2.sum() > 0 else np.ones(len(X)) / len(X)
        C.append(X[rng.choice(len(X), p=pr)])
    C = np.stack(C)
    for _ in range(iters):
        a = ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
        newC = np.stack([X[a == j].mean(0) if (a == j).any() else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    return C[np.argsort(C[:, 0])]


def assign_nearest(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    return ((X[:, None, :] - C[None]) ** 2).sum(-1).argmin(1)
