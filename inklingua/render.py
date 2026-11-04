"""Closed-loop style-matched rendering of the translation ([CL-RENDER]): the rendering is MEASURED with the same descriptor code that
produced the style targets and the five controls (font size, stroke, word gap, letter spacing, shear) are corrected from the residuals."""
from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .data import font_paths, load_font
from .style import compute_style_descriptors

_FONT_PICK_CACHE: Dict[str, Tuple[Optional[str], bool]] = {}
_SCRIPT_FONT_HINTS = ("notosans", "noto", "cjk", "droid", "freesans", "lohit", "mangal", "wqy", "unifont", "arial", "dejavu", "liberation")
_DESC_SCALE = np.array([5.0, 0.01, 0.03, 0.03, 0.05])          # one unit of mismatch: slant deg, stroke / H, density, gap / H, body ratio
_DESC_DEFAULT = (0.0, 0.05, 0.1, 0.12, 0.5)


def _glyph_bitmap(font, ch: str) -> np.ndarray:
    im = Image.new("L", (96, 96), 0)
    ImageDraw.Draw(im).text((8, 8), ch, font=font, fill=255)
    return np.asarray(im)


def _font_covers(font, text: str) -> bool:
    notdef = _glyph_bitmap(font, "\U0010ffff")
    for ch in set(text):
        if ch.isspace():
            continue
        g = _glyph_bitmap(font, ch)
        if g.max() == 0 or np.array_equal(g, notdef):
            return False
    return True


def pick_font_path(text: str) -> Tuple[Optional[str], bool]:
    """A font that covers all characters of `text` (bundled DejaVu first, then installed system fonts).  Returns (path, covered)."""
    key = "".join(sorted(set(text)))
    if key in _FONT_PICK_CACHE:
        return _FONT_PICK_CACHE[key]
    cands = list(font_paths())
    try:
        from matplotlib import font_manager
        sysf = sorted(font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf"),
                      key=lambda p: next((i for i, h in enumerate(_SCRIPT_FONT_HINTS) if h in os.path.basename(p).lower()), len(_SCRIPT_FONT_HINTS)))
        cands += [p for p in sysf if p not in cands][:120]
    except Exception:
        pass
    res: Tuple[Optional[str], bool] = (cands[0] if cands else None, False)
    for p in cands:
        try:
            if _font_covers(ImageFont.truetype(p, 32), text):
                res = (p, True)
                break
        except Exception:
            continue
    _FONT_PICK_CACHE[key] = res
    return res


def _mean_stroke_px(mask: np.ndarray) -> float:
    m = (mask > 127).astype(np.uint8)
    if m.sum() < 10:
        return 0.0
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-6)
    return float(2.0 * dt[ridge].mean()) if ridge.any() else 0.0


def _text_ls(d, xy, s: str, font) -> None:
    """Draw `s` with its baseline at xy[1] (anchor 'ls'); bitmap fallback fonts cannot anchor and are placed by their top."""
    try:
        d.text(xy, s, font=font, fill=255, anchor="ls")
    except Exception:
        d.text((xy[0], xy[1] - int(0.8 * getattr(font, "size", 16))), s, font=font, fill=255)


def _baseline_y(font, Hs: int) -> int:
    """Row of the common baseline: the x-height band is centred on the canvas (every glyph on ONE baseline, so full stops stay down and
    descenders stay visible)."""
    try:
        xh = -font.getbbox("x", anchor="ls")[1]
    except Exception:
        xh = int(0.5 * getattr(font, "size", 32))
    return int(Hs // 2 + xh // 2)


def render_styled_text_open_loop(text: str, desc: np.ndarray, H: int = 64) -> Tuple[np.ndarray, bool]:
    """The v3 renderer (sets the controls once from the predicted descriptors and never looks at the result); kept as the comparison."""
    d_ = np.asarray(desc, np.float64)
    slant, sw, _dens, gap, body = [float(v) if np.isfinite(v) else dflt for v, dflt in zip(d_, _DESC_DEFAULT)]
    path, covered = pick_font_path(text)
    SC = 2
    Hs = H * SC
    size = int(np.clip(body * Hs / 0.38, 24, 140))
    font = load_font(path, size)
    gap_px = int(max(0.28 * size, gap * Hs))
    tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
    words = text.split(" ") or [""]
    boxes = [tmp.textbbox((0, 0), w, font=font) for w in words]
    Wtot = sum(b[2] - b[0] for b in boxes) + gap_px * (len(words) - 1) + 4 * size
    canvas = Image.new("L", (int(Wtot), Hs), 0)
    d, x = ImageDraw.Draw(canvas), 2 * size
    base = _baseline_y(font, Hs)
    for w, b in zip(words, boxes):
        _text_ls(d, (x - b[0], base), w, font)
        x += (b[2] - b[0]) + gap_px
    arr = np.asarray(canvas, np.uint8)
    delta = sw * Hs - _mean_stroke_px(arr)
    k = int(min(round(abs(delta)), 6 if delta > 0 else 3))
    if k >= 1:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k + 1, k + 1))
        arr = cv2.dilate(arr, ker) if delta > 0 else cv2.erode(arr, ker)
    canvas = Image.fromarray(arr)
    t = math.tan(math.radians(slant))
    canvas = canvas.transform(canvas.size, Image.AFFINE, (1, t, -t * Hs / 2, 0, 1, 0), resample=Image.BILINEAR)
    arr = np.asarray(canvas, np.float32) / 255.0
    cols = np.flatnonzero(arr.max(0) > 0.1)
    if cols.size:
        arr = arr[:, max(0, cols[0] - 8):cols[-1] + 9]
    return cv2.resize(arr, (max(8, arr.shape[1] // SC), H), interpolation=cv2.INTER_AREA), covered


def _draw_line(text: str, path: Optional[str], size: int, track: float, gap_px: int, dil: int, slant: float, Hs: int) -> np.ndarray:
    font = load_font(path, size)
    words = text.split(" ") or [""]
    widths = [sum(font.getlength(ch) + track for ch in w) for w in words]
    Wtot = int(sum(widths) + gap_px * (len(words) - 1) + 4 * size)
    canvas = Image.new("L", (max(Wtot, 16), Hs), 0)
    d, x = ImageDraw.Draw(canvas), 2.0 * size
    base = _baseline_y(font, Hs)
    for w, wd in zip(words, widths):
        for ch in w:
            if not ch.isspace():
                _text_ls(d, (x, base), ch, font)
            x += font.getlength(ch) + track
        x += gap_px
    arr = np.asarray(canvas, np.uint8)
    if dil != 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(dil) + 1, abs(dil) + 1))
        arr = cv2.dilate(arr, ker) if dil > 0 else cv2.erode(arr, ker)
    t = math.tan(math.radians(slant))
    img = Image.fromarray(arr).transform(Image.fromarray(arr).size, Image.AFFINE, (1, t, -t * Hs / 2, 0, 1, 0), resample=Image.BILINEAR)
    out = np.asarray(img, np.float32) / 255.0
    cols = np.flatnonzero(out.max(0) > 0.1)
    return out[:, max(0, cols[0] - 8 * (Hs // 64)):cols[-1] + 9 * (Hs // 64)] if cols.size else out


def style_mismatch(measured: np.ndarray, target: np.ndarray) -> float:
    """Mean scaled absolute error over the descriptors that are finite in both vectors (inf if none)."""
    m, t = np.asarray(measured, np.float64), np.asarray(target, np.float64)
    ok = np.isfinite(m) & np.isfinite(t)
    return float((np.abs(m - t)[ok] / _DESC_SCALE[ok]).mean()) if ok.any() else float("inf")


def render_styled_text_info(text: str, desc: np.ndarray, H: int = 64, iters: int = 6) -> Dict[str, Any]:
    """CLOSED-LOOP rendering: measure -> correct (font size <- body ratio, dilation <- stroke, word gap <- gap ratio, letter spacing <- density,
    shear <- slant) -> re-measure, for `iters` rounds; the iterate with the smallest mismatch is returned (never worse than round 0)."""
    tgt = np.array([float(v) if np.isfinite(v) else np.nan for v in np.asarray(desc, np.float64)])
    use = np.where(np.isfinite(tgt), tgt, np.array(_DESC_DEFAULT))
    path, covered = pick_font_path(text)
    SC = 2
    Hs = H * SC
    size = int(np.clip(use[4] * Hs / 0.38, 24, 140))
    p = {"size": float(size), "track": 0.0, "gap": float(max(0.28 * size, use[3] * Hs)), "dil": 0, "slant": float(use[0])}
    best: Optional[Dict[str, Any]] = None
    hist: List[float] = []

    def render(p_: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        arr = _draw_line(text, path, int(round(p_["size"])), p_["track"], int(round(p_["gap"])), int(p_["dil"]), p_["slant"], Hs)
        img = cv2.resize(arr, (max(8, arr.shape[1] // SC), H), interpolation=cv2.INTER_AREA)
        return img, compute_style_descriptors(np.pad(img, ((0, 0), (8, 8))))

    for r in range(max(int(iters), 0) + 1):
        img, m = render(p)
        err = style_mismatch(m, tgt)
        hist.append(err)
        if best is None or err < best["mismatch"]:
            best = {"image": img, "measured": m, "mismatch": err, "params": dict(p)}
        if r == iters:
            break
        if np.isfinite(m[4]) and np.isfinite(tgt[4]) and m[4] > 0:
            p["size"] = float(np.clip(p["size"] * (tgt[4] / m[4]) ** 0.8, 24, 140))
        if np.isfinite(m[1]) and np.isfinite(tgt[1]):
            p["dil"] = int(np.clip(p["dil"] + round((tgt[1] - m[1]) * Hs), -3, 6))
        if np.isfinite(m[3]) and np.isfinite(tgt[3]) and m[3] > 0:
            p["gap"] = float(np.clip(p["gap"] * (tgt[3] / m[3]) ** 0.9, 0.2 * p["size"], 3.0 * p["size"]))
        if np.isfinite(m[2]) and np.isfinite(tgt[2]) and m[2] > 0 and tgt[2] > 0:
            p["track"] = float(np.clip(p["track"] + 0.25 * p["size"] * math.log(m[2] / tgt[2]), -0.1 * p["size"], 0.6 * p["size"]))
        if np.isfinite(m[0]) and np.isfinite(tgt[0]):
            p["slant"] = float(np.clip(p["slant"] + (tgt[0] - m[0]), -45.0, 45.0))
    assert best is not None
    return {"image": best["image"], "covered": covered, "measured": best["measured"], "target": tgt, "mismatch": best["mismatch"],
            "mismatch_round0": hist[0], "rounds": hist, "params": best["params"]}


def render_styled_text(text: str, desc: np.ndarray, H: int = 64, iters: int = 6) -> Tuple[np.ndarray, bool]:
    info = render_styled_text_info(text, desc, H, iters)
    return info["image"], info["covered"]
