"""
Data: preprocessing, sources (Hugging Face Hub / local parquet / synthetic), the line cache, the dataset class that serves BOTH
tasks (recognition + style), the collator with MixLine, width-bucketed batching and the data module that assembles one or
several corpora.

Guarantees that a reader of the paper must be able to trust (all tested in inklingua.selftest):
  * the hidden transcriptions of the unlabeled train lines never reach a batch of the training dataset;
  * the calibration lines (labeled, text held out) are images only for the training loop; their text is read by the trust rule
    and by diagnostics;
  * the alphabet is built from the labeled lines only;
  * no image occurs in two splits (content digest), split sizes match the published ones (strict by default);
  * right-to-left corpora are stored in visual order and displayed in logical order; the metrics are unchanged by that.
"""
from __future__ import annotations

import fcntl
import glob
import hashlib
import io
import json
import math
import os
import pickle
import random
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import Config, DatasetSpec
from .datasets_registry import get_spec
from .style import NUM_DESC, STYLE_FLOORS, STYLE_NAMES, compute_style_descriptors, kmeans_np
from .text import CharCodec, ctc_min_frames, is_linguistic, normalize_text, text_baseline_r2, to_logical, to_visual
from .utils import LOG, json_safe

try:
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        # not tqdm.auto: its asyncio class calls iter() in __init__, so a DataLoader started two iterators (twice the workers,
        # one idle pair holding prefetched batches)
        from tqdm.autonotebook import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **kw):  # type: ignore
        return x


# ==============================================================================
# preprocessing
# ==============================================================================
def preprocess_line(img: Image.Image, H: int, max_w: int, min_w: int = 32, pad_x: int = 8) -> np.ndarray:
    """PIL line image -> float32 [H, W] with ink=1, background=0 (per-line contrast normalised).  Aspect ratio is preserved; a
    blank margin of `pad_x` pixels is added on both sides so that shear / rotation augmentations never crop ink."""
    g = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = g.shape
    if h < 2 or w < 2:
        raise ValueError(f"degenerate line image of size {w}x{h}")
    s = min(H / h, max(8, max_w - 2 * pad_x) / w)
    nh, nw = max(1, min(H, int(round(h * s)))), max(1, int(round(w * s)))
    g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    bg, lo = np.percentile(g, 92), np.percentile(g, 1)
    ink = np.clip((bg - g) / max(bg - lo, 25.0), 0.0, 1.0)
    canvas = np.zeros((H, max(min_w, nw + 2 * pad_x)), np.float32)
    top = (H - nh) // 2
    canvas[top:top + nh, pad_x:pad_x + nw] = ink
    return canvas


def to_u8(ink: np.ndarray) -> np.ndarray:
    return np.round(np.clip(ink, 0.0, 1.0) * 255.0).astype(np.uint8)


# ==============================================================================
# sources
# ==============================================================================
class SplitSource:
    """Uniform access to one split of one corpus, independent of where it comes from."""

    def __len__(self) -> int: raise NotImplementedError
    def image(self, j: int) -> Image.Image: raise NotImplementedError
    def text(self, j: int) -> str: raise NotImplementedError
    def extra(self, j: int) -> Optional[str]: return None


class HFSource(SplitSource):
    """One split of a Hugging Face `datasets` object with an image and a text column."""

    IMAGE_COLS = ("image", "img", "line_image")
    TEXT_COLS = ("text", "transcription", "label", "sentence", "gt")

    def __init__(self, hf_ds, style_col: Optional[str] = None, image_col: Optional[str] = None, text_col: Optional[str] = None):
        from datasets import Image as HFImage
        self.ds = hf_ds
        cols = list(hf_ds.column_names)
        self.img_col = image_col or next((c for c in self.IMAGE_COLS if c in cols), None) or next(
            (c for c in cols if isinstance(hf_ds.features[c], HFImage)), None)
        self.txt_col = text_col or next((c for c in self.TEXT_COLS if c in cols), None)
        if self.img_col is None or self.txt_col is None:
            raise ValueError(f"cannot find an image and a text column among {cols} (features: {dict(hf_ds.features)})")
        if not isinstance(hf_ds.features[self.img_col], HFImage):
            self.ds = hf_ds = hf_ds.cast_column(self.img_col, HFImage())
        self._texts = [normalize_text(str(t)) for t in hf_ds[self.txt_col]]        # text column only -> no image decoding here
        self._extra = list(hf_ds[style_col]) if style_col and style_col in cols else None

    def __len__(self): return len(self._texts)

    def image(self, j):
        im = self.ds[int(j)][self.img_col]
        if not isinstance(im, Image.Image):
            raise TypeError(f"row {j}: expected a PIL image, got {type(im)}")
        return im

    def text(self, j): return self._texts[int(j)]
    def extra(self, j): return None if self._extra is None else str(self._extra[int(j)])


class ParquetSource(SplitSource):
    """One split read straight from its parquet file(s) with pyarrow: the transcriptions are read once; images are decoded on demand, one row
    group at a time (the line cache reads the selected lines in increasing order, so every row group is read once).  Unlike the `datasets`
    library this writes no second Arrow copy of the data to disk (the 12 registry corpora are ~15 GB of parquet)."""

    IMAGE_COLS = HFSource.IMAGE_COLS
    TEXT_COLS = HFSource.TEXT_COLS

    def __init__(self, files: Sequence[str], image_col: Optional[str] = None, text_col: Optional[str] = None, style_col: Optional[str] = None):
        import pyarrow.parquet as pq
        self.paths = list(files)
        self.files = [pq.ParquetFile(f) for f in self.paths]
        names = self.files[0].schema_arrow.names
        self.img_col = image_col or next((c for c in self.IMAGE_COLS if c in names), None)
        self.txt_col = text_col or next((c for c in self.TEXT_COLS if c in names), None)
        if self.img_col is None or self.txt_col is None:
            raise ValueError(f"cannot find an image and a text column among {names} in {self.paths[0]}")
        self._rg: List[Tuple[int, int, int, int]] = []
        texts: List[Any] = []
        extra: List[Any] = []
        start = 0
        for fi, pf in enumerate(self.files):
            for rg in range(pf.num_row_groups):
                n = pf.metadata.row_group(rg).num_rows
                self._rg.append((fi, rg, start, n))
                start += n
            texts += pf.read(columns=[self.txt_col]).column(0).to_pylist()
            if style_col and style_col in names:
                extra += pf.read(columns=[style_col]).column(0).to_pylist()
        self._texts = [normalize_text(str(t)) for t in texts]
        self._extra = extra if extra else None
        self._starts = np.array([r[2] for r in self._rg], np.int64)
        self._cur: Tuple[Optional[int], Optional[List[Any]]] = (None, None)

    def __len__(self): return len(self._texts)

    def image(self, j):
        j = int(j)
        k = int(np.searchsorted(self._starts, j, side="right") - 1)
        fi, rg, start, n = self._rg[k]
        if self._cur[0] != k:
            self._cur = (k, self.files[fi].read_row_group(rg, columns=[self.img_col]).column(0).to_pylist())
        v = self._cur[1][j - start]
        if isinstance(v, dict):
            if v.get("bytes") is not None:
                return Image.open(io.BytesIO(v["bytes"]))
            if v.get("path"):
                return Image.open(v["path"])
        if isinstance(v, (bytes, bytearray)):
            return Image.open(io.BytesIO(v))
        raise TypeError(f"row {j}: cannot decode an image from {type(v)}")

    def text(self, j): return self._texts[int(j)]
    def extra(self, j): return None if self._extra is None else str(self._extra[int(j)])


_SYN_WORDS = ("the of and to in is that it was for on are with as his they be at one have this from or had by "
              "hot word but what some we can out other were all there when up use your how said an each she "
              "which do their time if will way about many then them write would like so these her long make "
              "thing see him two has look more day could go come did number sound no most people my over know "
              "water than call first who may down side been now find head stand own page should country found "
              "answer school grow study still learn plant cover food sun four between state keep eye never last "
              "let thought city tree cross farm hard start might story saw far sea draw left late run").split()


def font_paths() -> List[str]:
    try:
        import matplotlib as mpl
        d = os.path.join(mpl.get_data_path(), "fonts", "ttf")
        names = ["DejaVuSans.ttf", "DejaVuSerif.ttf", "DejaVuSansMono.ttf", "DejaVuSans-Bold.ttf",
                 "DejaVuSans-Oblique.ttf", "DejaVuSerif-Italic.ttf"]
        return [os.path.join(d, n) for n in names if os.path.exists(os.path.join(d, n))]
    except Exception:
        return []


def load_font(path: Optional[str], size: int):
    try:
        if path:
            return ImageFont.truetype(path, size)
        return ImageFont.load_default(size=size)
    except Exception:
        return ImageFont.load_default()


def render_synthetic_line(text: str, rng: random.Random, fonts: List[str]) -> Image.Image:
    """Offline 'handwriting': varied fonts / slant / stroke / blur / noise (for smoke tests only)."""
    font = load_font(rng.choice(fonts) if fonts else None, rng.randint(40, 56))
    stroke = rng.choice([0, 0, 1, 2])
    ink = int(rng.uniform(0, 90))
    tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font, stroke_width=stroke)
    w = (x1 - x0) + 60
    canvas = Image.new("L", (w, 128), 255)
    ImageDraw.Draw(canvas).text((30 - x0, 64 - (y1 - y0) // 2 - y0), text, font=font, fill=ink, stroke_width=stroke, stroke_fill=ink)
    shear = rng.uniform(-0.35, 0.35)
    canvas = canvas.transform(canvas.size, Image.AFFINE, (1, shear, -shear * 64, 0, 1, 0), resample=Image.BILINEAR, fillcolor=255)
    arr = np.asarray(canvas, np.float32)
    arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.3, 1.1))
    arr = np.clip(arr + np.random.RandomState(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(1, 6), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


class SyntheticSource(SplitSource):
    def __init__(self, n: int, seed: int, words: Tuple[int, int] = (3, 6)):
        rng, fonts = random.Random(seed), font_paths()
        self._texts = [" ".join(rng.choice(_SYN_WORDS) for _ in range(rng.randint(*words))) for _ in range(n)]
        self._imgs = [render_synthetic_line(t, rng, fonts) for t in self._texts]

    def __len__(self): return len(self._texts)
    def image(self, j): return self._imgs[int(j)]
    def text(self, j): return self._texts[int(j)]


class ListSource(SplitSource):
    """Tiny in-memory split rendered from a list of texts (self-tests)."""

    def __init__(self, texts: List[str], seed: int = 0):
        rng, fonts = random.Random(seed), font_paths()
        self._t = list(texts)
        self._im = [render_synthetic_line(t, rng, fonts) for t in texts]

    def __len__(self): return len(self._t)
    def image(self, j): return self._im[int(j)]
    def text(self, j): return self._t[int(j)]


# ==============================================================================
# the line cache
# ==============================================================================
class InkStore:
    """Read-only sequence of preprocessed uint8 lines (H x w_i) kept in flat byte segments: a memory-mapped file of the disk line cache
    (the operating system's page cache is then shared by the two parallel runs and by every data-loader worker: no run holds a private
    copy of a corpus, and forked workers do not duplicate it copy-on-write) or an in-memory array.  `ref[i] = (segment, line)`, so
    subsets and concatenations are index operations that copy no pixels.  Pickling keeps file paths, never pixel data."""

    def __init__(self, segs: List[Dict[str, Any]], ref: np.ndarray):
        self.segs, self.ref = segs, np.asarray(ref, np.int64).reshape(-1, 2)
        self._maps: Dict[int, np.ndarray] = {}

    @staticmethod
    def from_arrays(arrs: Sequence[np.ndarray]) -> "InkStore":
        arrs = [np.ascontiguousarray(a, np.uint8) for a in arrs]
        off = np.zeros(len(arrs) + 1, np.int64)
        off[1:] = np.cumsum([a.size for a in arrs])
        flat = np.concatenate([a.ravel() for a in arrs]) if arrs else np.zeros(0, np.uint8)
        seg = {"path": None, "data": flat, "off": off, "h": np.array([a.shape[0] for a in arrs], np.int64),
               "w": np.array([a.shape[1] for a in arrs], np.int64)}
        return InkStore([seg], np.stack([np.zeros(len(arrs), np.int64), np.arange(len(arrs))], 1))

    @staticmethod
    def from_file(path: str, off: np.ndarray, h: np.ndarray, w: np.ndarray) -> "InkStore":
        seg = {"path": path, "data": None, "off": np.asarray(off, np.int64), "h": np.asarray(h, np.int64), "w": np.asarray(w, np.int64)}
        return InkStore([seg], np.stack([np.zeros(len(w), np.int64), np.arange(len(w))], 1))

    def _flat(self, s: int) -> np.ndarray:
        seg = self.segs[s]
        if seg["data"] is not None:
            return seg["data"]
        m = self._maps.get(s)
        if m is None:
            m = self._maps[s] = (np.memmap(seg["path"], np.uint8, mode="r") if int(seg["off"][-1]) > 0 else np.zeros(0, np.uint8))
        return m

    def __len__(self) -> int:
        return len(self.ref)

    def __getitem__(self, i: int) -> np.ndarray:
        s, j = (int(v) for v in self.ref[int(i)])
        seg = self.segs[s]
        a, b = int(seg["off"][j]), int(seg["off"][j + 1])
        return self._flat(s)[a:b].reshape(int(seg["h"][j]), int(seg["w"][j]))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def subset(self, keep: Sequence[int]) -> "InkStore":
        return InkStore(self.segs, self.ref[np.asarray(keep, np.int64)])

    @staticmethod
    def concat(stores: Sequence["InkStore"]) -> "InkStore":
        segs: List[Dict[str, Any]] = []
        refs = []
        for st in stores:
            r = st.ref.copy()
            r[:, 0] += len(segs)
            segs += st.segs
            refs.append(r)
        return InkStore(segs, np.concatenate(refs) if refs else np.zeros((0, 2), np.int64))

    def nbytes(self) -> int:
        return int(sum(int(s["off"][-1]) for s in self.segs))

    def __getstate__(self):
        return {"segs": self.segs, "ref": self.ref}

    def __setstate__(self, st):
        self.segs, self.ref, self._maps = st["segs"], st["ref"], {}


def build_line_cache(src: SplitSource, indices: np.ndarray, H: int, max_w: int, min_w: int, pad_x: int, desc: str,
                     keep_ink: bool = True, ink_path: Optional[str] = None) -> Dict[str, Any]:
    """One decoding pass over the selected lines: preprocessed uint8 ink (optional), widths, style descriptors and a content digest
    (used to prove that no image occurs in two splits).  A corrupt image raises immediately -- lines are never skipped silently.
    With `ink_path` the pixels are streamed to that file (the RAM never holds the corpus) and returned as a memory-mapped InkStore."""
    n = len(indices)
    width, style = np.zeros(n, np.int64), np.zeros((n, NUM_DESC), np.float32)
    heights, off = np.zeros(n, np.int64), np.zeros(n + 1, np.int64)
    arrs: List[np.ndarray] = []
    digest: List[str] = []
    fh = open(ink_path, "wb") if (keep_ink and ink_path) else None
    try:
        for k, j in enumerate(tqdm(indices, desc=desc, leave=False)):
            try:
                u8 = np.ascontiguousarray(to_u8(preprocess_line(src.image(int(j)), H, max_w, min_w, pad_x)), np.uint8)
            except Exception as e:
                raise RuntimeError(f"{desc}: line {int(j)} could not be decoded: {e}") from e
            heights[k], width[k] = u8.shape
            off[k + 1] = off[k] + u8.size
            style[k] = compute_style_descriptors(u8.astype(np.float32) / 255.0)
            digest.append(hashlib.blake2b(u8.tobytes(), digest_size=8).hexdigest())
            if fh is not None:
                fh.write(u8.tobytes())
            elif keep_ink:
                arrs.append(u8)
    finally:
        if fh is not None:
            fh.close()
    if not keep_ink:
        ink: Any = [None] * n
    elif ink_path:
        ink = InkStore.from_file(ink_path, off, heights, width)
    else:
        ink = InkStore.from_arrays(arrs)
    return {"ink": ink, "width": width, "style": style, "digest": digest}


def cached_line_cache(cache_dir: Optional[str], key: Dict[str, Any], build) -> Dict[str, Any]:
    """Disk cache of build_line_cache, shared by every run that reads the same lines with the same preprocessing (a corpus is decoded once
    per session instead of once per run).  `key` must identify the source, the selected lines and the preprocessing; a file lock keeps two
    parallel runs from building the same entry twice; writes are atomic.  `build(ink_path)` streams the pixels to `<entry>.ink` (memory-
    mapped by every reader); the metadata (widths, styles, digests, offsets) is a small pickle."""
    if not cache_dir:
        return build(None)
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:20]
    path = os.path.join(cache_dir, f"{key.get('corpus', 'lines')}_{key.get('split', '')}_{h}.pkl")
    ink_path = path[:-4] + ".ink"
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            if os.path.exists(path) and (os.path.exists(ink_path) or not key.get("ink", True)):
                with open(path, "rb") as f:
                    meta = pickle.load(f)
                if meta.get("ink_off") is not None:
                    meta["ink"] = InkStore.from_file(ink_path, meta.pop("ink_off"), meta.pop("ink_h"), meta["width"])
                else:
                    meta["ink"] = [None] * len(meta["width"])
                return meta
            out = build(ink_path + ".tmp")
            meta = {k: v for k, v in out.items() if k != "ink"}
            if isinstance(out["ink"], InkStore):
                seg = out["ink"].segs[0]
                meta["ink_off"], meta["ink_h"] = seg["off"], seg["h"]
                os.replace(ink_path + ".tmp", ink_path)
                out["ink"] = InkStore.from_file(ink_path, seg["off"], seg["h"], seg["w"])
            else:
                meta["ink_off"] = meta["ink_h"] = None
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, path)
            return out
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def _ink_concat(parts: List[Any]) -> Any:
    if any(isinstance(p, InkStore) for p in parts):
        return InkStore.concat([p if isinstance(p, InkStore) else InkStore.from_arrays(list(p)) for p in parts])
    return sum((list(p) for p in parts), [])


def _ink_subset(ink: Any, keep: np.ndarray) -> Any:
    return ink.subset(keep) if isinstance(ink, InkStore) else [ink[i] for i in keep]


def _concat_caches(caches: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not caches:
        return {"ink": [], "width": np.zeros(0, np.int64), "style": np.zeros((0, NUM_DESC), np.float32), "digest": []}
    return {"ink": _ink_concat([c["ink"] for c in caches]), "width": np.concatenate([c["width"] for c in caches]),
            "style": np.concatenate([c["style"] for c in caches]), "digest": sum((c["digest"] for c in caches), [])}


def _subset_cache(cache: Dict[str, Any], keep: np.ndarray) -> Dict[str, Any]:
    return {"ink": _ink_subset(cache["ink"], keep), "width": cache["width"][keep], "style": cache["style"][keep],
            "digest": [cache["digest"][i] for i in keep]}


# ==============================================================================
# the dataset class
# ==============================================================================
class HandwritingDataset(Dataset):
    """One item serves BOTH tasks: the recognition target `text` (+ per-sample CTC `weight`; 0 for lines without a usable label) and
    the style targets (5 standardised descriptors + style cluster id).  `oracle` holds the transcription of EVERY line in VISUAL
    order; for the unlabeled lines of the train split it is HIDDEN: returned by `reference()` / in batches only if `expose_hidden`,
    which is never the case for the training dataset.  `domain[i]` is the corpus of line i, `rtl[i]` its writing direction."""

    def __init__(self, sources: List[SplitSource], src_id: np.ndarray, indices: np.ndarray, ink: Optional[List[np.ndarray]],
                 width: np.ndarray, style: np.ndarray, style_ok: np.ndarray, style_raw: np.ndarray, cluster: np.ndarray,
                 oracle: List[str], labeled: np.ndarray, calib: Optional[np.ndarray], domain: np.ndarray, domain_specs: List[DatasetSpec],
                 img_h: int, max_w: int, min_w: int, pad_x: int, name: str, expose_hidden: bool = False):
        self.sources, self.src_id, self.indices, self.ink = sources, np.asarray(src_id), np.asarray(indices), ink
        self.width, self.style, self.cluster = width, style.astype(np.float32), cluster.astype(np.int64)
        self.style_ok, self.style_raw = np.asarray(style_ok, bool), np.asarray(style_raw, np.float32)
        self.oracle, self.labeled = list(oracle), np.asarray(labeled, bool)
        self.calib = np.zeros(len(self.oracle), bool) if calib is None else np.asarray(calib, bool)
        self.domain, self.domain_specs = np.asarray(domain, np.int64), list(domain_specs)
        self.rtl = np.array([self.domain_specs[d].rtl for d in self.domain], bool) if len(self.domain) else np.zeros(0, bool)
        self.img_h, self.max_w, self.min_w, self.pad_x, self.name = img_h, max_w, min_w, pad_x, name
        self.expose_hidden = expose_hidden
        self.tgt_text = [t if l else "" for t, l in zip(self.oracle, self.labeled)]
        self.tgt_weight = np.where(self.labeled, 1.0, 0.0).astype(np.float32)
        self.stratum = np.zeros(len(self.oracle), np.int64)       # difficulty cell (set by the data module)

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def domain_names(self) -> List[str]:
        return [s.key for s in self.domain_specs]

    def reference(self, i: int) -> str:
        """Ground-truth transcription of line i (visual order), or "" if it is hidden."""
        return self.oracle[i] if (self.expose_hidden or self.labeled[i] or self.calib[i]) else ""

    def display(self, i: int, text: Optional[str] = None) -> str:
        """Human-readable (logical-order) form of a visual-order string of line i (default: its reference)."""
        return to_logical(self.oracle[i] if text is None else text, bool(self.rtl[i]))

    def detok(self, i: int) -> bool:
        return bool(self.domain_specs[int(self.domain[i])].detokenize)

    def language(self, i: int) -> str:
        return self.domain_specs[int(self.domain[i])].language

    def set_pseudo(self, texts: Sequence[str], weights: np.ndarray, include_calib: bool = False) -> None:
        """Install pseudo-labels for the unlabeled lines (labeled lines are never overwritten).  Calibration lines receive
        pseudo-labels only with include_calib=True ([SHADOW]); their true text is never a training target in either case."""
        for i in np.flatnonzero(~self.labeled if include_calib else (~self.labeled & ~self.calib)):
            self.tgt_text[i], self.tgt_weight[i] = texts[i], float(weights[i])

    def get_ink(self, i: int) -> np.ndarray:
        if self.ink is not None:
            return self.ink[i].astype(np.float32) / 255.0
        src = self.sources[int(self.src_id[i])]
        return to_u8(preprocess_line(src.image(int(self.indices[i])), self.img_h, self.max_w, self.min_w, self.pad_x)).astype(np.float32) / 255.0

    def __getitem__(self, i: int) -> Dict[str, Any]:
        ink = self.get_ink(i)
        return {"index": i, "image": torch.from_numpy(ink)[None], "width": ink.shape[1],
                "text": self.tgt_text[i], "weight": float(self.tgt_weight[i]),
                "oracle": self.oracle[i] if (self.expose_hidden or self.labeled[i]) else "",     # calibration lines stay hidden here
                "style": torch.from_numpy(self.style[i]), "style_ok": torch.from_numpy(self.style_ok[i]),
                "cluster": int(self.cluster[i]), "labeled": bool(self.labeled[i]), "stratum": int(self.stratum[i]),
                "domain": int(self.domain[i])}


# ==============================================================================
# collation (+ MixLine) and batching
# ==============================================================================
class Collator:
    def __init__(self, codec: CharCodec, pad_multiple: int = 4, mix_prob: float = 0.0, max_w: int = 2560,
                 gap: Tuple[int, int] = (0, 24), seed: int = 0, nospace_domains: Sequence[int] = ()):
        self.codec, self.pm = codec, pad_multiple
        self.nospace = frozenset(int(d) for d in nospace_domains)       # corpora written without word spaces (Han): lines are juxtaposed
        self.mix_prob, self.max_w, self.gap, self.seed = float(mix_prob), int(max_w), tuple(gap), int(seed)

    def _mix(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """[MIXLINE] concatenate two adjacent LABELED lines of the SAME corpus (real images, exact label 'A B').  The style targets of a
        concatenation are masked (they describe one line), the style class is ignored.  Visual order is kept, so RTL lines concatenate
        correctly (the label of the right-hand image part comes last)."""
        rng = np.random.RandomState((int(sum(int(it["index"]) for it in items)) * 2654435761 + self.seed) % (2 ** 32))
        out: List[Dict[str, Any]] = []
        space = " " if " " in self.codec.c2i else ""
        k = 0
        while k < len(items):
            a = items[k]
            b = items[k + 1] if k + 1 < len(items) else None
            if (b is not None and a["labeled"] and b["labeled"] and a["weight"] > 0 and b["weight"] > 0 and a["text"] and b["text"]
                    and a.get("domain", 0) == b.get("domain", 0) and rng.rand() < self.mix_prob):
                g = int(rng.randint(self.gap[0], self.gap[1] + 1))
                sep = "" if int(a.get("domain", 0)) in self.nospace else space
                w = int(a["width"]) + g + int(b["width"])
                if w <= self.max_w:
                    img = torch.zeros(1, a["image"].shape[1], w)
                    img[:, :, :a["width"]] = a["image"][:, :, :a["width"]]
                    img[:, :, a["width"] + g:w] = b["image"][:, :, :b["width"]]
                    m = dict(a)
                    m.update(image=img, width=w, text=a["text"] + sep + b["text"], oracle=a["oracle"] + sep + b["oracle"],
                             style_ok=torch.zeros_like(a["style_ok"]), cluster=-100)
                    out.append(m)
                    k += 2
                    continue
            out.append(a)
            k += 1
        return out

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.mix_prob > 0 and len(items) > 1:
            items = self._mix(items)
        B, H = len(items), items[0]["image"].shape[1]
        Wm = int(math.ceil(max(it["width"] for it in items) / self.pm) * self.pm)
        imgs = torch.zeros(B, 1, H, Wm)
        tgt, tlen, minw = [], [], []
        for b, it in enumerate(items):
            imgs[b, :, :, :it["width"]] = it["image"]
            ids = self.codec.encode(it["text"]) if it["weight"] > 0 else []
            tgt += ids
            tlen.append(len(ids))
            minw.append(4 * ctc_min_frames(ids) if ids else 0)      # the horizontal-scale augmentation never goes below it
        return {"image": imgs, "width": torch.tensor([it["width"] for it in items]),
                "target": torch.tensor(tgt, dtype=torch.long), "target_len": torch.tensor(tlen, dtype=torch.long),
                "weight": torch.tensor([it["weight"] if tl > 0 else 0.0 for it, tl in zip(items, tlen)]),
                "style": torch.stack([it["style"] for it in items]),
                "style_ok": torch.stack([it["style_ok"] for it in items]),
                "min_width": torch.tensor(minw, dtype=torch.long),
                "cluster": torch.tensor([it["cluster"] for it in items], dtype=torch.long),
                "labeled": torch.tensor([it["labeled"] for it in items]),
                "index": torch.tensor([it["index"] for it in items], dtype=torch.long),
                "stratum": torch.tensor([it.get("stratum", 0) for it in items], dtype=torch.long),
                "domain": torch.tensor([it.get("domain", 0) for it in items], dtype=torch.long),
                "text": [it["text"] for it in items], "oracle": [it["oracle"] for it in items]}


class BucketBatchSampler(Sampler):
    """Batches lines of similar width together (less padding). Deterministic per (seed, epoch)."""

    def __init__(self, lengths: np.ndarray, batch_size: int, shuffle: bool, mega: int = 50, seed: int = 0,
                 indices: Optional[Sequence[int]] = None):
        self.lengths = np.asarray(lengths)
        self.indices = np.arange(len(lengths)) if indices is None else np.asarray(indices)
        self.bs, self.shuffle, self.mega, self.seed, self.epoch = batch_size, shuffle, mega, seed, 0

    def set_epoch(self, e: int) -> None:
        self.epoch = e

    def __len__(self) -> int:
        return int(math.ceil(len(self.indices) / self.bs))

    def __iter__(self):
        idx = self.indices.copy()
        rng = np.random.RandomState(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
            chunks = [idx[i:i + self.bs * self.mega] for i in range(0, len(idx), self.bs * self.mega)]
            idx = np.concatenate([c[np.argsort(-self.lengths[c], kind="stable")] for c in chunks])
        else:
            idx = idx[np.argsort(-self.lengths[idx], kind="stable")]
        batches = [idx[i:i + self.bs].tolist() for i in range(0, len(idx), self.bs)]
        if self.shuffle:
            rng.shuffle(batches)
        yield from batches


# ==============================================================================
# the data module
# ==============================================================================
class HandwritingDataModule:
    """Downloads (Hub) or reads (local parquet) one or several corpora, checks their integrity, splits them, builds datasets and
    loaders.  There is NO fallback to synthetic data: if the real data cannot be obtained the run stops."""

    def __init__(self, cfg: Config, base_units: Optional[Sequence[str]] = None):
        """base_units: alphabet of a checkpoint to warm-start from (mode "finetune"): the new alphabet EXTENDS it (old indices keep their
        meaning, new units are appended) so that the checkpoint's output layer can be grown function-preservingly."""
        self.cfg, self.dc = cfg, cfg.data
        self.base_units = list(base_units) if base_units else None
        self.specs: List[DatasetSpec] = [get_spec(k, self.dc.custom_sources) for k in self.dc.sources]
        if not self.specs:
            raise ValueError("cfg.data.sources is empty")
        self.codec: Optional[CharCodec] = None
        self.style_mean = np.zeros(NUM_DESC, np.float32)
        self.style_std = np.ones(NUM_DESC, np.float32)
        self.n_clusters = self.dc.style_num_clusters
        self.centroids: Optional[np.ndarray] = None
        self.train_ds = self.val_ds = self.test_ds = None
        self.text_baseline_r2 = np.zeros(NUM_DESC, np.float32)
        self.leak_active = np.zeros(NUM_DESC, bool)
        self.labeled_idx = self.calib_idx = self.unlabeled_idx = self.train_eval_idx = np.zeros(0, np.int64)
        self.n_strata, self.strata_cuts = 1, {}            # strata_cuts: {domain: cut points}
        self.val_sel_idx = self.val_gate_idx = np.zeros(0, np.int64)
        self.using_synthetic = False
        self.provenance: Dict[str, Any] = {}

    @property
    def is_synthetic(self) -> bool:
        return self.using_synthetic or self.dc.source == "synthetic"

    @property
    def domain_names(self) -> List[str]:
        return [s.key for s in self.specs]

    @property
    def primary_language(self) -> str:
        return self.specs[0].language

    # ---------------------------------------------------------------- loading
    def _download(self, load_dataset, spec: DatasetSpec):
        err = None
        for attempt in range(3):
            try:
                LOG.info("Downloading / loading '%s' (revision %s, attempt %d) ...", spec.hf_dataset, spec.hf_revision or "latest", attempt + 1)
                return load_dataset(spec.hf_dataset, cache_dir=os.path.join(self.cfg.data_dir, "hf", spec.key), revision=spec.hf_revision)
            except Exception as e:  # network / hub errors
                err = e
                LOG.warning("attempt %d failed: %s", attempt + 1, e)
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"could not obtain '{spec.hf_dataset}' ({err}).\nCheck the internet connection (on Kaggle: Settings -> "
                           "Internet on), or download the parquet files once and set DatasetSpec.local_dir / cfg.data.local_dir.  "
                           "There is deliberately NO synthetic fallback.")

    @staticmethod
    def _split_files(d: str, spec: DatasetSpec) -> Dict[str, List[str]]:
        """{split name: parquet files} under folder d, for the train / validation / test split names of the spec (with common aliases)."""
        aliases = {spec.train_split: (spec.train_split, "train"), spec.val_split: (spec.val_split, "validation", "val", "valid", "dev")}
        if spec.test_split:
            aliases[spec.test_split] = (spec.test_split, "test")
        files: Dict[str, List[str]] = {}
        for split, al in aliases.items():
            f: List[str] = []
            for a in dict.fromkeys(al):
                f += glob.glob(os.path.join(d, f"{a}*.parquet")) + glob.glob(os.path.join(d, "**", f"{a}*.parquet"), recursive=True)
            if not f:
                raise FileNotFoundError(f"no '{split}*.parquet' file found under {d}")
            files[split] = sorted(set(f))
        return files

    def _load_parquet(self, spec: DatasetSpec, local: Optional[str]) -> Dict[str, Optional[SplitSource]]:
        """Parquet reader path: a local folder, or the Hub snapshot of the pinned revision (parquet files only)."""
        if local:
            root, commit = local, None
        else:
            from huggingface_hub import snapshot_download
            err = None
            for attempt in range(3):
                try:
                    LOG.info("Downloading '%s' (revision %s, parquet only, attempt %d) ...", spec.hf_dataset, spec.hf_revision or "latest", attempt + 1)
                    root = snapshot_download(repo_id=spec.hf_dataset, repo_type="dataset", revision=spec.hf_revision, allow_patterns=["*.parquet"],
                                             cache_dir=os.path.join(self.cfg.data_dir, "hub", spec.key))
                    break
                except Exception as e:
                    err = e
                    LOG.warning("attempt %d failed: %s", attempt + 1, e)
                    time.sleep(5 * (attempt + 1))
            else:
                raise RuntimeError(f"could not obtain '{spec.hf_dataset}' ({err}).  Check the internet connection (Kaggle: Settings -> Internet on) "
                                   "or set DatasetSpec.local_dir.  There is deliberately NO synthetic fallback.")
            commit = os.path.basename(os.path.normpath(root)) if "snapshots" in root else None
        files = self._split_files(root, spec)
        col = self.dc.style_label_column
        mk = lambda split: ParquetSource(files[split], spec.image_col, spec.text_col, col)  # noqa: E731
        out = {"train": mk(spec.train_split), "val": mk(spec.val_split), "test": mk(spec.test_split) if spec.test_split else None}
        raw = [len(out["train"]), len(out["val"])] + ([len(out["test"])] if out["test"] is not None else [])
        prov = {"dataset": spec.hf_dataset or local, "requested_revision": spec.hf_revision, "resolved_commit": commit, "raw_sizes": raw,
                "language": spec.language, "script": spec.script, "era": spec.era, "rtl": spec.rtl, "licence": spec.licence, "local": bool(local),
                "reader": "parquet", "files": {k: [os.path.basename(f) for f in v] for k, v in files.items()}}
        if commit and spec.hf_revision and commit != spec.hf_revision:
            LOG.warning("[%s] resolved commit %s differs from the pinned revision %s", spec.key, commit, spec.hf_revision)
        LOG.info("[%s] splits: %s (parquet reader)", spec.key, dict(zip(("train", "validation", "test"), raw)))
        if spec.expected_sizes is not None and tuple(raw) != tuple(spec.expected_sizes):
            msg = f"dataset '{spec.key}': split sizes {raw} differ from the expected {tuple(spec.expected_sizes)} (wrong / truncated / other version?)"
            if self.dc.strict_checks:
                raise RuntimeError(msg + "  Set DatasetSpec.expected_sizes=None or cfg.data.strict_checks=False to override.")
            LOG.warning(msg)
        self.provenance.setdefault("sources", {})[spec.key] = prov
        return out

    def _load_local(self, load_dataset, d: str, spec: DatasetSpec):
        if not d or not os.path.isdir(d):
            raise FileNotFoundError(f"local_dir='{d}' of dataset '{spec.key}' is not a directory")
        if os.path.exists(os.path.join(d, "dataset_dict.json")):
            from datasets import load_from_disk
            return load_from_disk(d)
        aliases = {spec.train_split: (spec.train_split, "train"), spec.val_split: (spec.val_split, "validation", "val", "valid", "dev")}
        if spec.test_split:
            aliases[spec.test_split] = (spec.test_split, "test")
        files: Dict[str, List[str]] = {}
        for split, al in aliases.items():
            f: List[str] = []
            for a in dict.fromkeys(al):
                f += glob.glob(os.path.join(d, f"{a}*.parquet")) + glob.glob(os.path.join(d, "**", f"{a}*.parquet"), recursive=True)
            if not f:
                raise FileNotFoundError(f"no '{split}*.parquet' file found under {d}")
            files[split] = sorted(set(f))
        return load_dataset("parquet", data_files=files, cache_dir=os.path.join(self.cfg.data_dir, "hf_local"))

    def _load_spec(self, spec: DatasetSpec) -> Dict[str, Optional[SplitSource]]:
        """{'train': src, 'val': src, 'test': src or None} for one corpus."""
        local = spec.local_dir or (self.dc.local_dir if (self.dc.source == "local" and len(self.specs) == 1) else None)
        if self.dc.source == "local" and local is None:
            raise ValueError(f"cfg.data.source='local' but dataset '{spec.key}' has no local_dir (set DatasetSpec.local_dir, or cfg.data.local_dir "
                             f"for a single source)")
        if self.dc.reader == "parquet" and (local is None or not os.path.exists(os.path.join(local, "dataset_dict.json"))):
            return self._load_parquet(spec, local)
        try:
            from datasets import load_dataset
        except Exception as e:
            raise ImportError(f"`datasets` is required for real data: pip install datasets ({e})") from e
        dsd = self._load_local(load_dataset, local, spec) if local else self._download(load_dataset, spec)
        names = list(dsd.keys())

        def pick(*ks):
            return next((k for k in ks if k in names), None)
        k_tr, k_va = pick(spec.train_split, "train"), pick(spec.val_split, "validation", "val", "valid", "dev")
        k_te = pick(spec.test_split, "test") if spec.test_split else None
        if k_tr is None or k_va is None or (spec.test_split and k_te is None):
            raise RuntimeError(f"dataset '{spec.key}': expected splits {spec.train_split} / {spec.val_split} / {spec.test_split}, found {names}. "
                               "(Re-using the validation split as test, or carving validation out of train silently, is not allowed.)")
        raw = [len(dsd[k_tr]), len(dsd[k_va])] + ([len(dsd[k_te])] if k_te else [])
        prov = {"dataset": spec.hf_dataset or local, "requested_revision": spec.hf_revision, "raw_sizes": raw, "language": spec.language,
                "script": spec.script, "rtl": spec.rtl, "licence": spec.licence, "local": bool(local)}
        LOG.info("[%s] splits: %s | columns: %s", spec.key, dict(zip(("train", "validation", "test"), raw)), dsd[k_tr].column_names)
        if not local and spec.hf_dataset:
            try:                                                   # record the exact commit that was used
                from huggingface_hub import HfApi
                prov["resolved_commit"] = HfApi().dataset_info(spec.hf_dataset, revision=spec.hf_revision).sha
            except Exception as e:
                prov["resolved_commit"] = None
                LOG.info("could not query the dataset commit hash (%s)", e)
        if spec.expected_sizes is not None and tuple(raw) != tuple(spec.expected_sizes):
            msg = f"dataset '{spec.key}': split sizes {raw} differ from the expected {tuple(spec.expected_sizes)} (wrong / truncated / other version?)"
            if self.dc.strict_checks:
                raise RuntimeError(msg + "  Set DatasetSpec.expected_sizes=None or cfg.data.strict_checks=False to override.")
            LOG.warning(msg)
        self.provenance.setdefault("sources", {})[spec.key] = prov
        col = self.dc.style_label_column
        mk = lambda k: HFSource(dsd[k], col, spec.image_col, spec.text_col)  # noqa: E731
        return {"train": mk(k_tr), "val": mk(k_va), "test": mk(k_te) if k_te else None}

    def _synthetic(self) -> Dict[str, Optional[SplitSource]]:
        self.using_synthetic = True
        n_tr, n_va, n_te = self.dc.synthetic_sizes
        s, w = self.cfg.run.seed, self.dc.synthetic_words
        LOG.warning("=" * 78)
        LOG.warning("SYNTHETIC SMOKE TEST: fake handwriting, for plumbing tests only -- NOT FOR PUBLICATION")
        LOG.warning("=" * 78)
        self.provenance.setdefault("sources", {})["synthetic"] = {"synthetic": True, "sizes": list(self.dc.synthetic_sizes)}
        return {"train": SyntheticSource(n_tr, s, w), "val": SyntheticSource(n_va, s + 1, w), "test": SyntheticSource(n_te, s + 2, w)}

    def _load_sources(self) -> List[Dict[str, Optional[SplitSource]]]:
        """One {'train','val','test'} dict per corpus (same order as self.specs).  Overridden by the self-tests."""
        dc = self.dc
        self.provenance.update({"source": dc.source, "sources": {}, "synthetic": dc.source == "synthetic"})
        if dc.source == "synthetic":
            if not self.cfg.run.smoke_test:
                raise RuntimeError("cfg.data.source='synthetic' is only permitted through Config.smoke() (run.smoke_test=True): a real run "
                                   "never trains on synthetic data.")
            return [self._synthetic() for _ in self.specs]
        if dc.source not in ("hf", "local"):
            raise ValueError(f"unknown cfg.data.source '{dc.source}'")
        out = []
        for spec in self.specs:
            if spec.key == "synthetic":
                raise RuntimeError("the 'synthetic' corpus is only available with cfg.data.source='synthetic' (Config.smoke())")
            out.append(self._load_spec(spec))
        return out

    # ---------------------------------------------------------------- prepare
    @staticmethod
    def _select(n: int, limit: Optional[int], seed: int, block: int = 1) -> np.ndarray:
        """All n lines, or a seeded subset of `limit` lines made of WHOLE blocks of `block` consecutive lines (lines are stored form by form;
        a line-level random subset would break the form structure that the clustered bootstrap and the val_sel / val_gate split rely on)."""
        idx = np.arange(n)
        if limit is None or limit >= n:
            return idx
        block = max(1, int(block))
        nb = int(math.ceil(n / block))
        order = np.random.RandomState(seed).permutation(nb)
        keep, tot = [], 0
        for b in order:
            seg = idx[b * block:(b + 1) * block]
            keep.append(seg)
            tot += len(seg)
            if tot >= limit:
                break
        return np.sort(np.concatenate(keep))[:max(limit, 0)] if keep else idx[:0]

    def _decontaminate(self, cache, text, src_id, indices, domain) -> Dict[str, int]:
        """Drop train lines whose preprocessed image also occurs in validation or test, and validation lines that also occur in test
        (exact content digest).  Needed when corpora that share material are combined (e.g. two releases of one collection)."""
        rep = {}
        for k, others in (("train", ("val", "test")), ("val", ("test",))):
            bad = set().union(*[set(cache[o]["digest"]) for o in others])
            keep = np.array([d not in bad for d in cache[k]["digest"]], bool)
            rep[f"removed_from_{k}"] = int((~keep).sum())
            if (~keep).any():
                kk = np.flatnonzero(keep)
                cache[k] = _subset_cache(cache[k], kk)
                text[k] = [text[k][i] for i in kk]
                src_id[k], indices[k], domain[k] = src_id[k][kk], indices[k][kk], domain[k][kk]
        LOG.info("decontamination: %s", rep)
        return rep

    def _integrity(self, cache: Dict[str, Dict[str, Any]], text: Dict[str, List[str]]) -> Dict[str, Any]:
        rep: Dict[str, Any] = {}
        dig = {k: set(c["digest"]) for k, c in cache.items()}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            n = len(dig[a] & dig[b])
            rep[f"image_overlap_{a}_{b}"] = n
            if n:
                msg = f"{n} identical line images occur in both '{a}' and '{b}' (data leakage)"
                if self.dc.strict_checks:
                    raise RuntimeError(msg)
                LOG.warning(msg)
        for k in text:
            rep[f"empty_text_{k}"] = int(sum(1 for t in text[k] if not t))
        return rep

    def prepare(self) -> "HandwritingDataModule":
        dc, seed, prov = self.dc, self.cfg.run.seed, self.provenance
        assert dc.img_height % 16 == 0, "img_height must be a multiple of 16"
        per_corpus = self._load_sources()
        # ---- gather every split of every corpus: caches, VISUAL-order texts, domain ids ---------------------------------------
        cache: Dict[str, Dict[str, Any]] = {}
        text: Dict[str, List[str]] = {}
        src_id: Dict[str, np.ndarray] = {}
        indices: Dict[str, np.ndarray] = {}
        domain: Dict[str, np.ndarray] = {}
        sources: Dict[str, List[SplitSource]] = {"train": [], "val": [], "test": []}
        for k in ("train", "val", "test"):
            caches, texts, sids, idxs, doms = [], [], [], [], []
            for d, (spec, srcs) in enumerate(zip(self.specs, per_corpus)):
                src = srcs.get(k)
                if src is None:
                    continue
                cap = (spec.max_lines or {}).get({"train": "train", "val": "validation", "test": "test"}[k])
                sel = self._select(len(src), cap, dc.subset_seed + 100 * d + {"train": 0, "val": 1, "test": 2}[k], dc.cap_block)
                ckey = {"corpus": spec.key, "split": k, "source": spec.hf_dataset or spec.local_dir or "", "revision": spec.hf_revision,
                        "n": len(src), "sel": hashlib.sha1(np.asarray(sel, np.int64).tobytes()).hexdigest(), "H": dc.img_height,
                        "max_w": dc.max_img_width, "min_w": dc.min_img_width, "pad_x": dc.pad_x, "ink": bool(dc.cache_images), "v": 2}
                build = (lambda ink_path, src=src, sel=sel, spec=spec, k=k: build_line_cache(src, sel, dc.img_height, dc.max_img_width, dc.min_img_width,
                                                                                            dc.pad_x, f"decoding [{spec.key}/{k}]", dc.cache_images, ink_path))
                use_disk = dc.line_cache_dir if (spec.key != "synthetic" and not self.is_synthetic) else None
                caches.append(cached_line_cache(use_disk, ckey, build))
                texts += [to_visual(src.text(int(j)), spec.rtl) for j in sel]
                sids.append(np.full(len(sel), len(sources[k]), np.int64))
                sources[k].append(src)
                idxs.append(sel)
                doms.append(np.full(len(sel), d, np.int64))
            cache[k] = _concat_caches(caches)
            text[k] = texts
            src_id[k] = np.concatenate(sids) if sids else np.zeros(0, np.int64)
            indices[k] = np.concatenate(idxs) if idxs else np.zeros(0, np.int64)
            domain[k] = np.concatenate(doms) if doms else np.zeros(0, np.int64)
            limit = {"train": dc.limit_train, "val": dc.limit_val, "test": dc.limit_test}[k]
            if limit is not None and limit < len(text[k]):
                keep = self._select(len(text[k]), limit, dc.subset_seed + {"train": 0, "val": 1, "test": 2}[k], dc.cap_block)
                cache[k] = _subset_cache(cache[k], keep)
                text[k] = [text[k][i] for i in keep]
                src_id[k], indices[k], domain[k] = src_id[k][keep], indices[k][keep], domain[k][keep]
        if len(text["test"]) == 0:
            raise RuntimeError("no corpus contributes test lines (every DatasetSpec has test_split=None): nothing to evaluate on")
        if dc.decontaminate:
            prov["decontamination"] = self._decontaminate(cache, text, src_id, indices, domain)
        prov["integrity"] = self._integrity(cache, text)
        LOG.info("Example (train line 0, %s): %r | preprocessed width %d px", self.specs[int(domain["train"][0])].key, text["train"][0],
                 int(cache["train"]["width"][0]))

        # ---- labeled budget / calibration lines / unlabeled lines (decided BEFORE any label-dependent filtering) -----------------
        n_tr = len(text["train"])
        perm = np.random.RandomState(seed).permutation(n_tr)
        n_lab = n_tr if dc.labeled_fraction >= 1.0 else max(1, int(round(dc.labeled_fraction * n_tr)))
        labeled = np.zeros(n_tr, bool)
        labeled[perm[:n_lab]] = True
        drop, why = np.zeros(n_tr, bool), Counter()
        for i in np.flatnonzero(labeled):                      # only LABELED lines may be filtered by their text
            t = text["train"][i]
            frames = int(math.ceil(cache["train"]["width"][i] / 4))
            reason = ("empty transcription" if len(t) == 0 else
                      "no letter or digit in the transcription" if dc.drop_nonlinguistic and not is_linguistic(t)
                      else "transcription longer than max_text_len" if len(t) > dc.max_text_len
                      else "too few CTC frames" if ctc_min_frames(list(t)) > frames else None)
            if reason:
                drop[i] = True
                why[reason] += 1
        keep = np.flatnonzero(~drop)
        tr_cache, tr_text, tr_lab = _subset_cache(cache["train"], keep), [text["train"][i] for i in keep], labeled[keep]
        tr_sid, tr_idx, tr_dom = src_id["train"][keep], indices["train"][keep], domain["train"][keep]
        if why:
            LOG.info("dropped %d labeled train lines: %s", int(drop.sum()), dict(why))

        # ---- alphabet from the LABELED budget only (grapheme clusters; calibration lines are part of the budget) ------------------
        self.codec = CharCodec.from_texts([t for t, l in zip(tr_text, tr_lab) if l])
        if self.base_units:                                   # warm start: keep the checkpoint's indices, append what the new corpus adds
            from .alphabet import extend_codec_units
            self.codec, _groups = extend_codec_units(self.base_units, self.codec.units)
            LOG.info("alphabet extends the checkpoint's %d units with %d new ones", len(self.base_units), len(self.codec.units) - len(self.base_units))
        LOG.info("Alphabet (%d units, labeled lines only): %s", len(self.codec.units), self.codec.chars[:200] + ("..." if len(self.codec.units) > 200 else ""))
        oov = {}
        for k in ("val", "test"):
            tot = max(sum(len(t) for t in text[k]), 1)
            oov[k] = sum(1 for t in text[k] for c in t if c not in self.codec.c2i) / tot
        if max(oov.values()) > 0:
            LOG.info("characters outside the alphabet (counted as errors): val %.4f%% test %.4f%%", 100 * oov["val"], 100 * oov["test"])

        # ---- part of the labeled budget is HELD OUT (never trained on) to calibrate the trust rule / selection ---------------------
        calib = np.zeros(len(keep), bool)
        if (self.cfg.train.pseudo.enabled or self.cfg.train.frag.enabled) and dc.calib_fraction > 0 and not tr_lab.all():
            pos = np.flatnonzero(tr_lab)
            n_c = int(round(dc.calib_fraction * len(pos)))
            if n_c > 0:
                calib[np.random.RandomState(seed + 11).permutation(pos)[:n_c]] = True
        tr_lab_fit = tr_lab & ~calib
        LOG.info("labeled budget %d lines = %d for training + %d held out for calibration", int(tr_lab.sum()), int(tr_lab_fit.sum()), int(calib.sum()))

        # ---- style normalisation + style classes (images only) -------------------------------------------------------------------
        meta = {"train": tr_cache, "val": cache["val"], "test": cache["test"]}
        raw_tr = tr_cache["style"]
        with np.errstate(all="ignore"):
            mu = np.nan_to_num(np.nanmean(raw_tr, 0), nan=0.0)
            sd = np.nan_to_num(np.nanstd(raw_tr, 0), nan=1.0)
        self.style_mean, self.style_std = mu.astype(np.float32), np.maximum(sd, STYLE_FLOORS).astype(np.float32)
        norm, ok_d, raw = {}, {}, {}
        for k in meta:
            r = meta[k]["style"]
            z = (r - self.style_mean) / self.style_std
            ok_d[k], raw[k] = np.isfinite(z), r.astype(np.float32)
            norm[k] = np.clip(np.nan_to_num(z, nan=0.0), -3.0, 3.0).astype(np.float32)
        nan_frac = {n: float(1.0 - ok_d["train"][:, j].mean()) for j, n in enumerate(STYLE_NAMES)}
        LOG.info("style descriptors not measurable (masked, not invented): %s", {k: f"{100 * v:.1f}%" for k, v in nan_frac.items()})
        extra = {"train": [sources["train"][s].extra(int(j)) for s, j in zip(tr_sid, tr_idx)],
                 "val": [sources["val"][s].extra(int(j)) for s, j in zip(src_id["val"], indices["val"])],
                 "test": [sources["test"][s].extra(int(j)) for s, j in zip(src_id["test"], indices["test"])]}
        if dc.style_label_column and extra["train"] and extra["train"][0] is not None:
            vocab = {w: i for i, w in enumerate(sorted(set(extra["train"])))}
            self.n_clusters = len(vocab)
            cl = {k: np.array([vocab.get(e, -100) for e in extra[k]]) for k in extra}
            LOG.info("Style classes = %d writers from column '%s'", self.n_clusters, dc.style_label_column)
        else:
            self.centroids = kmeans_np(norm["train"], dc.style_num_clusters, seed=seed)
            self.n_clusters = len(self.centroids)
            cl = {k: ((norm[k][:, None, :] - self.centroids[None]) ** 2).sum(-1).argmin(1) for k in norm}
            for k in cl:                                       # a class label from <= 3 measured descriptors is not trusted
                cl[k] = np.where(ok_d[k].sum(1) >= NUM_DESC - 1, cl[k], -100)

        # ---- [STRATA] label-free difficulty cells: thin stroke + tight word gaps = hard; cut points per corpus (train images only) ---
        fc = self.cfg.train.frag
        n_str = max(1, int(fc.strata))
        doms_all = {"train": tr_dom, "val": domain["val"], "test": domain["test"]}
        hard = {k: -(np.where(ok_d[k][:, 1], norm[k][:, 1], 0.0) + np.where(ok_d[k][:, 3], norm[k][:, 3], 0.0)) for k in norm}
        per_dom = bool(fc.strata_per_domain) and len(self.specs) > 1
        self.strata_cuts = {}
        for d in range(len(self.specs)):
            h = hard["train"][tr_dom == d] if per_dom else hard["train"]
            self.strata_cuts[d] = np.quantile(h, np.linspace(0, 1, n_str + 1)[1:-1]) if (n_str > 1 and len(h) > n_str) else np.zeros(0)
            if not per_dom:
                break
        strat = {}
        for k in hard:
            s_ = np.zeros(len(hard[k]), np.int64)
            for d in range(len(self.specs)):
                m = doms_all[k] == d
                cuts = self.strata_cuts[d if per_dom else 0]
                s_[m] = np.digitize(hard[k][m], cuts) + (d * n_str if per_dom else 0)
            strat[k] = s_
        self.n_strata = n_str * (len(self.specs) if per_dom else 1)
        self.labeled_idx = np.flatnonzero(tr_lab_fit)
        self.calib_idx = np.flatnonzero(calib)
        self.unlabeled_idx = np.flatnonzero(~tr_lab)
        self.train_eval_idx = np.sort(self.labeled_idx[np.random.RandomState(seed + 7).permutation(len(self.labeled_idx))[:dc.train_eval_subset]])
        LOG.info("Train lines: %d (labeled %d / calibration %d / unlabeled %d) | val %d | test %d | corpora %s | strata %d", len(tr_idx),
                 len(self.labeled_idx), len(self.calib_idx), len(self.unlabeled_idx), len(text["val"]), len(text["test"]), self.domain_names, self.n_strata)

        # ---- how much of each descriptor does the TRANSCRIPTION alone explain?  (labeled lines only) ----------------------------
        self.text_baseline_r2 = text_baseline_r2([tr_text[i] for i in self.labeled_idx], norm["train"][self.labeled_idx],
                                                 ok_d["train"][self.labeled_idx], self.codec.units, seed=seed)
        self.leak_active = (self.text_baseline_r2 < 0.25) & (ok_d["train"].mean(0) > 0.5)
        LOG.info("style explained by the transcription alone (R2): %s | leakage penalty acts on: %s",
                 {n: round(float(v), 2) for n, v in zip(STYLE_NAMES, self.text_baseline_r2)}, [n for n, a_ in zip(STYLE_NAMES, self.leak_active) if a_])

        common = dict(img_h=dc.img_height, max_w=dc.max_img_width, min_w=dc.min_img_width, pad_x=dc.pad_x)
        self.train_ds = HandwritingDataset(sources["train"], tr_sid, tr_idx, tr_cache["ink"] if dc.cache_images else None,
                                           tr_cache["width"], norm["train"], ok_d["train"], raw["train"], cl["train"],
                                           tr_text, tr_lab_fit, calib, tr_dom, self.specs, name="train", expose_hidden=False, **common)
        for k, attr in (("val", "val_ds"), ("test", "test_ds")):
            setattr(self, attr, HandwritingDataset(
                sources[k], src_id[k], indices[k], cache[k]["ink"] if dc.cache_images else None, cache[k]["width"], norm[k], ok_d[k],
                raw[k], cl[k], text[k], np.ones(len(text[k]), bool), None, domain[k], self.specs, name=k, expose_hidden=True, **common))
        self.train_ds.stratum, self.val_ds.stratum, self.test_ds.stratum = strat["train"], strat["val"], strat["test"]

        # ---- [PROTOCOL] validation = val_sel (checkpoint selection) + val_gate (every post-hoc decision), disjoint whole form blocks ----
        nv, blk = len(text["val"]), max(1, int(dc.val_block))
        if 0.0 < dc.val_gate_fraction < 1.0 and nv >= 4 * blk:
            nb = int(math.ceil(nv / blk))
            gate_blocks = np.zeros(nb, bool)
            gate_blocks[np.random.RandomState(seed + 23).permutation(nb)[:max(1, int(round(dc.val_gate_fraction * nb)))]] = True
            is_gate = gate_blocks[np.arange(nv) // blk]
            self.val_gate_idx, self.val_sel_idx = np.flatnonzero(is_gate), np.flatnonzero(~is_gate)
        else:
            self.val_gate_idx = self.val_sel_idx = np.arange(nv)
            if dc.val_gate_fraction > 0 and nv < 4 * blk:
                LOG.warning("validation has %d lines (< 4 blocks of %d): no val_sel / val_gate split", nv, blk)
        LOG.info("validation: %d lines -> val_sel %d (checkpoint selection) + val_gate %d (post-hoc decisions)%s", nv, len(self.val_sel_idx),
                 len(self.val_gate_idx), "" if len(self.val_sel_idx) != nv else "  [NOT split]")
        nonling = {k: int(sum(1 for t in text[k] if not is_linguistic(t))) for k in ("val", "test")}
        if any(nonling.values()):
            LOG.warning("evaluation lines whose transcription has no letter / digit: val %d, test %d (kept in the main metric; the report "
                        "also gives the CER without them)", nonling["val"], nonling["test"])

        prov.update({"n_train_used": int(len(tr_idx)), "n_labeled": int(len(self.labeled_idx)), "n_calibration": int(len(self.calib_idx)),
                     "n_unlabeled": int(len(self.unlabeled_idx)), "n_val": int(nv), "n_test": int(len(text["test"])),
                     "dropped_labeled": dict(why), "alphabet_units": self.codec.units, "alphabet_size": len(self.codec.units), "oov_rate": oov,
                     "labeled_fraction": dc.labeled_fraction, "calib_fraction": dc.calib_fraction, "seed": seed,
                     "labeled_lines_sha1": hashlib.sha1(np.c_[tr_dom[self.labeled_idx], tr_idx[self.labeled_idx]].tobytes()).hexdigest(),
                     "calibration_lines_sha1": hashlib.sha1(np.c_[tr_dom[self.calib_idx], tr_idx[self.calib_idx]].tobytes()).hexdigest(),
                     "nonlinguistic_eval_lines": nonling, "style_not_measurable_fraction": nan_frac,
                     "style_explained_by_text_r2": dict(zip(STYLE_NAMES, self.text_baseline_r2.tolist())),
                     "domains": self.domain_names, "n_strata": self.n_strata,
                     "lines_per_domain": {k: {s.key: int((doms_all[k] == d).sum()) for d, s in enumerate(self.specs)} for k in doms_all},
                     "versions": {"torch": torch.__version__, "numpy": np.__version__}})
        try:
            import datasets as _hfd
            prov["versions"]["datasets"] = _hfd.__version__
        except Exception:
            pass
        os.makedirs(self.cfg.run_dir, exist_ok=True)
        with open(os.path.join(self.cfg.run_dir, "provenance.json"), "w") as f:
            json.dump(json_safe(prov), f, indent=2, ensure_ascii=False)
        if self.is_synthetic:
            with open(os.path.join(self.cfg.run_dir, "SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION.txt"), "w") as f:
                f.write("This run used generated fake handwriting. Its numbers have no scientific meaning.\n")
        return self

    # ---------------------------------------------------------------- loaders
    def denorm_style(self, v: np.ndarray) -> np.ndarray:
        return np.asarray(v) * self.style_std + self.style_mean

    def _loader(self, ds: HandwritingDataset, sampler: BucketBatchSampler, workers: Optional[int] = None, collator=None) -> DataLoader:
        nw = self.dc.num_workers if workers is None else workers
        return DataLoader(ds, batch_sampler=sampler, collate_fn=collator or Collator(self.codec), num_workers=nw,
                          pin_memory=torch.cuda.is_available(), persistent_workers=False)

    def train_loader(self, indices: Optional[Sequence[int]] = None) -> DataLoader:
        """`indices` = the lines of this epoch (may contain repeats); default: every train line once."""
        s = BucketBatchSampler(self.train_ds.width, self.dc.batch_size, True, self.dc.bucket_mega, seed=self.cfg.run.seed, indices=indices)
        tc = self.cfg.train
        nospace = [d for d, sp in enumerate(self.specs) if sp.script in ("Han",)]
        return self._loader(self.train_ds, s, collator=Collator(self.codec, mix_prob=tc.mixline_prob, max_w=self.dc.max_img_width,
                                                                  gap=tc.mixline_gap, seed=self.cfg.run.seed, nospace_domains=nospace))

    def eval_loader(self, ds: HandwritingDataset, indices: Optional[Sequence[int]] = None, batch_size: Optional[int] = None) -> DataLoader:
        s = BucketBatchSampler(ds.width, batch_size or self.dc.eval_batch_size, False, indices=indices)
        return self._loader(ds, s, workers=0 if ds.ink is not None else None)   # cached lines need no decoding workers
