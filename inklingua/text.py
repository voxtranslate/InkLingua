"""
Text: grapheme-cluster codec, right-to-left handling, IAM detokenisation, edit distance, CER / WER / chrF, character
classes, form-clustered bootstrap intervals and the "what does the transcription explain" regression.

Script-agnostic by design (v11):
  * the CTC alphabet is made of GRAPHEME CLUSTERS (base character + combining marks), not code points, so Devanagari matras,
    Arabic diacritics or 'e + dot below + acute' are one output unit each (v10 split them, which broke CTC feasibility counts
    and made the zero-shot adapter's units inconsistent with the trained model);
  * right-to-left scripts: the recogniser reads the image left to right, so an RTL transcription (logical order) is converted
    to VISUAL order (reversed cluster by cluster) for training and metrics -- edit distance is invariant under reversing both
    strings, so CER / WER do not change -- and converted back for every human-facing output and the LLM (`to_logical`);
  * character classes use Unicode categories, so uncased letters (Arabic, Chinese, ...) are letters, not punctuation.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import editdistance as _editdistance
except Exception:  # pragma: no cover
    _editdistance = None


# ---------------------------------------------------------------- normalisation, grapheme clusters, direction
def normalize_text(t: str) -> str:
    t = t.replace(" ", " ").replace("\t", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", t).strip()


def grapheme_clusters(text: str) -> List[str]:
    """Base character + its combining marks = one output unit (NFC text: 'e\\u0323\\u0301' keeps the dot below and the acute
    together, which no precomposed code point exists for).  Zero-width joiners inside a cluster stay attached to it."""
    out: List[str] = []
    for ch in unicodedata.normalize("NFC", text):
        if out and (unicodedata.combining(ch) or ch in ("‍", "‌")):
            out[-1] += ch
        else:
            out.append(ch)
    return out


def to_visual(text: str, rtl: bool) -> str:
    """Logical -> visual order for an RTL script (the order in which the recogniser reads the image, left to right)."""
    return "".join(reversed(grapheme_clusters(text))) if rtl else text


def to_logical(text: str, rtl: bool) -> str:
    """Visual -> logical order (the inverse of to_visual; reversal is its own inverse)."""
    return to_visual(text, rtl)


# ---------------------------------------------------------------- IAM detokenisation (English only)
_DQ = ('"', "``", "''")
_CONTRACTION = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m"}


def _quote_roles(toks: List[str], pos: List[int]) -> List[str]:
    """Opening / closing role of the quote marks at `pos` (token positions of one quote family).  Quotes alternate; only the
    state at the start of the line is unknown (IAM lines cut a quotation anywhere), so it is inferred: with an odd number of
    marks either the first one closes a quotation that began on the previous line or the last one opens one that continues on
    the next line -- the last one opens if a word follows it and it is preceded by a comma / colon, another quote mark or a word."""
    if not pos:
        return []
    first_closes = False
    if len(pos) % 2 == 1:
        k = pos[-1]
        has_next = k + 1 < len(toks)
        prev = toks[k - 1] if k > 0 else ""
        last_opens = has_next and (k == 0 or prev in _DQ or prev == "'" or prev[-1:] in ",:;" or prev[-1:].isalnum())
        first_closes = not last_opens
    return [("close" if (j % 2 == 0) == first_closes else "open") for j in range(len(pos))]


def detokenize(t: str) -> str:
    """IAM transcriptions are tokenised ("house , and"; \" Left here \" ; It 's).  Re-attach punctuation, brackets, contractions
    and quotation marks the way the sentence would be written -- for the LLM prompt / LM scoring.  No character other than spaces
    is ever added, dropped or moved (tested).  Only applied to corpora whose DatasetSpec.detokenize is True."""
    toks = t.replace(" ", " ").split()
    n = len(toks)
    roles: List[str] = ["word"] * n
    for family in (_DQ, ("'",)):
        pos = [i for i, w in enumerate(toks) if w in family]
        for i, r in zip(pos, _quote_roles(toks, pos)):
            roles[i] = r
    for i, w in enumerate(toks):
        if roles[i] != "word":
            continue
        if w in _CONTRACTION or (w and all(c in ".,;:!?%)]}" for c in w)):
            roles[i] = "close"
        elif w and all(c in "([{$£" for c in w):
            roles[i] = "open"
    out, glue = "", True
    for w, r in zip(toks, roles):
        if r == "close":
            out += w
            glue = False
        elif r == "open":
            out += ("" if glue else " ") + w
            glue = True
        else:
            out += ("" if glue else " ") + w
            glue = False
    return out.strip()


# ---------------------------------------------------------------- the codec
class CharCodec:
    """Index 0 is the CTC blank; every other index is one GRAPHEME CLUSTER.  The alphabet is built from the transcripts of the
    LABELED training lines only (transcriptions of unlabeled lines are never read by the training pipeline); units that occur only
    in validation / test simply count as errors.
    `chars` is the joined string of the units (kept for compatibility with older checkpoints: re-clustering it gives the units back)."""

    BLANK = 0

    def __init__(self, units):
        if isinstance(units, str):
            units = grapheme_clusters(units)
        self.units: List[str] = list(units)
        self.c2i = {u: i + 1 for i, u in enumerate(self.units)}
        self.i2c = {i + 1: u for i, u in enumerate(self.units)}

    @classmethod
    def from_texts(cls, texts: Sequence[str]) -> "CharCodec":
        units = set()
        for t in texts:
            units.update(grapheme_clusters(t))
        return cls(sorted(units))

    @property
    def chars(self) -> str:
        return "".join(self.units)

    @property
    def num_classes(self) -> int:
        return len(self.units) + 1

    def encode(self, text: str) -> List[int]:
        return [self.c2i[u] for u in grapheme_clusters(text) if u in self.c2i]

    def decode_ids(self, ids: Sequence[int]) -> str:
        return "".join(self.i2c.get(int(i), "") for i in ids)

    def __eq__(self, other) -> bool:
        return isinstance(other, CharCodec) and other.units == self.units


def ctc_feasible(text: str, frames: int) -> bool:
    """A CTC alignment exists only if the text fits into the frames (repeated units need a blank in between)."""
    u = grapheme_clusters(text)
    need = len(u) + sum(1 for a, b in zip(u, u[1:]) if a == b)
    return len(u) > 0 and need <= frames


def ctc_min_frames(ids: Sequence[int]) -> int:
    return len(ids) + sum(1 for a, b in zip(ids, ids[1:]) if a == b)


# ---------------------------------------------------------------- metrics
def edit_distance(a: Sequence, b: Sequence) -> int:
    if _editdistance is not None:
        return int(_editdistance.eval(a, b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    return edit_distance(ref, hyp) / max(len(ref), 1)


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    return edit_distance(r, h) / max(len(r), 1)


def align_chars(ref: str, hyp: str) -> List[Tuple[str, Optional[int], Optional[int]]]:
    """Levenshtein alignment. Returns ops ('match'|'sub'|'ins'|'del', ref_idx, hyp_idx)."""
    n, m = len(ref), len(hyp)
    D = np.zeros((n + 1, m + 1), dtype=np.int32)
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        ri = ref[i - 1]
        for j in range(1, m + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (ri != hyp[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and D[i, j] == D[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            ops.append(("match" if ref[i - 1] == hyp[j - 1] else "sub", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and D[i, j] == D[i - 1, j] + 1:
            ops.append(("del", i - 1, None))
            i -= 1
        else:
            ops.append(("ins", None, j - 1))
            j -= 1
    return ops[::-1]


def chrf(hyp: str, ref: str, max_n: int = 6, beta: float = 2.0) -> float:
    """Character n-gram F-score (chrF, 0..100; Popovic 2015: precision and recall averaged over the n-gram orders 1..6, then one
    F-beta with beta = 2; spaces ignored).  sacreBLEU averages the F-score per order instead; the two differ by < 0.01 chrF on
    sentence pairs, so numbers are comparable but not byte-identical."""
    h, r = hyp.replace(" ", ""), ref.replace(" ", "")
    if not h or not r:
        return 100.0 if h == r else 0.0
    ps, rs = [], []
    for n in range(1, max_n + 1):
        H = Counter(h[i:i + n] for i in range(len(h) - n + 1))
        R = Counter(r[i:i + n] for i in range(len(r) - n + 1))
        if not H or not R:
            continue
        ov = sum((H & R).values())
        ps.append(ov / sum(H.values()))
        rs.append(ov / sum(R.values()))
    if not ps:
        return 0.0
    P, R_ = float(np.mean(ps)), float(np.mean(rs))
    if P + R_ == 0:
        return 0.0
    return 100.0 * (1 + beta ** 2) * P * R_ / (beta ** 2 * P + R_)


CHAR_CLASSES = ("lower", "upper", "letter", "digit", "punct", "space")


def char_class(c: str) -> str:
    """Script-agnostic character class: cased letters keep their case, uncased letters (Arabic, Han, ...) are 'letter'."""
    if c == " ":
        return "space"
    cat = unicodedata.category(c[0])
    if cat == "Ll":
        return "lower"
    if cat == "Lu" or cat == "Lt":
        return "upper"
    if cat.startswith("L") or cat.startswith("M"):
        return "letter"
    if cat.startswith("N"):
        return "digit"
    return "punct"


def is_linguistic(t: str) -> bool:
    """A reference with at least one letter or digit (form separators such as '-----' are not handwriting)."""
    return any(unicodedata.category(c).startswith(("L", "N")) for c in t)


# ---------------------------------------------------------------- bootstrap intervals (form-clustered)
MIN_BOOTSTRAP_GROUPS = 8


def group_index(groups: Optional[Sequence[int]], n: int) -> np.ndarray:
    """Consecutive group ids 0..G-1 of the lines (None -> every line is its own group)."""
    if groups is None:
        return np.arange(n)
    g = np.asarray(groups)
    if len(g) != n:
        raise ValueError(f"groups has {len(g)} entries for {n} lines")
    inv = np.unique(g, return_inverse=True)[1]
    if int(inv.max()) + 1 < MIN_BOOTSTRAP_GROUPS:       # resampling < 8 blocks is degenerate: fall back to lines (too narrow, not degenerate)
        return np.arange(n)
    return inv


def group_sums(gid: np.ndarray, *arrs: np.ndarray) -> List[np.ndarray]:
    G = int(gid.max()) + 1 if len(gid) else 0
    return [np.bincount(gid, weights=np.asarray(a, np.float64), minlength=G) for a in arrs]


def bootstrap_cer_ci(ed: np.ndarray, ref_len: np.ndarray, n: int = 1000, seed: int = 0,
                     groups: Optional[Sequence[int]] = None) -> Tuple[float, float]:
    """95% percentile-bootstrap CI of the corpus CER = sum(ed) / sum(ref_len).  groups=None resamples LINES.  Evaluation lines
    are stored form by form and the errors cluster by form / writer (v3 test run: lag-1 autocorrelation of the line CER 0.46,
    design effect 4), so a line-level bootstrap is about HALF as wide as it should be.  Pass groups = line_index // block to
    resample whole blocks (cluster bootstrap)."""
    ed, ref_len = np.asarray(ed, np.float64), np.asarray(ref_len, np.float64)
    if len(ed) == 0 or n <= 0:
        return float("nan"), float("nan")
    gid = group_index(groups, len(ed))
    E, L = group_sums(gid, ed, ref_len)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(E), size=(n, len(E)))
    vals = E[idx].sum(1) / np.maximum(L[idx].sum(1), 1.0)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_bootstrap_delta(ed_a: np.ndarray, ed_b: np.ndarray, ref_len: np.ndarray, n: int = 1000, seed: int = 0,
                           groups: Optional[Sequence[int]] = None, n_tests: int = 1) -> Dict[str, float]:
    """Paired (cluster) bootstrap of CER(a) - CER(b) on the same lines (negative = a is better).  lo / hi are the 95 % limits;
    lo_adj / hi_adj the limits at 95 % / n_tests (Bonferroni): use hi_adj < 0 to adopt the best of n_tests candidates chosen on
    the SAME lines."""
    ed_a, ed_b, ref_len = (np.asarray(v, np.float64) for v in (ed_a, ed_b, ref_len))
    if len(ed_a) == 0 or n <= 0:
        nan = float("nan")
        return {"delta": nan, "lo": nan, "hi": nan, "lo_adj": nan, "hi_adj": nan, "p_a_better": nan}
    gid = group_index(groups, len(ed_a))
    Ea, Eb, L = group_sums(gid, ed_a, ed_b, ref_len)
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(Ea), size=(n, len(Ea)))
    d = (Ea[idx].sum(1) - Eb[idx].sum(1)) / np.maximum(L[idx].sum(1), 1.0)
    tail = 2.5 / max(int(n_tests), 1)
    return {"delta": float((ed_a.sum() - ed_b.sum()) / max(ref_len.sum(), 1.0)), "lo": float(np.percentile(d, 2.5)),
            "hi": float(np.percentile(d, 97.5)), "lo_adj": float(np.percentile(d, tail)),
            "hi_adj": float(np.percentile(d, 100.0 - tail)), "p_a_better": float((d < 0).mean())}


def paired_mean_ci(a: Sequence[float], b: Sequence[float], n: int = 1000, seed: int = 0,
                   groups: Optional[Sequence[int]] = None) -> Dict[str, float]:
    """Mean of (a - b) over lines with a paired-bootstrap 95 % interval, plus how often a is better / worse (form-clustered)."""
    d = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if len(d) == 0:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": 0}
    rng = np.random.RandomState(seed)
    if groups is None:
        bs = d[rng.randint(0, len(d), (n, len(d)))].mean(1)
    else:
        gid = group_index(groups, len(d))
        G = int(gid.max()) + 1
        sums, cnt = group_sums(gid, d, np.ones_like(d))
        pick = rng.randint(0, G, (n, G))
        bs = sums[pick].sum(1) / np.maximum(cnt[pick].sum(1), 1)
    return {"delta": float(d.mean()), "lo": float(np.percentile(bs, 2.5)), "hi": float(np.percentile(bs, 97.5)),
            "n": int(len(d)), "share_better": float((d > 1e-9).mean()), "share_worse": float((d < -1e-9).mean())}


# ---------------------------------------------------------------- what does the TRANSCRIPTION explain of a style descriptor?
def text_features(texts: Sequence[str], units: Sequence[str]) -> np.ndarray:
    """Bag-of-units description of transcriptions: unit proportions, log length, words / 10, mean word length / 10."""
    idx = {c: i for i, c in enumerate(units)}
    X = np.zeros((len(texts), len(units) + 3), np.float32)
    for r, t in enumerate(texts):
        cl = grapheme_clusters(t)
        n = max(len(cl), 1)
        for c in cl:
            j = idx.get(c)
            if j is not None:
                X[r, j] += 1.0 / n
        nw = max(len(t.split()), 1)
        X[r, -3], X[r, -2], X[r, -1] = math.log1p(len(cl)), nw / 10.0, (len(cl) / nw) / 10.0
    return X


def ridge_cv_r2(X: np.ndarray, Y: np.ndarray, OK: np.ndarray, folds: int = 5, lam: float = 0.1, seed: int = 0) -> np.ndarray:
    """Cross-validated linear R^2 of every column of Y from X (closed-form ridge, centred on the TRAINING fold).  Column j uses
    only the rows with OK[:, j]; the result is clipped to [0, 1]."""
    n = len(X)
    out = np.zeros(Y.shape[1], np.float32)
    if n < 2 * folds:
        return out
    fold = np.random.RandomState(seed).permutation(n) % folds
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    for j in range(Y.shape[1]):
        ok = OK[:, j]
        if ok.sum() < 3 * folds:
            continue
        ybar, sse, sst = float(Y[ok, j].mean()), 0.0, 0.0
        for f in range(folds):
            tr, te = ok & (fold != f), ok & (fold == f)
            if te.sum() == 0 or tr.sum() < 5:
                continue
            xm, ym = Xs[tr].mean(0), float(Y[tr, j].mean())
            A = Xs[tr] - xm
            Wt = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ (Y[tr, j] - ym))
            sse += float(((Y[te, j] - ((Xs[te] - xm) @ Wt + ym)) ** 2).sum())
            sst += float(((Y[te, j] - ybar) ** 2).sum())
        if sst > 0:
            out[j] = float(np.clip(1.0 - sse / sst, 0.0, 1.0))
    return out


def text_baseline_r2(texts: Sequence[str], Z: np.ndarray, OK: np.ndarray, units: Sequence[str], seed: int = 0) -> np.ndarray:
    return ridge_cv_r2(text_features(texts, units), np.asarray(Z, np.float64), np.asarray(OK, bool), seed=seed)
