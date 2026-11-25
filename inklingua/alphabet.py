"""
Function-preserving growth of the CTC alphabet ([ALPHA-SPLIT]) and language-prior resolution of tied variants ([LEX-TIE]).

Used by (i) `mode="finetune"` / `TrainConfig.init_from` -- a checkpoint trained on one corpus is continued on another whose labeled
lines contain new characters -- and (ii) the zero-shot adapter (inklingua/adapt.py).  New units are APPENDED as tied copies of the
visually closest known unit: the group of k units gets bias - log k, so the group's logsumexp equals the old logit and the model's
behaviour on the old alphabet is unchanged (tested to ~1e-5).
"""
from __future__ import annotations

import math
import unicodedata
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .text import CharCodec, grapheme_clusters

# Visually similar letters of other orthographies (a curated heuristic; extend it with AdaptConfig.visual_base).  Accented forms need no
# entry: Unicode decomposition already gives their base letter.
_VISUAL_BASE = {"ɛ": "e", "ɔ": "o", "ɓ": "b", "ɗ": "d", "ŋ": "n", "ƙ": "k", "ɲ": "n", "ə": "e", "ʃ": "s", "ʋ": "v", "ƴ": "y", "æ": "a", "œ": "o",
                "ø": "o", "ı": "i", "ł": "l", "đ": "d", "ð": "d", "ƒ": "f", "Ɛ": "E", "Ɔ": "O", "Ɓ": "B", "Ɗ": "D", "Ŋ": "N", "Ƙ": "K", "Ə": "E",
                "Ø": "O", "Ł": "L", "Đ": "D", "Æ": "A", "Œ": "O"}


def visual_base_of(ch: str, known: Sequence[str], extra: Optional[Dict[str, str]] = None) -> Optional[str]:
    """The KNOWN unit a new unit looks like (its base letter), or None.  Decomposition first (é -> e, ẹ́ -> e, ɛ̀ -> ɛ -> e), then the table."""
    known = set(known)
    if ch in known:
        return None
    table = _VISUAL_BASE if not extra else {**_VISUAL_BASE, **extra}
    if ch in table and table[ch] in known:
        return table[ch]
    base = "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c))
    if len(base) == 1 and base != ch:
        if base in known:
            return base
        if base in table and table[base] in known:
            return table[base]
    return None


class CharGroups:
    """Output units that share one visual shape.  `rep[i]` = the unit i is a copy of (itself for ordinary units).  New units are
    APPENDED so every old index keeps its meaning."""

    def __init__(self, old_units: Sequence[str], new_units: Sequence[str], extra: Optional[Dict[str, str]] = None):
        old_units, new_units = list(old_units), list(new_units)
        assert new_units[:len(old_units)] == old_units, "new alphabet must extend the old one (old indices must not move)"
        self.old_units, self.units = old_units, new_units
        self.n_old, self.V = len(old_units) + 1, len(new_units) + 1
        self.rep = np.arange(self.V)
        self.orphans: List[int] = []
        known = set(old_units)
        for i in range(self.n_old, self.V):
            b = visual_base_of(new_units[i - 1], known, extra)
            if b is None:
                self.orphans.append(i)
            else:
                self.rep[i] = old_units.index(b) + 1
        self.members: Dict[int, List[int]] = {}
        for i in range(1, self.V):
            self.members.setdefault(int(self.rep[i]), []).append(i)
        self.M = np.zeros((self.V, self.V))
        self.M[np.arange(self.V), self.rep] = 1.0

    def variants(self, i: int) -> List[int]:
        return self.members.get(int(self.rep[int(i)]), [int(i)])

    def collapse(self, lp: np.ndarray) -> np.ndarray:
        """Frame log-posteriors [..., V] with the mass of every group moved onto its base letter."""
        p = np.exp(np.asarray(lp, np.float64)) @ self.M
        return np.log(np.maximum(p, 1e-30)).astype(np.float32)

    def collapse_t(self, lp: torch.Tensor) -> torch.Tensor:
        M = torch.as_tensor(self.M, dtype=lp.dtype, device=lp.device)
        return torch.log(torch.clamp(lp.exp() @ M, min=1e-30))

    def base_ids(self, ids: Sequence[int]) -> List[int]:
        return [int(self.rep[int(i)]) for i in ids]

    @property
    def n_variant_groups(self) -> int:
        return sum(1 for m in self.members.values() if len(m) > 1)


def extend_codec(codec: CharCodec, new_text: str, extra: Optional[Dict[str, str]] = None) -> Tuple[CharCodec, CharGroups]:
    """Append the grapheme clusters of `new_text` that the alphabet lacks (never reorders: old indices keep their meaning)."""
    old = list(codec.units)
    add: List[str] = []
    for u in grapheme_clusters(new_text):
        if not u.isspace() and not unicodedata.combining(u[0]) and u not in old and u not in add:
            add.append(u)
    new = CharCodec(old + add)
    return new, CharGroups(old, new.units, extra)


def extend_codec_units(old_units: Sequence[str], wanted_units: Sequence[str], extra: Optional[Dict[str, str]] = None) -> Tuple[CharCodec, CharGroups]:
    """Same as extend_codec for a list of wanted units (the alphabet of a new corpus); spaces are kept if wanted."""
    old = list(old_units)
    add = [u for u in wanted_units if u not in old]
    new = CharCodec(old + sorted(set(add), key=add.index))
    return new, CharGroups(old, new.units, extra)


@torch.no_grad()
def extend_ctc_heads(model: nn.Module, groups: CharGroups, orphan_bias: float = -9.0) -> nn.Module:
    """[ALPHA-SPLIT] grow the CTC head and the inter-CTC head WITHOUT changing what the model says about the old alphabet.  Every new
    variant is a copy of its base letter's row, the group of k units gets bias - log k; orphans (no visual base) start with zero
    weights and a very negative bias."""
    V0, V = groups.n_old, groups.V
    for name in ("ctc_head", "aux_head"):
        old: nn.Linear = getattr(model, name)
        new = nn.Linear(old.in_features, V).to(old.weight.device, old.weight.dtype)
        new.weight[:V0], new.bias[:V0] = old.weight, old.bias
        for i in range(V0, V):
            r = int(groups.rep[i])
            if r != i:
                new.weight[i], new.bias[i] = old.weight[r], old.bias[r]
            else:
                new.weight[i], new.bias[i] = 0.0, float(orphan_bias)
        for mem in groups.members.values():
            if len(mem) > 1:
                new.bias[mem] -= math.log(len(mem))
        setattr(model, name, new)
    model.num_classes = V
    return model


# ---------------------------------------------------------------- [LEX-TIE]
def resolve_variants(ids: Sequence[int], frames: Sequence[int], lp: np.ndarray, groups: CharGroups, scorer=None, *, beam: int = 4,
                     vis_w: float = 1.0, lm_w: float = 1.0, rho: float = 0.0, decided_p: float = 0.98, max_options: int = 6,
                     max_combo: int = 48) -> Tuple[str, Dict[str, float]]:
    """The reading `ids` (base letters, from the COLLAPSED posteriors) fixes WHICH letter was written; this decides, for every letter that
    has variants (e / é / è / ê ...), WHICH variant:  score = vis_w * sum log p_vis(variant | group, frame) + lm_w * (log p_LM(line) + rho *
    #tokens).  A letter whose best variant has posterior >= decided_p is settled by vision alone.  Ambiguous letters are resolved word by
    word (joint search inside a word) with a beam over words."""
    units = groups.units
    base = [units[int(i) - 1] for i in ids]
    amb: List[Tuple[int, List[str], np.ndarray]] = []
    n_var, n_dec = 0, 0
    for j, (i, t) in enumerate(zip(ids, frames)):
        mem = groups.variants(int(i))
        if len(mem) == 1:
            continue
        n_var += 1
        v = np.asarray(lp[int(t), mem], np.float64)
        v = v - (v.max() + math.log(float(np.exp(v - v.max()).sum())))
        o = np.argsort(-v, kind="stable")
        if float(np.exp(v[o[0]])) >= decided_p:
            base[j] = units[mem[int(o[0])] - 1]
            n_dec += 1
            continue
        keep = o[:max_options]
        amb.append((j, [units[mem[int(k)] - 1] for k in keep], v[keep]))
    info = {"n_variant_pos": float(n_var), "n_decided": float(n_dec), "n_ambiguous": float(len(amb)), "n_lm_texts": 0.0, "margin": float("inf")}
    if not amb:
        return "".join(base), info
    word_of, w = [], 0
    for u in base:
        word_of.append(w)
        if u.isspace():
            w += 1
    groups_w: List[List[Tuple[int, List[str], np.ndarray]]] = []
    for item in amb:
        if groups_w and word_of[groups_w[-1][0][0]] == word_of[item[0]]:
            groups_w[-1].append(item)
        else:
            groups_w.append([item])
    states: List[Tuple[Dict[int, str], float]] = [({}, 0.0)]
    scores: List[float] = [0.0]
    for gw in groups_w:
        keep_n = [len(o) for _, o, _ in gw]
        while math.prod(keep_n) > max_combo and max(keep_n) > 1:
            keep_n[int(np.argmax(keep_n))] -= 1
        combos = [([], 0.0)]
        for (j, opts, vis), kn in zip(gw, keep_n):
            combos = [(c + [o], cv + vis_w * float(vv)) for c, cv in combos for o, vv in zip(opts[:kn], vis[:kn])]
        cands = [({**st, **{jj: o for (jj, _, _), o in zip(gw, c)}}, sv + cv) for st, sv in states for c, cv in combos]
        texts = []
        for st, _ in cands:
            b2 = list(base)
            for jj, o in st.items():
                b2[jj] = o
            texts.append("".join(b2))
        if scorer is not None:
            lm, nt = scorer(texts)
            tot = [cv + lm_w * (float(a_) + rho * float(n)) for (_, cv), a_, n in zip(cands, lm, nt)]
            info["n_lm_texts"] += float(len(texts))
        else:
            tot = [cv for _, cv in cands]
        order = np.argsort(-np.asarray(tot), kind="stable")[:max(1, beam)]
        states, scores = [cands[int(k)] for k in order], [float(tot[int(k)]) for k in order]
    info["margin"] = float(scores[0] - scores[1]) if len(scores) > 1 else float("inf")
    for jj, c in states[0][0].items():
        base[jj] = c
    return "".join(base), info
