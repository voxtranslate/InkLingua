"""
The interpretability suite (v11).  Every analysis answers one question about the trained model, works on the evaluation split without any
extra label beyond the references it already has, and writes the numbers behind its figure (CSV / JSON) next to the figure:

  layer probes              where in the network does the character identity (and the writer's style) become linearly readable?
  uncertainty decomposition  which signal predicts a wrong line: the model's confidence, the posterior entropy, MC-dropout disagreement,
                             view disagreement or the FRAG-RC character risk?
  character counterfactuals  erase the ink of one character: which other characters of the reading change?  (dependency structure)
  style neighbours           what does the style space group?  nearest lines in the style embedding and their measured descriptors
  alignments                 the CTC alignment of every line (character, frames, pixels, confidence) as jsonl for downstream work
  long-line stress test      CER on concatenated pairs of test lines vs the same lines read alone (length generalisation)
  group tables               CER with form-clustered intervals per corpus, per hardness cell and per style cluster
"""
from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .decoding import greedy_decode, greedy_runs, seq_confidence
from .frag import char_evidence
from .pseudo import PERTURBATIONS, apply_perturbation
from .riskcontrol import auroc
from .style import STYLE_NAMES, assign_nearest, kmeans_np
from .text import align_chars, bootstrap_cer_ci, cer, edit_distance
from .utils import LOG, json_safe, to_device, write_csv


def _ridge_classifier_cv(X: np.ndarray, y: np.ndarray, folds: int = 2, lam: float = 1.0, seed: int = 0) -> float:
    """Cross-validated accuracy of a closed-form least-squares classifier (ridge to one-hot targets): a cheap linear probe."""
    n = len(y)
    if n < 20:
        return float("nan")
    classes, yi = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        return float("nan")
    fold = np.random.RandomState(seed).permutation(n) % folds
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    correct = 0
    for f in range(folds):
        tr, te = fold != f, fold == f
        if te.sum() == 0 or tr.sum() < 5:
            continue
        A = np.c_[Xs[tr], np.ones(tr.sum())]
        Y = np.eye(len(classes))[yi[tr]]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ Y)
        pred = (np.c_[Xs[te], np.ones(te.sum())] @ W).argmax(1)
        correct += int((pred == yi[te]).sum())
    return float(correct / n)


def _ridge_r2_cv(X: np.ndarray, y: np.ndarray, folds: int = 2, lam: float = 1.0, seed: int = 0) -> float:
    n = len(y)
    if n < 20 or y.std() < 1e-9:
        return float("nan")
    fold = np.random.RandomState(seed).permutation(n) % folds
    Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
    sse, sst = 0.0, float(((y - y.mean()) ** 2).sum())
    for f in range(folds):
        tr, te = fold != f, fold == f
        A = np.c_[Xs[tr], np.ones(tr.sum())]
        W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (y[tr] - y[tr].mean()))
        sse += float(((y[te] - (np.c_[Xs[te], np.ones(te.sum())] @ W + y[tr].mean())) ** 2).sum())
    return float(np.clip(1.0 - sse / max(sst, 1e-9), 0.0, 1.0))


class InterpretMixin:
    # ------------------------------------------------------------------ layer probes
    @torch.inference_mode()
    def _layer_probes(self, ds, ids: Sequence[int], folder: str) -> Dict[str, Any]:
        """Hidden states after every encoder layer at the EMITTING frames (label = the emitted unit) -> linear probe accuracy per layer;
        mean-pooled hidden states -> linear R2 of the measured slant per layer.  Shows where content and style are represented."""
        self.model.eval()
        H_by_layer: Dict[int, List[np.ndarray]] = defaultdict(list)
        pooled: Dict[int, List[np.ndarray]] = defaultdict(list)
        labels: List[int] = []
        slant: List[float] = []
        for batch in self.data.eval_loader(ds, [int(i) for i in ids]):
            batch = to_device(batch, self.device)
            o = self.model(batch["image"], batch["width"], keep_hidden=True)
            lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
            hid = [h.float().cpu().numpy() for h in o["hidden"]]
            for b in range(lp.shape[0]):
                L = lens[b]
                am = lp[b, :L].argmax(1)
                emit = np.flatnonzero(am != 0)
                if len(emit) == 0:
                    continue
                labels += am[emit].tolist()
                for li, h in enumerate(hid):
                    H_by_layer[li].append(h[b, emit])
                    pooled[li].append(h[b, :L].mean(0))
                slant.append(float(ds.style_raw[int(batch["index"][b]), 0]))
        if not labels:
            return {}
        y = np.array(labels)
        keep = np.isin(y, [c for c, n in zip(*np.unique(y, return_counts=True)) if n >= 5])
        sl = np.array(slant)
        ok = np.isfinite(sl)
        rows = []
        for li in sorted(H_by_layer):
            X = np.concatenate(H_by_layer[li])
            acc = _ridge_classifier_cv(X[keep], y[keep], seed=li)
            Xp = np.stack(pooled[li])
            r2 = _ridge_r2_cv(Xp[ok], sl[ok], seed=li) if ok.sum() >= 20 else float("nan")
            rows.append({"layer": li, "name": "encoder input (content + FiLM)" if li == 0 else f"after layer {li}", "char_probe_accuracy": acc, "slant_probe_r2": r2,
                         "n_frames": int(keep.sum())})
        maj = float(np.bincount(np.unique(y[keep], return_inverse=True)[1]).max() / max(keep.sum(), 1))
        out = {"layers": rows, "majority_class_accuracy": maj, "n_lines": len(slant), "note": "2-fold ridge probes; accuracy far above the majority class = linearly "
               "readable character identity; slant R2 from the mean-pooled state = how much writer style the content stream still carries at that depth"}
        write_csv(os.path.join(folder, "layer_probes.csv"), ["layer", "name", "char_probe_accuracy", "slant_probe_r2", "n_frames"],
                   [[r["layer"], r["name"], r["char_probe_accuracy"], r["slant_probe_r2"], r["n_frames"]] for r in rows])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot([r["layer"] for r in rows], [r["char_probe_accuracy"] for r in rows], "o-", label="character identity (probe accuracy)")
        ax.axhline(maj, color="gray", ls=":", label="majority class")
        ax2 = ax.twinx()
        ax2.plot([r["layer"] for r in rows], [r["slant_probe_r2"] for r in rows], "s--", color="purple", label="slant (probe R2)")
        ax.set_xlabel("encoder depth (0 = input of the Transformer)"), ax.set_ylabel("probe accuracy"), ax2.set_ylabel("R2", color="purple")
        ax.legend(loc="center right", fontsize=8), ax2.legend(loc="lower right", fontsize=8)
        ax.set_title("where does the reading form?  linear probes per layer", fontsize=9)
        self._fig("10_explanations/layer_probes.png", "Layer-wise linear probes: character identity and slant", fig)
        return out

    # ------------------------------------------------------------------ uncertainty decomposition
    def _uncertainty_decomposition(self, ds, R: Dict[str, Any], folder: str) -> Dict[str, Any]:
        """Per line: 1 - sequence confidence, mean posterior entropy at the emitting frames, MC-dropout disagreement, view disagreement and
        (when the checkpoint carries a trust rule) the mean FRAG-RC character risk.  AUROC of each for 'line has an error', their rank correlations
        and the CER of the lines each signal would keep at 80 % coverage."""
        ic = self.ic
        pos = {int(i): k for k, i in enumerate(R["idx"])}
        ids = self._random_subset([int(i) for i in R["idx"]], ic.uncertainty_lines, 31)
        sig: Dict[str, Dict[int, float]] = defaultdict(dict)
        self.model.eval()
        drop_mods = [m for m in self.model.modules() if isinstance(m, nn.Dropout)]
        K = int(ic.mc_dropout_samples)
        for batch in self.data.eval_loader(ds, ids):
            batch = to_device(batch, self.device)
            x, w = batch["image"], batch["width"]
            with torch.no_grad():
                o = self.model(x, w)
                o2 = self.model(apply_perturbation(PERTURBATIONS["shear +0.08"], x, w), w)
            lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
            lp2, aux = o2["log_probs"].cpu().numpy(), o["aux_log_probs"].cpu().numpy()
            reads = []
            for b in range(lp.shape[0]):
                i, L = int(batch["index"][b]), lens[b]
                ids_, cf, fr = greedy_decode(lp[b, :L])
                text = self.data.codec.decode_ids(ids_)
                reads.append(text)
                p = np.exp(lp[b, :L])
                am = p.argmax(1)
                emit = am != 0
                ent = -(p * np.log(np.clip(p, 1e-9, 1))).sum(1)
                sig["1 - sequence confidence"][i] = 1.0 - seq_confidence(cf)
                sig["posterior entropy (emitting frames)"][i] = float(ent[emit].mean()) if emit.any() else float(ent.mean())
                sig["view disagreement (shear)"][i] = cer(text, self.data.codec.decode_ids(greedy_decode(lp2[b, :L])[0]))
                if getattr(self, "frag_model", None) is not None:
                    _, _, _, Fm = char_evidence(lp[b, :L], lp2[b, :L], aux[b, :L])
                    sig["FRAG-RC mean character risk"][i] = float(self.frag_model.risk(Fm).mean()) if len(Fm) else 1.0
            if K > 0 and drop_mods:
                for m in drop_mods:
                    m.train()
                try:
                    dis = np.zeros(lp.shape[0])
                    with torch.no_grad():
                        for k in range(K):
                            torch.manual_seed(1000 + k)
                            ok = self.model(x, w)
                            lpk, lk = ok["log_probs"].cpu().numpy(), ok["out_lens"].cpu().tolist()
                            for b in range(lp.shape[0]):
                                dis[b] += cer(reads[b], self.data.codec.decode_ids(greedy_decode(lpk[b, :lk[b]])[0])) / K
                finally:
                    for m in drop_mods:
                        m.eval()
                for b in range(lp.shape[0]):
                    sig[f"MC-dropout disagreement ({K} passes)"][int(batch["index"][b])] = float(dis[b])
        names = list(sig)
        S = np.array([[sig[n][i] for n in names] for i in ids], np.float64)
        bad = np.array([R["ed"][pos[i]] > 0 for i in ids])
        ed = np.array([R["ed"][pos[i]] for i in ids], np.float64)
        rl = np.maximum(np.array([R["reflen"][pos[i]] for i in ids], np.float64), 1.0)
        res: Dict[str, Any] = {"n_lines": len(ids), "line_error_rate": float(bad.mean()), "signals": {}}
        for j, n in enumerate(names):
            o = np.argsort(S[:, j], kind="stable")
            k80 = max(1, int(round(0.8 * len(ids))))
            res["signals"][n] = {"auroc_line_has_error": auroc(S[:, j], bad), "CER_of_kept_lines_at_80pct_coverage": float(ed[o[:k80]].sum() / rl[o[:k80]].sum())}
        if len(names) > 1:
            rk = np.argsort(np.argsort(S, axis=0), axis=0).astype(np.float64)
            C = np.corrcoef(rk.T)
            res["spearman_between_signals"] = {names[a]: {names[b]: float(C[a, b]) for b in range(len(names))} for a in range(len(names))}
        write_csv(os.path.join(folder, "uncertainty_signals.csv"), ["line_index", "has_error", "edits", "ref_len"] + names,
                   [[i, int(bad[k]), int(ed[k]), int(rl[k])] + S[k].tolist() for k, i in enumerate(ids)])
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        aur = [res["signals"][n]["auroc_line_has_error"] for n in names]
        ax[0].barh(range(len(names)), aur, color="steelblue")
        ax[0].set_yticks(range(len(names))), ax[0].set_yticklabels(names, fontsize=8), ax[0].set_xlabel("AUROC: line has an error"), ax[0].axvline(0.5, color="k", ls=":")
        ax[0].set_title("which uncertainty signal predicts a wrong line?", fontsize=9)
        for j, n in enumerate(names):
            o = np.argsort(S[:, j], kind="stable")
            cov = np.arange(1, len(ids) + 1) / len(ids)
            risk = np.cumsum(ed[o]) / np.maximum(np.cumsum(rl[o]), 1)
            ax[1].plot(cov, risk * 100, label=n)
        ax[1].set_xlabel("coverage (most certain lines first)"), ax[1].set_ylabel("CER of kept lines (%)"), ax[1].legend(fontsize=7)
        ax[1].set_title("selective prediction by each signal", fontsize=9)
        self._fig("02_uncertainty/uncertainty_decomposition.png", "Uncertainty decomposition: confidence vs entropy vs MC-dropout vs views vs FRAG-RC risk", fig)
        return res

    # ------------------------------------------------------------------ character counterfactuals
    @torch.inference_mode()
    def _counterfactuals(self, ds, ids: Sequence[int], folder: str) -> List[Dict[str, Any]]:
        """Erase the ink of one emitted character (its CTC frames x 4 px) and read again: which characters of the reading change?  A character
        that changes when ANOTHER one is erased depends on context (good for words, bad for isolated glyphs); the diagonal is the character's
        own dependence on its ink."""
        codec = self.data.codec
        out = []
        self.model.eval()
        for i in ids:
            ink = ds.get_ink(int(i))
            H, W0 = ink.shape
            x = torch.from_numpy(np.pad(ink, ((0, 0), (0, (-W0) % 4))))[None, None].to(self.device)
            w = torch.tensor([W0], device=self.device)
            o = self.model(x, w)
            T = int(o["out_lens"][0])
            lp = o["log_probs"][0, :T].cpu().numpy()
            ids_, st, en = greedy_runs(lp)
            if not len(ids_) or len(ids_) > 60:
                continue
            base = codec.decode_ids(ids_)
            xs = x.repeat(len(ids_), 1, 1, 1)
            for k in range(len(ids_)):
                xs[k, :, :, int(st[k]) * 4:int(en[k]) * 4] = 0.0
            D = np.zeros((len(ids_), len(ids_)), np.int8)
            new_texts = []
            for c in range(0, len(ids_), 32):
                oc = self.model(xs[c:c + 32], w.repeat(min(32, len(ids_) - c)))
                lpc, lc = oc["log_probs"].cpu().numpy(), oc["out_lens"].cpu().tolist()
                for q in range(lpc.shape[0]):
                    k = c + q
                    t2 = codec.decode_ids(greedy_decode(lpc[q, :lc[q]])[0])
                    new_texts.append(t2)
                    for op, ri, hi in align_chars(base, t2):
                        if op != "match" and ri is not None:
                            D[k, ri] = 1
            n_ctx = int((D.sum(1) - np.diag(D)).sum())
            out.append({"line": int(i), "reading": ds.display(int(i), base), "gt": ds.display(int(i)), "chars": [codec.i2c.get(c, "?") for c in ids_],
                        "self_dependence_share": float(np.diag(D).mean()), "context_changes_per_erasure": float(n_ctx / max(len(ids_), 1)),
                        "readings_after_erasure": [ds.display(int(i), t) for t in new_texts], "dependency": D.tolist()})
            fig, ax = plt.subplots(2, 1, figsize=(max(6, min(16, W0 / 80)), 2.2 + 0.18 * len(ids_)), gridspec_kw={"height_ratios": [1, max(1.5, 0.12 * len(ids_))]})
            ax[0].imshow(1 - ink, cmap="gray", aspect="auto"), ax[0].set_yticks([]), ax[0].set_xticks([])
            ax[0].set_title(f"line {i}: reading '{out[-1]['reading']}'", fontsize=8, loc="left")
            ax[1].imshow(D, cmap="Reds", aspect="auto", vmin=0, vmax=1)
            labs = [c if c != " " else "␣" for c in out[-1]["chars"]]
            ax[1].set_xticks(range(len(labs))), ax[1].set_xticklabels(labs, fontsize=6), ax[1].set_yticks(range(len(labs))), ax[1].set_yticklabels(labs, fontsize=6)
            ax[1].set_xlabel("character of the reading that CHANGED"), ax[1].set_ylabel("character whose ink was ERASED")
            self._fig(f"10_explanations/counterfactuals/line_{int(i):05d}.png", f"Character counterfactuals - line {i}", fig)
        with open(os.path.join(folder, "counterfactuals.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(out), f, indent=1, ensure_ascii=False)
        return out

    # ------------------------------------------------------------------ style neighbours
    def _style_neighbours(self, ds, R: Dict[str, Any], folder: str) -> Dict[str, Any]:
        """Nearest lines in the style embedding (cosine) for a few random query lines, with their measured descriptors; plus the mean absolute
        descriptor difference of embedding neighbours vs random pairs (a style space that groups similar hands has a ratio well below 1)."""
        E = np.asarray(R["style_emb"], np.float64)
        n = len(E)
        if n < 20:
            return {}
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
        raw = np.asarray(R["style_raw"], np.float64)
        rng = np.random.RandomState(17)
        q_pos = rng.choice(n, min(self.ic.neighbour_queries, n), replace=False)
        rows, ratios = [], {}
        nb_diff, rd_diff = [], []
        for q in range(n):
            sims = En @ En[q]
            sims[q] = -np.inf
            nb = np.argsort(-sims)[:5]
            r = rng.randint(0, n, 5)
            with warnings.catch_warnings():                     # a descriptor not measurable on all of these lines is NaN, by design
                warnings.simplefilter("ignore", RuntimeWarning)
                nb_diff.append(np.nanmean(np.abs(raw[nb] - raw[q]), 0))
                rd_diff.append(np.nanmean(np.abs(raw[r] - raw[q]), 0))
        with np.errstate(all="ignore"):
            nbm, rdm = np.nanmean(np.stack(nb_diff), 0), np.nanmean(np.stack(rd_diff), 0)
            ratios = {s: float(a / b) if b > 0 else float("nan") for s, a, b in zip(STYLE_NAMES, nbm, rdm)}
        fig, axes = plt.subplots(len(q_pos), 6, figsize=(18, 1.3 * len(q_pos)))
        axes = np.atleast_2d(axes)
        for r_, q in enumerate(q_pos):
            sims = En @ En[q]
            sims[q] = -np.inf
            nb = np.argsort(-sims)[:5]
            for c_, j in enumerate([q] + nb.tolist()):
                i = int(R["idx"][j])
                ax = axes[r_, c_]
                ax.imshow(1 - ds.get_ink(i), cmap="gray", aspect="auto"), ax.set_xticks([]), ax.set_yticks([])
                ax.set_title(("QUERY " if c_ == 0 else f"sim {sims[j]:.2f} ") + f"line {i} | slant {raw[j, 0]:+.0f} stroke {raw[j, 1]:.3f}", fontsize=6)
                rows.append([int(R["idx"][q]), c_, i, float(sims[j]) if c_ else 1.0] + raw[j].tolist())
        self._fig("05_style/style_neighbours.png", "Nearest neighbours in the style embedding (query = left column)", fig)
        write_csv(os.path.join(folder, "style_neighbours.csv"), ["query_line", "rank", "line_index", "cosine"] + STYLE_NAMES, rows)
        return {"neighbour_vs_random_descriptor_gap_ratio": ratios, "note": "< 1 means embedding neighbours have more similar measured style than random pairs"}

    # ------------------------------------------------------------------ alignments
    @torch.inference_mode()
    def _export_alignments(self, ds, ids: Sequence[int], path: str) -> int:
        """01_recognition/alignments.jsonl: the CTC alignment of every line (character, first / end frame, pixel span, peak confidence)."""
        codec = self.data.codec
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for batch in self.data.eval_loader(ds, [int(i) for i in ids]):
                batch = to_device(batch, self.device)
                o = self.model(batch["image"], batch["width"])
                lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
                for b in range(lp.shape[0]):
                    i, L = int(batch["index"][b]), lens[b]
                    ids_, st, en = greedy_runs(lp[b, :L])
                    p = np.exp(lp[b, :L])
                    chars = [{"ch": codec.i2c.get(c, "?"), "frame0": int(s), "frame1": int(e), "px0": int(s * 4), "px1": int(e * 4),
                              "conf": float(p[s:e, c].max())} for c, s, e in zip(ids_, st, en)]
                    f.write(json.dumps(json_safe({"line_index": i, "corpus": ds.domain_names[int(ds.domain[i])], "rtl": bool(ds.rtl[i]),
                                                  "reading_visual_order": codec.decode_ids(ids_), "reading": ds.display(i, codec.decode_ids(ids_)),
                                                  "frames": int(L), "chars": chars}), ensure_ascii=False) + "\n")
                    n += 1
        return n

    # ------------------------------------------------------------------ long-line stress test
    @torch.inference_mode()
    def _longline_stress(self, ds, R: Dict[str, Any], folder: str) -> Dict[str, Any]:
        """Pairs of test lines of the same corpus are concatenated (12 px gap) and read as one line; the CER of the concatenation is compared with
        the CER of the two lines read alone (same references).  A large gap = the model does not generalise to longer lines."""
        n_pairs = int(self.ic.longline_pairs)
        if n_pairs <= 0 or len(R["idx"]) < 4:
            return {}
        codec = self.data.codec
        rng = np.random.RandomState(23)
        idx = np.asarray(R["idx"])
        pos = {int(i): k for k, i in enumerate(idx)}
        sep = " " if " " in codec.c2i else ""
        pairs = []
        by_dom: Dict[int, List[int]] = defaultdict(list)
        for i in idx:
            by_dom[int(ds.domain[int(i)])].append(int(i))
        tries = 0
        while len(pairs) < n_pairs and tries < 20 * n_pairs:
            tries += 1
            d = int(rng.choice(list(by_dom)))
            if len(by_dom[d]) < 2:
                continue
            a, b = rng.choice(by_dom[d], 2, replace=False)
            if ds.width[a] + 12 + ds.width[b] <= self.cfg.data.max_img_width:
                pairs.append((int(a), int(b)))
        if not pairs:
            return {}
        ed_cat, ed_sep, rl, lens = [], [], [], []
        self.model.eval()
        for c in range(0, len(pairs), 8):
            chunk = pairs[c:c + 8]
            imgs, widths = [], []
            for a, b in chunk:
                ia, ib = ds.get_ink(a), ds.get_ink(b)
                g = np.zeros((ia.shape[0], 12), np.float32)
                imgs.append(np.concatenate([ia, g, ib], 1))
                widths.append(imgs[-1].shape[1])
            Wm = int(np.ceil(max(widths) / 4) * 4)
            x = torch.zeros(len(chunk), 1, imgs[0].shape[0], Wm)
            for k, im in enumerate(imgs):
                x[k, 0, :, :im.shape[1]] = torch.from_numpy(im)
            o = self.model(x.to(self.device), torch.tensor(widths, device=self.device))
            lp, ln = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
            for k, (a, b) in enumerate(chunk):
                ref = ds.oracle[a] + ("" if ds.domain_specs[int(ds.domain[a])].script in ("Han",) else sep) + ds.oracle[b]
                hyp = codec.decode_ids(greedy_decode(lp[k, :ln[k]])[0])
                ed_cat.append(edit_distance(ref, hyp))
                ed_sep.append(R["ed"][pos[a]] + R["ed"][pos[b]])
                rl.append(len(ref))
                lens.append(len(ref))
        ed_cat, ed_sep, rl = np.array(ed_cat, np.float64), np.array(ed_sep, np.float64), np.maximum(np.array(rl, np.float64), 1)
        res = {"pairs": len(pairs), "CER_concatenated": float(ed_cat.sum() / rl.sum()), "CER_same_lines_read_alone": float(ed_sep.sum() / rl.sum()),
               "mean_length_chars": float(np.mean(lens)), "note": "a concatenation separates the two lines by 12 px (MixLine-style); the reference is 'A B'"}
        res["generalisation_gap"] = res["CER_concatenated"] - res["CER_same_lines_read_alone"]
        with open(os.path.join(folder, "longline_stress.json"), "w") as f:
            json.dump(json_safe(res), f, indent=2)
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.bar(["read alone (sum)", "concatenated"], [res["CER_same_lines_read_alone"] * 100, res["CER_concatenated"] * 100], color=["seagreen", "indianred"])
        ax.set_ylabel("CER (%)"), ax.set_title(f"long-line stress test ({len(pairs)} pairs, mean {res['mean_length_chars']:.0f} chars)", fontsize=9)
        self._fig("07_robustness/longline_stress.png", "Long-line stress test: concatenated pairs vs the same lines read alone", fig)
        return res

    # ------------------------------------------------------------------ group tables
    def _group_tables(self, ds, R: Dict[str, Any], folder: str) -> Dict[str, Any]:
        """CER (+ form-clustered 95 % CI) per corpus, per hardness cell and per style cluster, for the greedy and the final readings."""
        idx = np.asarray(R["idx"])
        ed, rl = np.asarray(R["ed"], np.float64), np.maximum(np.asarray(R["reflen"], np.float64), 1.0)
        edf = np.asarray(R.get("ed_final", R["ed"]), np.float64)
        blk = max(1, int(self.ic.bootstrap_block))
        grp = idx // blk
        X = ds.style[idx].astype(np.float64)
        k = min(8, max(2, len(idx) // 50)) if len(idx) >= 100 else 1
        clus = assign_nearest(X, kmeans_np(X, k, seed=0)) if k > 1 else np.zeros(len(idx), int)
        groupings = {"corpus": np.array([ds.domain_names[int(ds.domain[i])] for i in idx]), "hardness_cell": ds.stratum[idx].astype(str),
                     "style_cluster": clus.astype(str)}
        rows, out = [], {}
        for gname, g in groupings.items():
            out[gname] = {}
            for v in sorted(set(g.tolist())):
                m = g == v
                if m.sum() < 3:
                    continue
                lo, hi = bootstrap_cer_ci(ed[m], rl[m], min(self.ic.bootstrap, 500), 3, groups=grp[m])
                rec = {"lines": int(m.sum()), "CER": float(ed[m].sum() / rl[m].sum()), "CER_ci95": [lo, hi], "CER_final": float(edf[m].sum() / rl[m].sum())}
                out[gname][v] = rec
                rows.append([gname, v, rec["lines"], rec["CER"], lo, hi, rec["CER_final"]])
        write_csv(os.path.join(folder, "groups.csv"), ["grouping", "group", "lines", "CER", "ci_lo", "ci_hi", "CER_final"], rows)
        return out

    def run_interpretability(self, ds, R: Dict[str, Any]) -> Dict[str, Any]:
        """Run every analysis, each guarded so that one failure never costs the evaluation."""
        ic, out = self.ic, {}
        if not ic.interpret:
            return out
        all_ids = [int(i) for i in R["idx"]]
        jobs = [
            ("layer_probes", lambda: self._layer_probes(ds, self._random_subset(all_ids, ic.probe_lines, 41), self.dirs["10_explanations"])),
            ("uncertainty_decomposition", lambda: self._uncertainty_decomposition(ds, R, self.dirs["02_uncertainty"])),
            ("counterfactuals", lambda: {"lines": [c["line"] for c in self._counterfactuals(ds, [i for i, _ in self._pick(R, ic.counterfactual_lines)],
                                                                                        self.dirs["10_explanations"])]}),
            ("style_neighbours", lambda: self._style_neighbours(ds, R, self.dirs["05_style"])),
            ("longline_stress", lambda: self._longline_stress(ds, R, self.dirs["07_robustness"])),
            ("groups", lambda: self._group_tables(ds, R, self.dirs["03_error_structure"])),
        ]
        if ic.export_alignments:
            jobs.append(("alignments_exported", lambda: self._export_alignments(ds, all_ids, os.path.join(self.dirs["01_recognition"], "alignments.jsonl"))))
        for name, fn in jobs:
            try:
                out[name] = fn()
            except Exception:
                LOG.exception("interpretability analysis '%s' failed (the rest of the evaluation is unaffected)", name)
        return out
