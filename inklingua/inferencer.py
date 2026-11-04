"""
Evaluation: metrics with form-clustered intervals, the structured output folders (one question per folder), calibration, error structure,
codebooks, the style task, post-hoc options, robustness, the trust rule, LLM rescoring, per-sample explanations, translation and the run
report.  Layout of <run>/inference/<split>/ (docs/INFERENCE_OUTPUTS.md):

    00_summary/           metrics.json, report.html (flat table of every number), v11_extras.json, training_curves.png
    01_recognition/       predictions.csv, worst_lines.txt, alignments.jsonl, cer_distribution.png, top_confusions.png
    02_uncertainty/       calibration.png, word_marks.json, uncertainty_decomposition.png + uncertainty_signals.csv
    03_error_structure/   error_structure.png, groups.csv (per corpus / hardness cell / style cluster)
    04_posthoc/           posthoc.json (view ensemble, AdaBN, writer self-distillation, space bonus; adopted or not, and why)
    05_style/             style regression, confusion, embeddings, disentanglement probe, FiLM, style_neighbours.png/.csv
    06_codebooks/         usage, code -> character, PCA
    07_robustness/        perturbation profile, longline_stress.png/.json
    08_self_distillation/ pseudo_label_reliability.png, the FRAG-RC trust audit on the evaluation split
    09_language_model/    llm_rescoring_tuning.png, llm_rescoring.json
    10_explanations/      samples/<tag>/ (dashboards + single panels), layer_probes.png/.csv, counterfactuals/
    11_translation/       translations.json, translation_ablation.png, panels/
    12_report/            index.html + the numbers behind every panel (inklingua/report.py)
"""
from __future__ import annotations

import csv
import html
import json
import os
import shutil
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import Config, ModelConfig, dc_from_dict
from .data import HandwritingDataModule
from .decoding import ctc_prefix_beam_search, greedy_decode, log_softmax_np, seq_confidence
from .explain import (ALIGN_CAP, SAMPLE_VIZ, adaptive_calibration, chlabel, code_char_stats, contrast_rollout, coverage_at_risk, pca2,
                      risk_at_coverage, risk_coverage, save_fig)
from .frag import FragModel, cell_theta, char_evidence, char_wrong_labels
from .interpret import InterpretMixin
from .llm import Hypothesis, LLMEngine, calibrate_word_mark, fuse_scores
from .model import InkLinguaNet, ctc_per_sample
from .posthoc import PosthocMixin
from .protocol import log_test_look
from .pseudo import PERTURBATIONS, RELIABILITY_FEATURES, ReliabilityModel, SelectionRule, apply_perturbation, reliability_features, reliability_views
from .render import compute_style_descriptors, render_styled_text_info, render_styled_text_open_loop, style_mismatch
from .riskcontrol import auroc, risk_control_threshold
from .style import NUM_DESC, STYLE_NAMES
from .text import (CharCodec, align_chars, bootstrap_cer_ci, cer, chrf, detokenize, edit_distance, grapheme_clusters, is_linguistic, paired_bootstrap_delta,
                   paired_mean_ci, ridge_cv_r2, text_baseline_r2)
from .utils import LOG, json_safe, to_device, torch_load

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

FOLDERS = ("00_summary", "01_recognition", "02_uncertainty", "03_error_structure", "04_posthoc", "05_style", "06_codebooks", "07_robustness",
           "08_self_distillation", "09_language_model", "10_explanations", "11_translation", "12_report")


class Inferencer(PosthocMixin, InterpretMixin):
    def __init__(self, cfg: Config, data: HandwritingDataModule, llm: LLMEngine, device: torch.device):
        self.cfg, self.data, self.llm, self.device, self.ic = cfg, data, llm, device, cfg.infer
        self.model: Optional[InkLinguaNet] = None
        self.ckpt_info: Dict[str, Any] = {}
        self.out = cfg.infer_dir
        self.dirs = {f: os.path.join(self.out, f) for f in FOLDERS}
        self.fig_list: List[Tuple[str, str]] = []
        self.frag_state: Optional[Dict[str, Any]] = None
        self.frag_model: Optional[FragModel] = None
        self._wsd_lp: Dict[int, Dict[int, np.ndarray]] = {}
        self._ens_views: Tuple[str, ...] = ()
        self.run_extra: Dict[str, Any] = {}

    # ------------------------------------------------------------------ loading
    def _resolve(self, name: str) -> str:
        d = self.cfg.ckpt_dir
        if name == "best":
            p = os.path.join(d, "best.pt")
            return p if os.path.exists(p) else os.path.join(d, "last.pt")
        if name == "last":
            return os.path.join(d, "last.pt")
        return name

    def load(self, path: Optional[str] = None) -> None:
        p = path or self._resolve(self.ic.checkpoint)
        if not os.path.exists(p):
            raise FileNotFoundError(f"No checkpoint at {p}. Train first (cfg.run.do_train=True).")
        st = torch_load(p)
        weights = st["weights"] if "weights" in st else st["ema"]
        mcfg = dc_from_dict(ModelConfig, st["cfg"]["model"]) if "cfg" in st else self.cfg.model
        units = list(st.get("units") or grapheme_clusters(st.get("chars", self.data.codec.chars)))
        if units != self.data.codec.units:
            raise RuntimeError(f"the checkpoint was trained with a different alphabet ({len(units)} units) than the one derived from the current data "
                               f"configuration ({len(self.data.codec.units)}).  Use the same cfg.data (sources, seed, labeled_fraction, limits) and "
                               "cfg.run.name that produced the checkpoint (mode 'finetune' is the way to continue on other data).")
        ck_h = (st.get("cfg") or {}).get("data", {}).get("img_height")
        if ck_h is not None and ck_h != self.cfg.data.img_height:
            raise RuntimeError(f"checkpoint img_height={ck_h} but cfg.data.img_height={self.cfg.data.img_height}")
        self.model = InkLinguaNet(mcfg, len(units) + 1, NUM_DESC, st.get("n_clusters", self.data.n_clusters), self.cfg.data.img_height).to(self.device)
        self.model.load_state_dict(weights)
        self.model.eval()
        fs = st.get("frag_state")
        self.frag_state = fs
        self.frag_model = FragModel.from_state(fs["model"]) if fs else None
        if fs and fs.get("n_strata") not in (None, self.data.n_strata):
            LOG.warning("the FRAG-RC rule has %s cells but the data has %d: cell indices are clipped", fs.get("n_strata"), self.data.n_strata)
        self.ckpt_info = {"path": p, "epoch": st.get("epoch"), "weights_from": st.get("from", "ema"), "val_sel_cer_at_save": st.get("val_cer"),
                          "frag_rule_epoch": (fs or {}).get("epoch"), "domains": st.get("domains")}
        LOG.info("Loaded %s (epoch %s, %s weights)", p, st.get("epoch"), self.ckpt_info["weights_from"])

    def _fig(self, rel: str, caption: str, fig) -> None:
        save_fig(fig, os.path.join(self.out, rel))
        self.fig_list.append((rel, caption))

    # ------------------------------------------------------------------ passes over a dataset
    @torch.no_grad()
    def _predict(self, ds, indices, perturb=None) -> Tuple[List[str], np.ndarray]:
        preds, conf = [""] * len(ds), np.zeros(len(ds))
        for batch in self.data.eval_loader(ds, indices):
            batch = to_device(batch, self.device)
            x = batch["image"] if perturb is None else apply_perturbation(perturb, batch["image"], batch["width"])
            out = self.model(x, batch["width"])
            lp, lens = out["log_probs"].cpu().numpy(), out["out_lens"].cpu().tolist()
            for b in range(lp.shape[0]):
                ids, cf, _ = greedy_decode(lp[b, :lens[b]])
                i = int(batch["index"][b])
                preds[i], conf[i] = self.data.codec.decode_ids(ids), seq_confidence(cf)
        return preds, conf

    @staticmethod
    def _random_subset(ids: Sequence[int], k: int, seed: int) -> List[int]:
        ids = list(ids)
        if k >= len(ids):
            return sorted(int(i) for i in ids)
        return sorted(int(i) for i in np.random.RandomState(seed).permutation(ids)[:k])

    @torch.no_grad()
    def _collect(self, ds, indices: Optional[Sequence[int]] = None, extras: bool = True, cons: bool = False, keep_lp: Optional[Sequence[int]] = None) -> Dict[str, Any]:
        """One pass: readings (visual order), confidences, reliability features, style predictions, embeddings, codebook usage.  `ds.oracle` is
        read directly (evaluation code); for unlabeled TRAIN lines it is the hidden transcription, used for diagnostics only."""
        m, codec, pc = self.model, self.data.codec, self.cfg.train.pseudo
        Kl, Kh, V = m.vq_low.K, m.vq_high.K, codec.num_classes
        views = reliability_views(pc) if cons else ()
        n_gate = len(pc.cons_views) if (cons and pc.consistency) else 0
        keep = set(int(i) for i in keep_lp) if keep_lp is not None else set()
        R: Dict[str, Any] = {k: [] for k in ("idx", "gt", "pred", "conf", "char_conf", "ed", "reflen", "ed_w", "reflen_w", "style_pred", "style_raw", "cl_pred",
                                             "cl_true", "style_emb", "content", "gamma", "X", "cons")}
        R["lp"] = {}
        R["use_low"], R["use_high"], R["code_char"] = np.zeros(Kl), np.zeros(Kh), np.zeros((Kh, V))
        for batch in tqdm(self.data.eval_loader(ds, indices), desc="inference pass", leave=False):
            batch = to_device(batch, self.device)
            out = m(batch["image"], batch["width"], extras=extras)
            lp, lens = out["log_probs"].cpu().numpy(), out["out_lens"].cpu().tolist()
            B = lp.shape[0]
            cons_v = np.zeros((B, len(views)))
            reads = [greedy_decode(lp[b, :lens[b]]) for b in range(B)]
            for vi, v in enumerate(views):
                xv = apply_perturbation(PERTURBATIONS[v], batch["image"], batch["width"])
                ov = m(xv, batch["width"])
                lpv, lnv = ov["log_probs"].cpu().numpy(), ov["out_lens"].cpu().tolist()
                for b in range(B):
                    cons_v[b, vi] = cer(codec.decode_ids(reads[b][0]), codec.decode_ids(greedy_decode(lpv[b, :lnv[b]])[0]))
            if extras:
                il, ih = out["idx_low"].cpu().numpy(), out["idx_high"].cpu().numpy()
            for b in range(B):
                T = lens[b]
                ids, cf, fr = reads[b]
                text, i_ds = codec.decode_ids(ids), int(batch["index"][b])
                gt = ds.oracle[i_ds]
                R["idx"].append(i_ds)
                R["gt"].append(gt), R["pred"].append(text), R["conf"].append(seq_confidence(cf))
                R["char_conf"].append(cf)
                R["ed"].append(edit_distance(gt, text)), R["reflen"].append(len(gt))
                R["ed_w"].append(edit_distance(gt.split(), text.split())), R["reflen_w"].append(len(gt.split()))
                R["X"].append(reliability_features(lp[b, :T], ids, cf, fr, cons_v[b]))
                R["cons"].append(float(cons_v[b, :n_gate].mean()) if n_gate else float("nan"))
                if i_ds in keep:
                    R["lp"][i_ds] = lp[b, :T].astype(np.float16)
                if extras:
                    R["use_low"] += np.bincount(il[b, :T], minlength=Kl)
                    R["use_high"] += np.bincount(ih[b, :T], minlength=Kh)
                    am = lp[b, :T].argmax(1)
                    for t in np.flatnonzero(am != 0):
                        R["code_char"][ih[b, t], am[t]] += 1
            R["style_pred"].append(out["style_reg"].cpu().numpy())
            R["style_raw"].append(ds.style_raw[batch["index"].cpu().numpy()])
            R["cl_pred"].append(out["style_cls"].argmax(1).cpu().numpy())
            R["cl_true"].append(batch["cluster"].cpu().numpy())
            R["style_emb"].append(out["style_emb"].cpu().numpy())
            R["content"].append(out["content_pool"].cpu().numpy())
            if extras and out.get("film_gamma") is not None:
                R["gamma"].append(out["film_gamma"].float().cpu().numpy())
        for k in ("style_pred", "style_raw", "cl_pred", "cl_true", "style_emb", "content"):
            R[k] = np.concatenate(R[k])
        R["conf"], R["ed"], R["reflen"] = np.array(R["conf"]), np.array(R["ed"]), np.array(R["reflen"])
        R["ed_w"], R["reflen_w"] = np.array(R["ed_w"]), np.array(R["reflen_w"])
        R["X"], R["cons"] = np.array(R["X"]), np.array(R["cons"])
        R["cer_ps"] = R["ed"] / np.maximum(R["reflen"], 1)
        R["gamma"] = np.concatenate(R["gamma"]) if R["gamma"] else np.zeros((0, 1))
        return R

    # ------------------------------------------------------------------ per-sample explanation
    def _explain(self, ds, i: int, full: bool = True, tag: str = "") -> Dict[str, Any]:
        dev, codec, model = self.device, self.data.codec, self.model
        ink = ds.get_ink(i)
        H, W0 = ink.shape
        x = torch.from_numpy(np.pad(ink, ((0, 0), (0, (-W0) % 4))))[None, None].to(dev)
        w = torch.tensor([W0], device=dev)
        with torch.no_grad():
            out = model(x, w, extras=True, need_attn=full)
        T = int(out["out_lens"][0])
        lp = out["log_probs"][0, :T].cpu().numpy()
        ids, cf, fr = greedy_decode(lp)
        pred, gt = codec.decode_ids(ids), ds.oracle[i]
        sp = out["style_reg"][0].cpu().numpy()
        ctx: Dict[str, Any] = {
            "i": i, "tag": tag, "img": ink, "W": W0, "H": H, "T": T, "gt": gt, "pred": pred, "gt_display": ds.display(i), "pred_display": ds.display(i, pred),
            "detok": ds.detok(i), "conf": seq_confidence(cf), "char_conf": cf, "frames": fr, "ids": ids, "probs": np.exp(lp), "units": codec.units,
            "pred_units": [codec.i2c.get(c, "?") for c in ids], "cer": cer(gt, pred), "ops": align_chars(gt, pred),
            "nbest": [(codec.decode_ids(a), l) for a, l in ctc_prefix_beam_search(lp, self.ic.beam)],
            "style_pred": self.data.denorm_style(sp), "style_true": ds.style_raw[i].astype(np.float64),
            "cl_pred": int(out["style_cls"][0].argmax()), "cl_true": int(ds.cluster[i]),
            "idx_low": out["idx_low"][0, :T].cpu().numpy(), "idx_high": out["idx_high"][0, :T].cpu().numpy(),
            "style_att": out["style_att"][0, :T].cpu().numpy(), "attn_roll": None, "gradxin": np.zeros((H, W0)), "ig": np.zeros((H, W0)),
            "occ_x": np.zeros(1), "occ_d": np.zeros(1), "char_attr": np.zeros((0, W0))}
        if not full:
            return ctx
        if out["attn"]:
            Tp = out["attn"][0].shape[-1]
            Rl = torch.eye(Tp)
            for A in out["attn"]:
                A = 0.5 * A[0].float().mean(0).cpu() + 0.5 * torch.eye(Tp)
                Rl = (A / A.sum(-1, keepdim=True)) @ Rl
            ctx["attn_roll"] = Rl.numpy()[:T, :T]
            if ctx["frames"]:
                rows = np.stack([ctx["attn_roll"][f] / max(ctx["attn_roll"][f].sum(), 1e-9) for f in ctx["frames"]])
                far = np.abs(np.arange(T)[None] - np.asarray(ctx["frames"])[:, None]) > 3
                ctx["attn_nonlocal"] = float((rows * far).sum(1).mean())
                ctx["attn_sink"] = contrast_rollout(rows)[1]
        if ids:
            with torch.enable_grad():
                ctx.update(self._grad_maps(x, w, ids, fr, W0))
        return ctx

    def _grad_maps(self, x, w, ids, frames, W0) -> Dict[str, Any]:
        dev, model = self.device, self.model
        K = len(ids)
        tgt, tl = torch.tensor(ids, device=dev), torch.tensor([K], device=dev)
        xg = x.clone().requires_grad_(True)
        out = model(xg, w)
        lp = out["log_probs"]
        loss = ctc_per_sample(lp, out["out_lens"], tgt, tl).sum()
        g, = torch.autograd.grad(loss, xg, retain_graph=True)
        gxi = (g * xg).abs()[0, 0, :, :W0].detach().cpu().numpy()
        rows = []
        for k in range(min(K, 70)):
            gk, = torch.autograd.grad(lp[0, frames[k], ids[k]], xg, retain_graph=True)
            rows.append((gk * xg).abs()[0, 0].sum(0)[:W0].detach().cpu().numpy())
        import cv2
        A = np.stack(rows)
        A = np.stack([cv2.GaussianBlur(r[None].astype(np.float32), (0, 0), 3)[0] for r in A])
        A = A / (A.max(1, keepdims=True) + 1e-9)
        S = self.ic.ig_steps
        xb = (torch.linspace(1.0 / S, 1.0, S, device=dev).view(-1, 1, 1, 1) * x).requires_grad_(True)
        ob = model(xb, w.repeat(S))
        lb = ctc_per_sample(ob["log_probs"], ob["out_lens"], tgt.repeat(S), tl.repeat(S))
        gb, = torch.autograd.grad((-lb).sum(), xb)
        ig = (x[0, 0] * gb.mean(0)[0])[:, :W0].detach().cpu().numpy()
        with torch.no_grad():
            win, stride = self.ic.occl_window, self.ic.occl_stride
            starts = list(range(0, max(1, W0 - win + 1), stride))
            xs = x.repeat(len(starts), 1, 1, 1)
            for n, s0 in enumerate(starts):
                xs[n, :, :, s0:s0 + win] = 0
            base = float(ctc_per_sample(*self._fw(x, w), tgt, tl))
            ds_ = []
            for c in range(0, len(starts), 32):
                xc = xs[c:c + 32]
                lo, ln = self._fw(xc, w.repeat(len(xc)))
                ds_ += (ctc_per_sample(lo, ln, tgt.repeat(len(xc)), tl.repeat(len(xc))) - base).cpu().tolist()
        return {"gradxin": gxi, "char_attr": A, "ig": ig, "occ_x": np.array(starts) + win / 2.0, "occ_d": np.array(ds_)}

    def _fw(self, x, w):
        o = self.model(x, w)
        return o["log_probs"], o["out_lens"]

    def _sample_figs(self, ctx, tag: str) -> None:
        W = ctx["W"]
        wi = float(np.clip(W / 90.0, 8, 17))
        n, ratios = len(SAMPLE_VIZ), [r for _, _, r in SAMPLE_VIZ]
        fig, axes = plt.subplots(n, 1, figsize=(wi, 0.85 * sum(ratios) + 1.0), gridspec_kw={"height_ratios": ratios})
        for ax, (_, fn, _) in zip(axes, SAMPLE_VIZ):
            fn(ax, ctx)
            ax.set_xlim(0, W)
        axes[-1].set_xlabel("pixel column (frame stride = 4 px)")
        self._fig(f"10_explanations/samples/{tag}/dashboard.png", f"Explanation dashboard - {tag}", fig)
        for name, fn, r in SAMPLE_VIZ:
            f, ax = plt.subplots(figsize=(wi, 1.1 * r + 1.0))
            fn(ax, ctx)
            ax.set_xlim(0, W)
            save_fig(f, os.path.join(self.dirs["10_explanations"], "samples", tag, f"{name}.png"))

    # ------------------------------------------------------------------ translation
    def _hyps(self, ds, ctx, nbest: int) -> List[Hypothesis]:
        """Hypotheses in LOGICAL order (RTL lines are reversed back); confidences follow the units."""
        nb = ctx["nbest"]
        lp0 = dict(nb).get(ctx["pred"], max(l for _, l in nb))
        alts = [(t, l) for t, l in nb if t != ctx["pred"]][:max(0, nbest - 1)]
        lps = np.array([lp0] + [l for _, l in alts])
        post = np.exp(lps - lps.max())
        post /= post.sum()
        i = ctx["i"]
        cf = list(ctx["char_conf"])
        if bool(ds.rtl[i]):
            cf = cf[::-1]
        return [Hypothesis(ds.display(i, ctx["pred"]), cf, float(lp0), float(post[0]))] + [
            Hypothesis(ds.display(i, t), [], float(l), float(p)) for (t, l), p in zip(alts, post[1:])]

    def _rescored_reading(self, ds, ctx) -> Optional[str]:
        rs = getattr(self, "_rescore", None)
        if not rs or not rs.get("adopted") or not self.llm.can_score() or not ctx.get("nbest"):
            return None
        nb = ctx["nbest"]
        i = ctx["i"]
        lm, nt = self.llm.score_texts([detokenize(ds.display(i, t)) if ds.detok(i) else ds.display(i, t) for t, _ in nb], ds.language(i))
        return nb[int(np.argmax(fuse_scores(np.array([l for _, l in nb]), lm, nt, rs["beta"], rs["rho"])))][0]

    def _translate_block(self, ds, ctxs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cascade evaluation (handwriting -> text -> translation).  Headline numbers use ONLY the randomly drawn lines.  Arms: plain = top-1
        reading, plain prompt | uncertain = marked text + N-best, uncertain prompt | rescored = the validated LLM-rescored reading.  The primary
        `translation` column follows cfg.llm.use_uncertainty_prompt.  Every arm is scored against an oracle made with ITS OWN prompt."""
        if not ctxs:
            return {}
        if not self.llm.available():
            LOG.warning("Translation skipped (no LLM available).")
            return {}
        rows, n_panels = [], 0
        risk_of = getattr(self, "_risk_by_idx", {})
        use_unc = bool(self.cfg.llm.use_uncertainty_prompt)
        for ctx in tqdm(ctxs, desc="translating", leave=False):
            i = ctx["i"]
            detok = ds.detok(i)
            src_lang = ds.language(i)
            tgt_lang = self.cfg.llm.target_language if src_lang != self.cfg.llm.target_language else ("en" if src_lang != "en" else "fr")
            T = lambda h, unc_: self.llm.translate(h, uncertain=unc_, detok=detok, source_language=src_lang, target_language=tgt_lang)  # noqa: E731
            hyps = self._hyps(ds, ctx, self.cfg.llm.nbest)
            unc = T(hyps, True)
            plain = T(hyps, False)["translation"]
            row = {"index": i, "tag": ctx.get("tag", "random"), "source_language": src_lang, "target_language": tgt_lang,
                   "gt": ctx["gt_display"], "pred": ctx["pred_display"], "cer": ctx["cer"],
                   "marked_text": unc["marked_text"], "uncertain_translation": unc["translation"], "plain_translation": plain,
                   "translation": unc["translation"] if use_unc else plain, "prompt_used": "uncertain" if use_unc else "plain",
                   "risk": risk_of.get(int(i), float("nan"))}
            if self.ic.translation_ablation:
                ora = T([Hypothesis(ctx["gt_display"])], False)["translation"]
                ora_u = T([Hypothesis(ctx["gt_display"], [1.0] * len(ctx["gt_display"]))], True)["translation"]
                row.update({"oracle_translation": ora, "oracle_translation_uncertain_prompt": ora_u, "chrf_plain_vs_oracle": chrf(plain, ora),
                            "chrf_uncertain_vs_oracle": chrf(unc["translation"], ora), "chrf_uncertain_vs_oracle_same_prompt": chrf(unc["translation"], ora_u)})
                rr = self._rescored_reading(ds, ctx)
                if rr is not None:
                    rt = plain if rr == ctx["pred"] else T([Hypothesis(ds.display(i, rr))], False)["translation"]
                    row.update({"rescored_reading": ds.display(i, rr), "rescored_translation": rt, "chrf_rescored_vs_oracle": chrf(rt, ora)})
            if self.ic.render_translation and row["translation"] and n_panels < self.ic.render_translation_max:
                self._translation_panel(ctx, row)
                n_panels += 1
            rows.append(row)
        rnd = [r for r in rows if r["tag"] == "random"]
        summ: Dict[str, Any] = {"n": len(rows), "n_random": len(rnd), "target_language": self.cfg.llm.target_language, "source_language": self.cfg.llm.source_language,
                                "primary_prompt": "uncertain" if use_unc else "plain", "rows": rows}
        if self.ic.translation_ablation and rnd:
            blk = max(1, int(self.ic.bootstrap_block))
            gr = [int(r['index']) // blk for r in rnd]
            pl = [r["chrf_plain_vs_oracle"] for r in rnd]
            un = [r["chrf_uncertain_vs_oracle"] for r in rnd]
            un_m = [r["chrf_uncertain_vs_oracle_same_prompt"] for r in rnd]
            summ.update({"chrf_plain_vs_oracle": float(np.mean(pl)), "chrf_uncertain_vs_oracle_plain_prompt_oracle": float(np.mean(un)),
                         "chrf_uncertain_vs_oracle_same_prompt": float(np.mean(un_m)),
                         "chrf_uncertain_minus_plain_prompt_confounded": paired_mean_ci(un, pl, self.ic.bootstrap, 0, gr),
                         "chrf_uncertain_minus_plain_same_prompt_oracles": paired_mean_ci(un_m, pl, self.ic.bootstrap, 0, gr)})
            ok_ = [r for r in rnd if r["cer"] == 0]
            if ok_:
                summ["prompt_effect_on_error_free_lines"] = {
                    "n": len(ok_), "chrf_uncertain_prompt_vs_plain_prompt": float(np.mean([chrf(r["uncertain_translation"], r["oracle_translation"]) for r in ok_])),
                    "note": "100 = the two prompts give the same translation of a correctly read line; the shortfall is prompt variance, not HTR error"}
            err = [r for r in rnd if r["cer"] > 0]
            summ["n_random_lines_with_errors"] = len(err)
            if err:
                e_pl, e_um = [r["chrf_plain_vs_oracle"] for r in err], [r["chrf_uncertain_vs_oracle_same_prompt"] for r in err]
                summ.update({"chrf_plain_on_lines_with_errors": float(np.mean(e_pl)), "chrf_uncertain_same_prompt_on_lines_with_errors": float(np.mean(e_um)),
                             "chrf_uncertain_minus_plain_same_prompt_on_lines_with_errors": paired_mean_ci(e_um, e_pl, self.ic.bootstrap, 0, [int(r['index']) // blk for r in err])})
            rs_rows = [r for r in rnd if "chrf_rescored_vs_oracle" in r]
            if rs_rows:
                summ["rescored_arm"] = {"n": len(rs_rows), "reading_changed_share": float(np.mean([r["rescored_reading"] != r["pred"] for r in rs_rows])),
                                        "chrf_rescored": float(np.mean([r["chrf_rescored_vs_oracle"] for r in rs_rows])),
                                        "chrf_rescored_minus_plain": paired_mean_ci([r["chrf_rescored_vs_oracle"] for r in rs_rows], [r["chrf_plain_vs_oracle"] for r in rs_rows],
                                                                                    self.ic.bootstrap, 0, [int(r['index']) // blk for r in rs_rows])}
            elif getattr(self, "_rescore", None) is not None and not self._rescore.get("adopted"):
                summ["rescored_arm"] = {"skipped": "LLM N-best rescoring showed no significant gain on the val_gate lines, so it is not used"}
            rk = np.array([r["risk"] for r in rnd], np.float64)
            if np.isfinite(rk).all() and len(rnd) >= 10:
                o = np.argsort(rk, kind="stable")
                sel = {}
                for cov in (1.0, 0.8, 0.6, 0.4):
                    k = max(1, int(round(cov * len(rnd))))
                    keep = o[:k]
                    sel[f"{int(cov * 100)}%"] = {"n_answered": int(k), "chrf_plain": float(np.mean([pl[i] for i in keep])),
                                                 "chrf_uncertain_same_prompt": float(np.mean([un_m[i] for i in keep])),
                                                 "mean_cer_of_answered_lines": float(np.mean([rnd[i]["cer"] for i in keep]))}
                summ["selective_translation"] = {"coverage": sel, "ranking": "learned reliability score of the reading (val_gate-fitted)",
                                                 "note": "lines above the coverage cut-off would be answered with '[illegible]' instead of a translation"}
            fid = [r for r in rows if "rendered_style" in r and r.get("font_covers_text", True)]
            if fid:
                ps_ = np.array([r["predicted_style"] for r in fid], np.float64)
                rs_c = np.array([r["rendered_style"] for r in fid], np.float64)
                rs_o = np.array([r["rendered_style_open_loop"] for r in fid], np.float64)
                with np.errstate(all="ignore"):
                    summ["style_transfer_fidelity"] = {
                        "n": len(fid), "descriptors": STYLE_NAMES,
                        "mae_closed_loop": [float(np.nanmean(np.abs(ps_[:, j] - rs_c[:, j]))) for j in range(NUM_DESC)],
                        "mae_open_loop_v3": [float(np.nanmean(np.abs(ps_[:, j] - rs_o[:, j]))) for j in range(NUM_DESC)],
                        "mean_scaled_mismatch_closed_loop": float(np.mean([r["style_mismatch"] for r in fid])),
                        "mean_scaled_mismatch_open_loop_v3": float(np.mean([r["style_mismatch_open_loop"] for r in fid]))}
            fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
            labs, a, b = [f"random lines (n={len(rnd)})"], [summ["chrf_plain_vs_oracle"]], [summ["chrf_uncertain_vs_oracle_same_prompt"]]
            if err:
                labs.append(f"... with HTR errors (n={len(err)})")
                a.append(summ["chrf_plain_on_lines_with_errors"])
                b.append(summ["chrf_uncertain_same_prompt_on_lines_with_errors"])
            xx = np.arange(len(labs))
            ax[0].bar(xx - 0.18, a, 0.36, label="plain top-1 (oracle: plain prompt)")
            ax[0].bar(xx + 0.18, b, 0.36, label="uncertainty-aware (oracle: same prompt)")
            ax[0].set_xticks(xx), ax[0].set_xticklabels(labs, fontsize=8), ax[0].set_ylabel("chrF vs translation of the TRUE text")
            d_ = summ["chrf_uncertain_minus_plain_same_prompt_oracles"]
            ax[0].set_title(f"Cascade loss, prompt-matched oracles\nuncertain - plain = {d_['delta']:+.2f} chrF [{d_['lo']:+.2f}, {d_['hi']:+.2f}]", fontsize=8)
            ax[0].legend(fontsize=7), ax[0].set_ylim(0, 105)
            if "selective_translation" in summ:
                cv = summ["selective_translation"]["coverage"]
                ax[1].plot(list(cv), [v["chrf_plain"] for v in cv.values()], "o-", label="plain")
                ax[1].plot(list(cv), [v["chrf_uncertain_same_prompt"] for v in cv.values()], "s-", label="uncertainty-aware")
                ax[1].set_xlabel("share of lines translated (most reliable first); rest -> '[illegible]'"), ax[1].set_ylabel("chrF of the answered lines")
                ax[1].set_title("selective translation", fontsize=8), ax[1].legend(fontsize=7)
            else:
                ax[1].axis("off")
            self._fig("11_translation/translation_ablation.png", "Translation: plain vs uncertainty-aware (prompt-matched) and selective translation", fig)
        return summ

    def _translation_panel(self, ctx, row) -> None:
        info = render_styled_text_info(row["translation"], ctx["style_pred"], ctx["H"], iters=6)
        styled, covered, rs = info["image"], info["covered"], info["measured"]
        op, _ = render_styled_text_open_loop(row["translation"], ctx["style_pred"], ctx["H"])
        ro = compute_style_descriptors(np.pad(op, ((0, 0), (8, 8))))
        tgt = np.asarray(ctx["style_pred"], np.float64)
        row.update({"rendered_style": rs.tolist(), "rendered_style_open_loop": ro.tolist(), "predicted_style": list(map(float, ctx["style_pred"])),
                    "font_covers_text": bool(covered), "style_mismatch": float(style_mismatch(rs, tgt)), "style_mismatch_open_loop": float(style_mismatch(ro, tgt))})
        fig = plt.figure(figsize=(float(np.clip(ctx["W"] / 90, 8, 14)), 5.6))
        gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.4])
        a0, a1, a2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        a0.imshow(1 - ctx["img"], cmap="gray", aspect="equal"), a0.set_title("handwritten input", fontsize=8, loc="left")
        a1.imshow(1 - styled, cmap="gray", aspect="equal")
        f_ = lambda v, d=2: "n/a" if not np.isfinite(v) else f"{v:.{d}f}"          # noqa: E731
        a1.set_title(f"translation ({self.cfg.llm.target_language}) rendered with the PREDICTED style [closed-loop]  -  slant {tgt[0]:+.0f}/{f_(rs[0], 0)} deg | "
                     f"stroke {tgt[1]:.3f}/{f_(rs[1], 3)} | ink {tgt[2]:.2f}/{f_(rs[2])} | gap {tgt[3]:.2f}/{f_(rs[3])} | body {tgt[4]:.2f}/{f_(rs[4])}  "
                     f"(predicted/rendered; mismatch {row['style_mismatch']:.2f} vs {row['style_mismatch_open_loop']:.2f} open-loop)"
                     + ("" if covered else "  [font lacks some glyphs]"), fontsize=6.5, loc="left")
        for a in (a0, a1):
            a.set_xticks([]), a.set_yticks([])
        a2.axis("off")
        txt = f"GT        : {row['gt']}\nHTR       : {row['pred']}   (CER {row['cer']:.3f})\nsent to LLM: {row['marked_text']}\ntranslation: {row['translation']}"
        if "oracle_translation" in row:
            txt += f"\nplain top-1 : {row['plain_translation']}   (chrF {row['chrf_plain_vs_oracle']:.1f})\noracle(GT)  : {row['oracle_translation']}\n" \
                   f"uncertain   : chrF {row['chrf_uncertain_vs_oracle']:.1f} vs plain-prompt oracle, {row['chrf_uncertain_vs_oracle_same_prompt']:.1f} vs same-prompt oracle"
        a2.text(0, 1, txt, va="top", ha="left", fontsize=7.5, family="monospace", wrap=True)
        self._fig(f"11_translation/panels/line_{ctx['i']:05d}.png", f"Translation panel - line {ctx['i']} [{row['tag']}]", fig)

    # ------------------------------------------------------------------ recognition errors
    def _align_cache(self, R: Dict[str, Any], sub_ids: Sequence[int]) -> None:
        keep = set(int(i) for i in sub_ids)
        R["ops"] = {k: align_chars(R["gt"][k], R["pred"][k]) for k, i in enumerate(R["idx"]) if i in keep}

    def _recognition_block(self, R: Dict[str, Any], ds) -> Dict[str, Any]:
        M: Dict[str, Any] = {}
        n = len(R["gt"])
        ed, rl = R["ed"].astype(np.float64), np.maximum(R["reflen"], 1).astype(np.float64)
        ling = np.array([is_linguistic(g) for g in R["gt"]])
        M["n_nonlinguistic_lines"] = int((~ling).sum())
        M["CER_linguistic"] = float(ed[ling].sum() / max(rl[ling].sum(), 1))
        ed_ns = np.array([edit_distance(g.replace(" ", ""), p.replace(" ", "")) for g, p in zip(R["gt"], R["pred"])], np.float64)
        rl_ns = np.array([max(len(g.replace(" ", "")), 1) for g in R["gt"]], np.float64)
        M["CER_ignoring_spaces"] = float(ed_ns.sum() / rl_ns.sum())
        o = np.argsort(-ed)
        M["error_concentration"] = {f"share_of_edits_in_worst_{p}pct_lines": float(ed[o[:max(1, int(round(n * p / 100)))]].sum() / max(ed.sum(), 1)) for p in (1, 5, 10)}
        M["lines_with_cer_above_0.5"] = int((R["cer_ps"] > 0.5).sum())
        sub = [k for k in range(n) if k in R["ops"]]
        subs, dels, inss, sp_ops = Counter(), Counter(), Counter(), Counter()
        for k in sub:
            gt, pr = R["gt"][k], R["pred"][k]
            for op, ri, hi in R["ops"][k]:
                if op == "sub":
                    subs[(gt[ri], pr[hi])] += 1
                    sp_ops["sub_with_space"] += int(gt[ri] == " " or pr[hi] == " ")
                elif op == "del":
                    dels[gt[ri]] += 1
                    sp_ops["space_deleted"] += int(gt[ri] == " ")
                elif op == "ins":
                    inss[pr[hi]] += 1
                    sp_ops["space_inserted"] += int(pr[hi] == " ")
        tot_ops = sum(subs.values()) + sum(dels.values()) + sum(inss.values())
        if tot_ops:
            M["space_errors"] = {"n_alignment_lines": len(sub), "share_of_all_char_errors": float((sp_ops["sub_with_space"] + sp_ops["space_deleted"] + sp_ops["space_inserted"]) / tot_ops),
                                 **{k: int(v) for k, v in sp_ops.items()}}
        if len(self.data.specs) > 1:
            blk = max(1, int(self.ic.bootstrap_block))
            M["per_corpus"] = {}
            for d_, name in enumerate(ds.domain_names):
                m = np.array([int(ds.domain[int(i)]) == d_ for i in R["idx"]])
                if m.any():
                    lo, hi = bootstrap_cer_ci(ed[m], rl[m], self.ic.bootstrap, 0, groups=np.asarray(R["idx"])[m] // blk)
                    M["per_corpus"][name] = {"lines": int(m.sum()), "CER": float(ed[m].sum() / rl[m].sum()), "CER_ci95": [lo, hi],
                                             "WER": float(R["ed_w"][m].sum() / max(R["reflen_w"][m].sum(), 1)), "exact_line_accuracy": float((R["cer_ps"][m] == 0).mean())}
        with open(os.path.join(self.dirs["01_recognition"], "worst_lines.txt"), "w", encoding="utf-8") as f:
            f.write("The 40 lines with the most character errors (check them by eye: corpora contain transcription errors and mis-segmented lines).\n"
                    "Lines whose reference has no letter or digit are listed at the end: they are not handwriting.  Right-to-left lines are shown in logical order.\n")
            for j in [j for j in o if ling[j]][:40]:
                i = int(R["idx"][j])
                f.write(f"line {i:5d} [{ds.domain_names[int(ds.domain[i])]}] | edits {int(ed[j]):3d} | CER {R['cer_ps'][j]:.2f} | conf {R['conf'][j]:.3f}\n"
                        f"   GT  : {ds.display(i)}\n   PRED: {ds.display(i, R['pred'][j])}\n")
            for j in [j for j in o if not ling[j]][:10]:
                f.write(f"[non-linguistic reference] line {R['idx'][j]:5d} | edits {int(ed[j]):3d} | GT: {R['gt'][j]!r} | PRED: {R['pred'][j]!r}\n")
        fig, ax = plt.subplots(1, 3, figsize=(16, 4))
        ax[0].hist(R["cer_ps"], bins=40, color="steelblue"), ax[0].set_xlabel("line CER"), ax[0].set_ylabel("#lines")
        ax[0].set_yscale("log"), ax[0].set_title("per-line CER distribution (log count)", fontsize=9)
        ax[1].scatter(R["reflen"], R["cer_ps"], s=6, alpha=0.4), ax[1].set_xlabel("#characters in line"), ax[1].set_ylabel("line CER"), ax[1].set_title("CER vs line length", fontsize=9)
        cs = np.cumsum(np.sort(ed)[::-1]) / max(ed.sum(), 1)
        ax[2].plot(np.arange(1, n + 1) / n * 100, cs * 100), ax[2].set_xlabel("% of lines (worst first)"), ax[2].set_ylabel("% of all character errors"), ax[2].grid(alpha=0.3)
        ax[2].set_title("how concentrated are the errors?", fontsize=9)
        self._fig("01_recognition/cer_distribution.png", "Per-line CER distribution and error concentration", fig)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
        for a, (cnt, ttl) in zip(ax, [(subs, "top substitutions (ref -> hyp)"), (dels, "most deleted chars"), (inss, "most inserted chars")]):
            top = cnt.most_common(15)
            lab = [f"{chlabel(k[0])}->{chlabel(k[1])}" if isinstance(k, tuple) else chlabel(k) for k, _ in top]
            a.barh(range(len(top))[::-1], [v for _, v in top], color="indianred"), a.set_yticks(range(len(top))[::-1])
            a.set_yticklabels(lab, fontsize=8), a.set_title(ttl, fontsize=9)
        self._fig("01_recognition/top_confusions.png", f"Character confusions ({len(sub)} random lines)", fig)
        M["top_confusions"] = [{"ref": k[0], "hyp": k[1], "count": int(v)} for k, v in subs.most_common(30)]
        return M

    # ------------------------------------------------------------------ is the confidence trustworthy?
    def _char_outcomes(self, R: Dict[str, Any], T: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        conf, ok = [], []
        for k, i_ds in enumerate(R["idx"]):
            if k not in R["ops"] or i_ds not in R["lp"]:
                continue
            lp = R["lp"][i_ds].astype(np.float32)
            if T != 1.0:
                lp = log_softmax_np(lp / T)
            _, cf, _ = greedy_decode(lp)
            for op, ri, hi in R["ops"][k]:
                if hi is not None and hi < len(cf):
                    conf.append(cf[hi])
                    ok.append(float(op == "match"))
        return np.array(conf), np.array(ok)

    def _calibrate_marking(self, Rval: Dict[str, Any], same_split: bool) -> Dict[str, Any]:
        info = calibrate_word_mark(Rval["pred"], Rval["gt"], Rval["char_conf"], self.ic.mark_precision)
        if info.get("threshold") is not None:
            self.cfg.llm.word_conf_thresh, self.cfg.llm.word_conf_agg = float(info["threshold"]), info["agg"]
        info["used_threshold"], info["used_agg"] = self.cfg.llm.word_conf_thresh, self.cfg.llm.word_conf_agg
        info["fitted_on"] = "val_gate lines" + (" (SAME lines as evaluated: optimistic)" if same_split else "")
        LOG.info("[CAL-MARK] %s: %s-confidence < %.4f -> flagged words are wrong %s of the time (recall %s, %s of the words flagged)", info["fitted_on"],
                 info["used_agg"], info["used_threshold"], f"{info['precision']:.0%}" if "precision" in info else "n/a",
                 f"{info['recall']:.0%}" if "recall" in info else "n/a", f"{info['share_marked']:.0%}" if "share_marked" in info else "n/a")
        with open(os.path.join(self.dirs["02_uncertainty"], "word_marks.json"), "w") as f:
            json.dump(json_safe(info), f, indent=2)
        return info

    def _calibration_block(self, R: Dict[str, Any], Rval: Dict[str, Any], same_split: bool) -> Dict[str, Any]:
        M: Dict[str, Any] = {}
        grid = np.geomspace(0.6, 6.0, 31)
        nll = []
        for T in grid:
            c, a = self._char_outcomes(Rval, float(T))
            nll.append(adaptive_calibration(c, a, 15)["nll"] if len(c) >= 15 else float("nan"))
        if not np.isfinite(nll).any():
            return {}
        T_star = float(grid[int(np.nanargmin(nll))])
        c0, a0 = self._char_outcomes(R, 1.0)
        c1, a1 = self._char_outcomes(R, T_star)
        cal0, cal1 = adaptive_calibration(c0, a0), adaptive_calibration(c1, a1)
        M["temperature"] = {"T": T_star, "fitted_on": "val_gate lines" + (" (SAME lines as evaluated: optimistic)" if same_split else ""),
                            "char_ece_before": cal0["ece"], "char_ece_after": cal1["ece"], "char_nll_before": cal0["nll"], "char_nll_after": cal1["nll"],
                            "mean_conf_before": cal0["mean_conf"], "mean_conf_after": cal1["mean_conf"], "char_accuracy": cal0["accuracy"], "n_chars": cal0["n"]}
        bad = R["ed"] > 0
        sc_conf = 1.0 - R["conf"]
        if same_split:
            fold = np.random.RandomState(0).permutation(len(R["X"])) % 2
            sc_rel = np.zeros(len(R["X"]))
            for f in (0, 1):
                mdl = ReliabilityModel().fit(R["X"][fold != f], R["cer_ps"][fold != f])
                sc_rel[fold == f] = mdl.predict(R["X"][fold == f])
        else:
            sc_rel = ReliabilityModel().fit(Rval["X"], Rval["cer_ps"]).predict(R["X"])
        R["risk_rel"] = sc_rel
        scores = {"CTC sequence confidence": sc_conf, "learned reliability (val-fitted)": sc_rel,
                  "oracle ordering (true CER)": R["cer_ps"] + 1e-9 * np.random.RandomState(0).rand(len(sc_conf))}
        rcs = {k: risk_coverage(v, R["ed"], R["reflen"]) for k, v in scores.items()}
        base = float(R["ed"].sum() / max(R["reflen"].sum(), 1))
        M["selective_prediction"] = {
            "corpus_CER_at_full_coverage": base,
            "auroc_line_has_error": {"CTC sequence confidence": auroc(sc_conf, bad), "learned reliability": auroc(sc_rel, bad)},
            "aurc": {k: v["aurc"] for k, v in rcs.items()},
            "CER_at_80pct_coverage": {k: risk_at_coverage(v, 0.8) for k, v in rcs.items()},
            "CER_at_50pct_coverage": {k: risk_at_coverage(v, 0.5) for k, v in rcs.items()},
            "coverage_at_CER_leq_2pct": {k: coverage_at_risk(v, 0.02) for k, v in rcs.items()},
            "coverage_at_CER_leq_5pct": {k: coverage_at_risk(v, 0.05) for k, v in rcs.items()}}
        sat = (R["conf"] > 0.95)
        M["confidence_saturation"] = {"share_of_lines_with_conf_above_0.95": float(sat.mean()),
                                      "CER_of_those_lines": float(R["ed"][sat].sum() / max(R["reflen"][sat].sum(), 1)) if sat.any() else float("nan"),
                                      "share_of_erroneous_lines_with_conf_above_0.95": float(sat[bad].mean()) if bad.any() else float("nan")}
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        ax[0].plot([0, 1], [0, 1], "k--", lw=1)
        for cal, lab in ((cal0, f"before (ECE {cal0['ece']:.3f})"), (cal1, f"after T={T_star:.2f} (ECE {cal1['ece']:.3f})")):
            if cal["points"]:
                ax[0].plot([p[0] for p in cal["points"]], [p[1] for p in cal["points"]], "o-", ms=4, label=lab)
        ax[0].set_xlabel("character confidence"), ax[0].set_ylabel("character accuracy"), ax[0].legend(fontsize=8), ax[0].set_title("reliability diagram, equal-mass bins", fontsize=9)
        for k, v in rcs.items():
            ax[1].plot(v["coverage"], v["risk"] * 100, label=f"{k} (AURC {v['aurc'] * 100:.2f})")
        ax[1].axhline(base * 100, color="k", ls=":", lw=1), ax[1].set_xlabel("coverage (fraction of lines answered)"), ax[1].set_ylabel("CER of answered lines (%)")
        ax[1].legend(fontsize=7), ax[1].set_title("selective prediction (risk - coverage)", fontsize=9)
        ok_ = R["cer_ps"] == 0
        ax[2].hist([R["conf"][ok_], R["conf"][~ok_]], bins=25, label=["exact lines", "lines with errors"], stacked=True)
        ax[2].set_xlabel("CTC sequence confidence"), ax[2].set_yscale("log"), ax[2].legend(fontsize=8), ax[2].set_title("confidence vs correctness (log count)", fontsize=9)
        self._fig("02_uncertainty/calibration.png", "Calibration (ECE with temperature) and selective prediction", fig)
        LOG.info("calibration: char ECE %.4f -> %.4f (T=%.2f) | line AUROC CTC conf %.3f vs learned %.3f", cal0["ece"], cal1["ece"], T_star,
                 M["selective_prediction"]["auroc_line_has_error"]["CTC sequence confidence"], M["selective_prediction"]["auroc_line_has_error"]["learned reliability"])
        return M

    # ------------------------------------------------------------------ discrete codebooks
    def _codebook_block(self, R: Dict[str, Any]) -> Dict[str, Any]:
        codec, m, M = self.data.codec, self.model, {}
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
        for a, key, name in zip(ax, ("use_low", "use_high"), ("stroke codebook", "component codebook")):
            u = R[key]
            p = u / max(u.sum(), 1)
            ppl = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
            M[f"{name.split()[0]}_codebook"] = {"size": int(len(u)), "perplexity": ppl, "dead_fraction": float((u == 0).mean())}
            a.bar(range(len(u)), np.sort(u)[::-1], color="teal"), a.set_yscale("log")
            a.set_title(f"{name}: usage (perplexity {ppl:.1f}/{len(u)}, dead {100 * (u == 0).mean():.1f}%)", fontsize=9), a.set_xlabel("code (sorted)")
        self._fig("06_codebooks/codebook_usage.png", "Codebook utilisation on the evaluation split", fig)
        cc_m = R["code_char"]
        rows = np.argsort(-cc_m.sum(1))[:40]
        cols = np.argsort(-cc_m.sum(0))[:30]
        sub_m = cc_m[np.ix_(rows, cols)]
        sub_m = sub_m / np.maximum(sub_m.sum(1, keepdims=True), 1)
        M.update(code_char_stats(cc_m, min_count=20))
        fig, ax = plt.subplots(figsize=(11, 8))
        im = ax.imshow(sub_m, aspect="auto", cmap="magma")
        ax.set_xticks(range(len(cols))), ax.set_xticklabels([chlabel(codec.i2c.get(int(c), "blank")) for c in cols], fontsize=7)
        ax.set_yticks(range(len(rows))), ax.set_yticklabels([f"code {int(r)}" for r in rows], fontsize=6)
        ax.set_title(f"which character does each component code stand for?  (codes with >= 20 emissions: purity {M['component_code_purity']:.2f}, "
                     f"chance {M['component_code_purity_chance']:.2f}, NMI {M['component_code_nmi']:.2f})", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        self._fig("06_codebooks/code_vs_character.png", "Component code -> emitted character (row-normalised)", fig)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for a, vq, key, name in zip(ax, (m.vq_low, m.vq_high), ("use_low", "use_high"), ("stroke", "component")):
            P = pca2(vq.embed.detach().cpu().numpy())
            a.scatter(P[:, 0], P[:, 1], s=8 + 120 * R[key] / max(R[key].max(), 1), alpha=0.6, c=np.log1p(R[key]), cmap="viridis")
            a.set_title(f"{name} codebook (PCA, size ~ usage)", fontsize=9)
        self._fig("06_codebooks/codebook_pca.png", "Codebook geometry", fig)
        return M

    # ------------------------------------------------------------------ connected task: style
    def _style_block(self, R: Dict[str, Any]) -> Dict[str, Any]:
        dm, codec, m, M = self.data, self.data.codec, self.model, {}
        sp, st = dm.denorm_style(R["style_pred"]), R["style_raw"].astype(np.float64)
        okm = np.isfinite(st)
        sd, mu = dm.style_std.astype(np.float64), dm.style_mean.astype(np.float64)
        M["style_measurable_fraction"] = dict(zip(STYLE_NAMES, okm.mean(0).tolist()))
        r2, mae, mae_base, corr = [], [], [], []
        for j in range(NUM_DESC):
            k = okm[:, j]
            if k.sum() < 30:
                r2.append(np.nan), mae.append(np.nan), mae_base.append(np.nan), corr.append(np.nan)
                continue
            y, p = st[k, j], sp[k, j]
            sst = ((y - y.mean()) ** 2).sum()
            r2.append(float(1 - ((y - p) ** 2).sum() / sst) if y.std() / sd[j] >= 0.2 else np.nan)
            mae.append(float(np.abs(y - p).mean())), mae_base.append(float(np.abs(y - mu[j]).mean()))
            corr.append(float(np.corrcoef(y, p)[0, 1]) if p.std() > 1e-9 else np.nan)
        M["style_r2"], M["style_corr"] = dict(zip(STYLE_NAMES, r2)), dict(zip(STYLE_NAMES, corr))
        M["style_mae"], M["style_mae_of_the_train_mean"] = dict(zip(STYLE_NAMES, mae)), dict(zip(STYLE_NAMES, mae_base))
        M["style_r2_text_only_floor_train"] = dict(zip(STYLE_NAMES, dm.text_baseline_r2.tolist()))
        valid = R["cl_true"] >= 0
        M["style_cluster_acc"] = float((R["cl_pred"] == R["cl_true"])[valid].mean()) if valid.any() else float("nan")
        fig, ax = plt.subplots(1, NUM_DESC, figsize=(4 * NUM_DESC, 3.8))
        for j, a in enumerate(ax):
            k = okm[:, j]
            if k.sum() >= 2:
                a.scatter(st[k, j], sp[k, j], s=5, alpha=0.4)
                lo, hi = min(st[k, j].min(), sp[k, j].min()), max(st[k, j].max(), sp[k, j].max())
                a.plot([lo, hi], [lo, hi], "k--", lw=1)
            a.set_title(f"{STYLE_NAMES[j]}  R2={r2[j]:.2f}  r={corr[j]:.2f}\nmeasured on {100 * okm[:, j].mean():.0f}% of lines", fontsize=8)
            a.set_xlabel("target (classical image analysis)"), a.set_ylabel("predicted")
        self._fig("05_style/style_pred_vs_true.png", "Connected task: style-descriptor regression (measured entries only)", fig)
        K = m.style_enc.cls.out_features
        cm = np.zeros((K, K))
        for t, p in zip(R["cl_true"][valid], R["cl_pred"][valid]):
            cm[t, p] += 1
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues"), ax.set_xlabel("predicted style cluster"), ax.set_ylabel("true style cluster")
        ax.set_title(f"style-cluster confusion (acc {M['style_cluster_acc']:.2f})", fontsize=9)
        self._fig("05_style/style_confusion.png", "Style-cluster confusion matrix", fig)
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        Ps, Pc = pca2(R["style_emb"]), pca2(R["content"])
        k0 = okm[:, 0]
        sc = ax[0].scatter(Ps[k0, 0], Ps[k0, 1], c=st[k0, 0], s=6, cmap="coolwarm"), ax[0].set_title("style embedding, colour = slant", fontsize=9)
        fig.colorbar(sc[0], ax=ax[0])
        clv = np.asarray(R["cl_true"]).reshape(-1)
        okc = clv >= 0
        ax[1].scatter(Ps[okc, 0], Ps[okc, 1], c=clv[okc], s=6, cmap="tab10", vmin=-0.5, vmax=9.5), ax[1].set_title("style embedding, colour = style cluster", fontsize=9)
        sc = ax[2].scatter(Pc[k0, 0], Pc[k0, 1], c=st[k0, 0], s=6, cmap="coolwarm"), ax[2].set_title("CONTENT embedding, colour = slant (should be mixed)", fontsize=9)
        fig.colorbar(sc[0], ax=ax[2])
        self._fig("05_style/style_vs_content_embedding.png", "Style vs content embedding spaces (PCA)", fig)
        Z = np.nan_to_num((st - mu) / sd, nan=0.0)
        r2s = ridge_cv_r2(R["style_emb"], Z, okm)
        r2c = ridge_cv_r2(R["content"], Z, okm)
        floor = text_baseline_r2(R["gt"], Z, okm, codec.units)
        lowv = okm.sum(0) < 30
        r2s, r2c, floor = (np.where(lowv, np.nan, v) for v in (r2s, r2c, floor))
        M["probe_r2_style_emb"], M["probe_r2_content_emb"] = dict(zip(STYLE_NAMES, r2s.tolist())), dict(zip(STYLE_NAMES, r2c.tolist()))
        M["probe_r2_transcription_only"] = dict(zip(STYLE_NAMES, floor.tolist()))
        exc = np.maximum(r2c - floor, 0.0)
        M["content_leakage_excess_over_transcription"] = dict(zip(STYLE_NAMES, exc.tolist()))
        M["content_leakage_excess_mean"] = float(np.nanmean(exc)) if np.isfinite(exc).any() else float("nan")
        fig, ax = plt.subplots(1, 2, figsize=(13, 4))
        xx = np.arange(NUM_DESC)
        ax[0].bar(xx - 0.27, r2s, 0.27, label="style embedding"), ax[0].bar(xx, r2c, 0.27, label="content embedding")
        ax[0].bar(xx + 0.27, floor, 0.27, label="transcription only (floor)", color="gray")
        ax[0].set_xticks(xx), ax[0].set_xticklabels(STYLE_NAMES, rotation=20, fontsize=8), ax[0].set_ylabel("5-fold ridge R2 (measured entries)"), ax[0].legend(fontsize=8)
        ax[0].set_title("how much style can be read out of each stream?", fontsize=9)

        def corr_to_pcs(E):
            Ec = E - E.mean(0)
            _, _, Vt = np.linalg.svd(Ec, full_matrices=False)
            Zp = Ec @ Vt[:8].T
            Zs = Zp / (Zp.std(0) + 1e-9)
            C = np.full((NUM_DESC, Zs.shape[1]), np.nan)
            for j in range(NUM_DESC):
                k = okm[:, j]
                if k.sum() > 30:
                    y = (st[k, j] - st[k, j].mean()) / (st[k, j].std() + 1e-9)
                    C[j] = (y[:, None] * Zs[k]).mean(0)
            return C

        Cm = np.concatenate([corr_to_pcs(R["style_emb"]), np.full((NUM_DESC, 1), np.nan), corr_to_pcs(R["content"])], 1)
        im = ax[1].imshow(Cm, cmap="seismic", vmin=-1, vmax=1, aspect="auto")
        ax[1].set_yticks(range(NUM_DESC)), ax[1].set_yticklabels(STYLE_NAMES, fontsize=8), ax[1].set_xticks([3.5, 12.5])
        ax[1].set_xticklabels(["style PCs 1-8", "content PCs 1-8"], fontsize=8), ax[1].set_title("correlation: descriptors vs embedding PCs", fontsize=9)
        fig.colorbar(im, ax=ax[1])
        self._fig("05_style/disentanglement_probe.png", "Content / style disentanglement probe", fig)
        G = R["gamma"]
        if len(G) > 5 and G.shape[1] > 1:
            Gc = G - G.mean(0, keepdims=True)
            M["film"] = {"gamma_rms": float(np.sqrt((G ** 2).mean())), "gamma_between_line_std": float(G.std(0).mean()),
                         "share_of_gamma_energy_that_varies_between_lines": float((Gc ** 2).sum() / max((G ** 2).sum(), 1e-12)),
                         "slant_r2_from_gamma": float(ridge_cv_r2(G, Z[:, :1], okm[:, :1])[0])}
            k0i = np.flatnonzero(okm[:, 0])
            if len(k0i) > 5:
                o = k0i[np.argsort(st[k0i, 0])]
                fig, ax = plt.subplots(figsize=(10, 4))
                vmax = max(float(np.abs(Gc).max()), 1e-6)
                im = ax.imshow(Gc[o], aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                ax.set_xlabel("feature channel"), ax.set_ylabel("lines sorted by measured slant"), fig.colorbar(im, ax=ax)
                ax.set_title(f"style-FiLM scale gamma minus its mean over lines  (between-line share of energy {M['film']['share_of_gamma_energy_that_varies_between_lines']:.2f})", fontsize=8)
                self._fig("05_style/film_gamma.png", "Style FiLM modulation (centred)", fig)
        return M

    # ------------------------------------------------------------------ robustness
    def _robustness(self, ds) -> Dict[str, Any]:
        sub = self._random_subset(range(len(ds)), self.ic.robustness_lines, 5)
        clean, _ = self._predict(ds, sub)
        res = {}
        for name, fn in PERTURBATIONS.items():
            pr, _ = self._predict(ds, sub, None if name == "clean" else fn)
            ed = np.array([edit_distance(ds.oracle[i], pr[i]) for i in sub], np.float64)
            rl = np.array([len(ds.oracle[i]) for i in sub], np.float64)
            lo, hi = bootstrap_cer_ci(ed, rl, self.ic.bootstrap, 0, groups=np.asarray(sub) // max(1, int(self.ic.bootstrap_block)))
            res[name] = {"cer": float(ed.sum() / max(rl.sum(), 1)), "cer_ci95": [lo, hi], "n": len(sub), "consistency": 1 - float(np.mean([cer(clean[i], pr[i]) for i in sub]))}
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        names = list(res)
        yerr = np.array([[res[k]["cer"] - res[k]["cer_ci95"][0], res[k]["cer_ci95"][1] - res[k]["cer"]] for k in names]).T
        ax[0].bar(names, [res[k]["cer"] for k in names], yerr=np.maximum(yerr, 0), color="indianred", capsize=2), ax[0].set_ylabel("CER vs ground truth")
        ax[1].bar(names, [res[k]["consistency"] for k in names], color="seagreen"), ax[1].set_ylabel("consistency = 1 - CER(clean pred, perturbed pred)")
        for a in ax:
            a.tick_params(axis="x", rotation=35, labelsize=8)
        ax[0].set_title(f"robustness to input perturbations ({len(sub)} random lines, 95% CI)", fontsize=9), ax[1].set_title("prediction consistency (model inertia)", fontsize=9)
        self._fig("07_robustness/robustness_profile.png", "Robustness / consistency under perturbations", fig)
        return res

    # ------------------------------------------------------------------ self-distillation diagnostics (hidden ground truth)
    def _pl_reliability(self) -> Dict[str, Any]:
        """How trustworthy are the lines the legacy rule WOULD select?  Checked against the hidden transcriptions of the unlabeled train lines."""
        ds, dm, pc = self.data.train_ds, self.data, self.cfg.train.pseudo
        unl, cal = dm.unlabeled_idx, dm.calib_idx
        if len(unl) == 0:
            return {}
        idx = self._random_subset(unl, self.ic.pl_diag_samples, 3)
        U = self._collect(ds, idx, extras=False, cons=True)
        res: Dict[str, Any] = {"n_unlabeled_lines": len(idx), "CER_on_unlabeled": float(U["ed"].sum() / max(U["reflen"].sum(), 1))}
        bad = U["ed"] > 0
        curves = {"CTC sequence confidence": risk_coverage(1.0 - U["conf"], U["ed"], U["reflen"]),
                  "oracle ordering": risk_coverage(U["cer_ps"] + 1e-9 * np.random.RandomState(0).rand(len(idx)), U["ed"], U["reflen"])}
        res["auroc_ctc_confidence"] = auroc(1.0 - U["conf"], bad)
        if len(cal) >= pc.min_calib:
            C = self._collect(ds, cal, extras=False, cons=True)
            rule = SelectionRule.fit(C["X"], C["ed"].astype(np.float64), C["reflen"].astype(np.float64), pc, seed=self.cfg.run.seed)
            risk = rule.predict(U["X"])
            curves["learned line risk (calibration-fitted)"] = risk_coverage(risk, U["ed"], U["reflen"])
            res["auroc_learned_risk"] = auroc(risk, bad)
            res["calibration"] = {k: float(v) for k, v in rule.info.items()}
            res["selected_by_rule_at_z"] = {}
            for z_ in (0.0, 1.0, pc.risk_z):
                th_, k_, _c = risk_control_threshold(rule.oof, C["ed"].astype(np.float64), C["reflen"].astype(np.float64), pc.target_cer, z_, pc.min_selected_calib, pc.risk_smoothing)
                s_ = risk <= th_ if (k_ > 0 and np.isfinite(th_)) else np.zeros(len(idx), bool)
                res["selected_by_rule_at_z"][f"z={z_}"] = {"calibration_lines_kept": int(k_), "coverage_unlabeled": float(s_.mean()),
                                                         "CER_vs_hidden_GT": float(U["ed"][s_].sum() / max(U["reflen"][s_].sum(), 1)) if s_.any() else float("nan")}
        else:
            LOG.warning("no / too few calibration lines (%d): the learned line reliability could not be evaluated", len(cal))
        sat = U["conf"] > 0.95
        res["ctc_confidence_saturation"] = {"share_of_lines_above_0.95": float(sat.mean()),
                                            "CER_of_those_lines": float(U["ed"][sat].sum() / max(U["reflen"][sat].sum(), 1)) if sat.any() else float("nan")}
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
        for k, v in curves.items():
            ax[0].plot(v["coverage"], v["risk"] * 100, label=f"{k} (AURC {v['aurc'] * 100:.2f})")
        ax[0].axhline(pc.target_cer * 100, color="k", ls="--", lw=1, label=f"target CER {100 * pc.target_cer:.1f}%")
        ax[0].set_xlabel("fraction of unlabeled lines selected"), ax[0].set_ylabel("CER of the selected lines vs hidden GT (%)")
        ax[0].set_yscale("log"), ax[0].legend(fontsize=7), ax[0].set_title("which signal ranks pseudo-label quality?", fontsize=9)
        ths = np.linspace(0.3, 0.999, 24)
        cov = [float((U["conf"] >= t).mean()) for t in ths]
        cc = [float(U["ed"][U["conf"] >= t].sum() / max(U["reflen"][U["conf"] >= t].sum(), 1)) if (U["conf"] >= t).any() else np.nan for t in ths]
        ax[1].plot(ths, cov, "o-", label="coverage"), ax[1].plot(ths, np.array(cc) * 100, "^-", label="CER of selected (%)")
        ax[1].set_xlabel("CTC sequence-confidence threshold"), ax[1].legend(fontsize=8), ax[1].set_title("saturation: raising the confidence threshold hardly reduces the CER", fontsize=9)
        self._fig("08_self_distillation/pseudo_label_reliability.png", "How trustworthy are the line-level pseudo-labels the model would select?", fig)
        return res

    # ------------------------------------------------------------------ [LLM-N] constrained LLM rescoring
    @torch.no_grad()
    def _nbest_pass(self, ds, indices) -> Dict[int, Dict[str, Any]]:
        res: Dict[int, Dict[str, Any]] = {}
        codec = self.data.codec
        ov = self._wsd_lp.get(id(ds))
        if ov is not None:
            miss = [int(i) for i in indices if int(i) not in ov]
            lps = {int(i): ov[int(i)] for i in indices if int(i) in ov}
            if miss:
                lps.update(self._lp_views(miss, ds, ()))
        else:
            lps = self._lp_views([int(i) for i in indices], ds, getattr(self, "_ens_views", ()))
        for i, lp in tqdm(lps.items(), desc="N-best", leave=False):
            nb = ctc_prefix_beam_search(lp, self.ic.rescore_beam)
            res[int(i)] = {"texts": [codec.decode_ids(ids) for ids, _ in nb], "ctc": [float(l) for _, l in nb], "greedy": codec.decode_ids(greedy_decode(lp)[0]),
                           "ref": ds.oracle[int(i)], "lang": ds.language(int(i)), "lm_texts": [detokenize(ds.display(int(i), codec.decode_ids(ids))) if ds.detok(int(i)) else ds.display(int(i), codec.decode_ids(ids)) for ids, _ in nb]}
        return res

    def _llm_rescoring_eval(self, ds) -> Dict[str, Any]:
        ic = self.ic
        if not ic.llm_rescoring:
            return {}
        if not self.llm.can_score():
            LOG.warning("LLM rescoring evaluation skipped (needs cfg.llm.backend='causal' and a loadable model).")
            return {}
        rng = np.random.RandomState(0)
        val_perm = rng.permutation(np.asarray(self.data.val_gate_idx))
        n_tune = min(ic.rescore_val_lines, len(val_perm) // 2 if ds is self.data.val_ds else len(val_perm))
        val_idx = sorted(int(i) for i in val_perm[:n_tune])
        pool = [int(i) for i in val_perm[n_tune:]] if ds is self.data.val_ds else list(range(len(ds)))
        ev_idx = sorted(pool if ic.rescore_test_lines is None else [int(i) for i in np.random.RandomState(1).permutation(pool)[:ic.rescore_test_lines]])
        nb_val, nb_ev = self._nbest_pass(self.data.val_ds, val_idx), self._nbest_pass(ds, ev_idx)

        def lm_scores(nb):
            for lang in sorted({v["lang"] for v in nb.values()}):
                vs = [v for v in nb.values() if v["lang"] == lang]
                lm, nt = self.llm.score_texts([t for v in vs for t in v["lm_texts"]], lang)
                pos = 0
                for v in vs:
                    k = len(v["texts"])
                    v["lm"], v["ntok"] = lm[pos:pos + k], nt[pos:pos + k]
                    pos += k

        lm_scores(nb_val), lm_scores(nb_ev)

        def choose(v, beta, rho):
            return int(np.argmax(fuse_scores(v["ctc"], v["lm"], v["ntok"], beta, rho)))

        def corpus(nb, beta, rho):
            ed = np.array([edit_distance(v["ref"], v["texts"][choose(v, beta, rho)]) for v in nb.values()], np.float64)
            return ed, np.array([len(v["ref"]) for v in nb.values()], np.float64)

        grid: Dict[str, float] = {}
        best = (float("inf"), 0.0, 0.0)
        betas, rhos = list(ic.rescore_betas), list(ic.rescore_rhos)
        done: set = set()
        extensions = rho_ext = 0
        while True:
            for beta in betas:
                for rho in (rhos if beta > 0 else (0.0,)):
                    if (beta, rho) in done:
                        continue
                    done.add((beta, rho))
                    ed, rl = corpus(nb_val, beta, rho)
                    c = float(ed.sum() / max(rl.sum(), 1))
                    grid[f"beta={beta},rho={rho}"] = c
                    if c < best[0] - 1e-12:
                        best = (c, beta, rho)
            if ic.rescore_extend_grid and best[1] >= max(betas) and extensions < 5:
                betas.append(float(max(betas)) * 2.0)
                extensions += 1
                continue
            if ic.rescore_extend_rho and best[1] > 0 and best[2] >= max(rhos) and rho_ext < 3:
                rhos.append(float(max(rhos)) + 2.0)
                rho_ext += 1
                continue
            break
        _, beta, rho = best
        ed_vb, rl_v = corpus(nb_val, 0.0, 0.0)
        ed_vn, _ = corpus(nb_val, beta, rho)
        blk = max(1, int(ic.bootstrap_block))
        gv, ge = np.asarray(list(nb_val.keys())) // blk, np.asarray(list(nb_ev.keys())) // blk
        val_gate = paired_bootstrap_delta(ed_vn, ed_vb, rl_v, ic.bootstrap, 1, groups=gv, n_tests=max(1, len(grid) - 1))
        adopted = bool(beta > 0 and val_gate["hi_adj"] < 0)
        self._rescore = {"beta": float(beta), "rho": float(rho), "adopted": adopted}
        self._nb_eval = nb_ev
        ed_base, rl = corpus(nb_ev, 0.0, 0.0)
        ed_new, _ = corpus(nb_ev, beta, rho)
        ed_greedy = np.array([edit_distance(v["ref"], v["greedy"]) for v in nb_ev.values()], np.float64)
        ed_orcl = np.array([min(edit_distance(v["ref"], t) for t in v["texts"]) for v in nb_ev.values()], np.float64)
        changed = float(np.mean([choose(v, beta, rho) != 0 for v in nb_ev.values()]))
        tot = max(rl.sum(), 1)
        out = {"n_tune_lines": len(nb_val), "n_eval_lines": len(nb_ev), "beta": beta, "rho": rho, "beta_grid_extended_times": extensions,
               "beta_is_interior": bool(beta < max(betas)), "rho_grid_extended_times": rho_ext, "rho_is_interior": bool(rho < max(rhos) or beta == 0),
               "val_cer_grid": grid, "val_cer_baseline": grid.get("beta=0.0,rho=0.0"), "CER_greedy": float(ed_greedy.sum() / tot),
               "CER_beam_top1": float(ed_base.sum() / tot), "CER_rescored": float(ed_new.sum() / tot), "CER_oracle_nbest": float(ed_orcl.sum() / tot),
               "CER_rescored_ci95": list(bootstrap_cer_ci(ed_new, rl, ic.bootstrap, 0, groups=ge)), "fraction_of_lines_changed": changed,
               "val_paired_delta_vs_beam": val_gate, "adopted_on_validation": adopted,
               "paired_bootstrap_rescored_minus_beam": paired_bootstrap_delta(ed_new, ed_base, rl, ic.bootstrap, 0, groups=ge), "llm": self.cfg.llm.model_name}
        LOG.info("[LLM-N] beta=%.2f rho=%.1f (tuned on %d val_gate lines) | eval CER beam %.4f -> rescored %.4f (oracle N-best %.4f) | %s", beta, rho, len(nb_val),
                 out["CER_beam_top1"], out["CER_rescored"], out["CER_oracle_nbest"], "ADOPTED" if adopted else "not adopted")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for r_ in sorted({float(k.split("rho=")[1]) for k in grid}):
            xs = sorted({b for b, _ in done if (f"beta={b},rho={r_}" in grid)})
            if len(xs) > 1:
                ax.plot(xs, [grid[f"beta={b},rho={r_}"] for b in xs], "o-", label=f"rho={r_}")
        ax.axhline(grid.get("beta=0.0,rho=0.0", np.nan), color="k", ls="--", label="recogniser only")
        ax.set_xscale("symlog", linthresh=0.1), ax.set_xlabel("beta (LM weight)"), ax.set_ylabel("val_gate CER"), ax.legend(fontsize=8)
        ax.set_title("LLM N-best rescoring: tuning on val_gate lines", fontsize=9)
        self._fig("09_language_model/llm_rescoring_tuning.png", "LLM N-best rescoring - tuning curve (val_gate lines only)", fig)
        with open(os.path.join(self.dirs["09_language_model"], "llm_rescoring.json"), "w") as f:
            json.dump(json_safe(out), f, indent=2)
        return out

    # ------------------------------------------------------------------ gallery picks
    def _pick(self, R, n_total: int) -> List[Tuple[int, str]]:
        """Gallery lines with their tag: the worst lines, the longest exactly-read lines, and random lines."""
        ling = np.array([is_linguistic(g) for g in R["gt"]])
        order = np.array([j for j in np.argsort(R["cer_ps"], kind="stable") if ling[j]], dtype=int)
        k = max(1, n_total // 3)
        worst = [int(R["idx"][j]) for j in order[::-1][:k] if R["cer_ps"][j] > 0]
        longest_ok = [j for j in order if R["cer_ps"][j] == 0]
        best = [int(R["idx"][j]) for j in sorted(longest_ok, key=lambda j: -R["reflen"][j])[:k]]
        taken = set(worst) | set(best)
        ok_idx = {int(i) for i, g in zip(R["idx"], ling) if g}
        rest = [int(i) for i in np.random.RandomState(0).permutation(R["idx"]) if int(i) not in taken and int(i) in ok_idx]
        picked = [(i, "worst") for i in worst] + [(i, "best") for i in best] + [(i, "random") for i in rest]
        return picked[:n_total]

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        cfg, ic = self.cfg, self.ic
        if self.model is None:
            self.load()
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)
        if ic.split != "test":
            LOG.warning("evaluating on '%s': the checkpoint was SELECTED on val_sel, so this number is optimistic and must not be reported as the result.", ic.split)
        ds = {"test": self.data.test_ds, "val": self.data.val_ds}[ic.split]
        look = log_test_look(cfg, self.ckpt_info, ic.split)
        idx = None if ic.max_eval_samples is None else self._random_subset(range(len(ds)), ic.max_eval_samples, 9)
        t0 = time.time()
        all_idx = list(range(len(ds))) if idx is None else idx
        sub_ids = self._random_subset(all_idx, ALIGN_CAP, 4)
        R = self._collect(ds, idx, extras=True, cons=True, keep_lp=sub_ids)
        n = len(R["gt"])
        same_split = ds is self.data.val_ds
        if same_split and idx is None:
            Rval = R
        else:
            val_sel = self._random_subset([int(i) for i in self.data.val_gate_idx], ic.selective_val_lines, 6)
            Rval = self._collect(self.data.val_ds, val_sel, extras=False, cons=True, keep_lp=self._random_subset(val_sel, ALIGN_CAP, 7))
        self._align_cache(R, sub_ids)
        if Rval is not R:
            self._align_cache(Rval, list(Rval["lp"].keys()))
        blk = max(1, int(ic.bootstrap_block))
        g_ci = np.asarray(R["idx"]) // blk
        cer_ci = bootstrap_cer_ci(R["ed"], R["reflen"], ic.bootstrap, 0, groups=g_ci)
        wer_ci = bootstrap_cer_ci(R["ed_w"], R["reflen_w"], ic.bootstrap, 0, groups=g_ci)
        metrics: Dict[str, Any] = {
            "checkpoint": self.ckpt_info, "split": ic.split, "n_lines": n, "corpora": ds.domain_names,
            "SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION": bool(self.data.is_synthetic),
            "CER": float(R["ed"].sum() / max(R["reflen"].sum(), 1)), "CER_ci95": list(cer_ci),
            "WER": float(R["ed_w"].sum() / max(R["reflen_w"].sum(), 1)), "WER_ci95": list(wer_ci),
            "mean_line_CER": float(R["cer_ps"].mean()), "exact_line_accuracy": float((R["cer_ps"] == 0).mean()),
            "labeled_fraction": cfg.data.labeled_fraction, "calibration_fraction": cfg.data.calib_fraction,
            "model_selection": "val_sel CER (EMA or student); all post-hoc tuning on val_gate (see protocol/protocol.json)",
            "evaluation_looks_so_far": look, "val_sel_lines": int(len(self.data.val_sel_idx)), "val_gate_lines": int(len(self.data.val_gate_idx)),
            "provenance": self.data.provenance}
        LOG.info("[%s] CER %.4f [%.4f, %.4f] | WER %.4f [%.4f, %.4f] | exact lines %.3f (%d lines)", ic.split, metrics["CER"], cer_ci[0], cer_ci[1], metrics["WER"],
                 wer_ci[0], wer_ci[1], metrics["exact_line_accuracy"], n)
        metrics.update(self._recognition_block(R, ds))
        cal = self._calibration_block(R, Rval, same_split)
        if cal:
            metrics["calibration"] = cal
        metrics["uncertainty_marking"] = self._calibrate_marking(Rval, same_split)
        metrics["error_structure"] = self._error_structure_block(R)
        metrics.update(self._codebook_block(R))
        metrics.update(self._style_block(R))
        LOG.info("global analyses done (%.0fs)", time.time() - t0)
        post = self._posthoc_block(ds, R)
        if post:
            metrics["posthoc"] = post
            metrics["CER_final_with_validated_posthoc"] = post["test_CER_final"]
            with open(os.path.join(self.dirs["04_posthoc"], "posthoc.json"), "w") as f:
                json.dump(json_safe(post), f, indent=2)
            R["ed_final"] = np.asarray(self.run_extra.get("final_edits", R["ed"]))
        if ic.robustness_lines > 0:
            metrics["robustness"] = self._robustness(ds)
        if ic.pl_diag_samples > 0:
            metrics["pseudo_label_reliability"] = self._pl_reliability()
        resc = self._llm_rescoring_eval(ds)
        if resc:
            metrics["llm_rescoring"] = resc
            adopted_ = bool(resc.get("adopted_on_validation"))
            metrics["CER_system_final"] = float(resc["CER_rescored"] if adopted_ else resc["CER_beam_top1"])
            metrics["CER_system_final_note"] = ("beam top-1 + validated options: " + ("LLM N-best rescoring ADOPTED on val_gate" if adopted_ else
                                                "LLM rescoring NOT adopted (no significant val_gate gain)") + "; CER above is the greedy reading without it")
        metrics["interpretability"] = self.run_interpretability(ds, R)
        curves = os.path.join(cfg.curve_dir, "curves_all.png")
        if os.path.exists(curves):
            shutil.copy(curves, os.path.join(self.dirs["00_summary"], "training_curves.png"))
            self.fig_list.insert(0, ("00_summary/training_curves.png", "Training curves"))

        # ---- per-sample explanations + translation ----------------------------------------------------------------------------------
        vis = self._pick(R, ic.n_vis)
        ctxs = [self._explain(ds, i, full=True, tag=tag) for i, tag in tqdm(vis, desc="explaining samples", leave=False)]
        nl = [c["attn_nonlocal"] for c in ctxs if "attn_nonlocal" in c]
        if nl:
            metrics["attention_rollout_nonlocal_share"] = float(np.mean(nl))
        sk = [c["attn_sink"] for c in ctxs if np.isfinite(c.get("attn_sink", float("nan")))]
        if sk:
            metrics["attention_sink_mass"] = float(np.mean(sk))
        taken = {c["i"] for c in ctxs}
        need = max(ic.n_translate - len(ctxs), 0)
        gt_of = {int(i): g for i, g in zip(R["idx"], R["gt"])}
        extra = [int(j) for j in np.random.RandomState(2).permutation(R["idx"]) if int(j) not in taken and is_linguistic(gt_of[int(j)])][:need]
        tr_ctxs = ctxs[:ic.n_translate] + [self._explain(ds, j, full=False, tag="random") for j in extra]
        rk_ = R.get("risk_rel")
        self._risk_by_idx = ({int(i): float(v) for i, v in zip(R["idx"], rk_)} if rk_ is not None else {int(i): float(1.0 - c) for i, c in zip(R["idx"], R["conf"])})
        tr = self._translate_block(ds, tr_ctxs)
        for k, ctx in enumerate(ctxs):
            self._sample_figs(ctx, f"sample_{k:02d}_{ctx['tag']}_line{ctx['i']:05d}")
        if tr:
            metrics["translation"] = {k: v for k, v in tr.items() if k != "rows"}
            with open(os.path.join(self.dirs["11_translation"], "translations.json"), "w", encoding="utf-8") as f:
                json.dump(json_safe(tr), f, indent=2, ensure_ascii=False, allow_nan=False)

        # ---- files -----------------------------------------------------------------------------------------------------------------
        with open(os.path.join(self.dirs["01_recognition"], "predictions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            pp = R.get("pred_posthoc")
            w.writerow(["line_index", "corpus", "ground_truth", "prediction", "CER", "confidence", "view_disagreement", "edits", "ref_len"]
                       + (["prediction_posthoc", "CER_posthoc"] if pp else []) + [f"measured_{s}" for s in STYLE_NAMES] + [f"pred_{s}" for s in STYLE_NAMES])
            st_, sp_ = R["style_raw"], self.data.denorm_style(R["style_pred"])
            for k in range(n):
                i = int(R["idx"][k])
                ex = ([ds.display(i, pp[i]), f"{cer(R['gt'][k], pp[i]):.4f}"] if pp else [])
                w.writerow([i, ds.domain_names[int(ds.domain[i])], ds.display(i), ds.display(i, R["pred"][k]), f"{R['cer_ps'][k]:.4f}", f"{R['conf'][k]:.4f}",
                            "" if not np.isfinite(R["cons"][k]) else f"{R['cons'][k]:.4f}", int(R["ed"][k]), int(R["reflen"][k])] + ex
                           + ["" if not np.isfinite(v) else f"{v:.4f}" for v in st_[k]] + [f"{v:.4f}" for v in sp_[k]])
        with open(os.path.join(self.dirs["00_summary"], "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(metrics), f, indent=2, allow_nan=False, ensure_ascii=False)
        with open(os.path.join(self.dirs["00_summary"], "config_used.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        if self.data.is_synthetic:
            with open(os.path.join(self.out, "SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION.txt"), "w") as f:
                f.write("These outputs come from generated fake handwriting; the numbers have no scientific meaning.\n")
        if ic.html_report:
            self._html(metrics)
        if ic.report:
            try:
                from .report import build_run_report
                self._write_report_extras(R, ds)
                build_run_report(self.out, cfg.run_dir, curve_dir=cfg.curve_dir)
            except Exception:
                LOG.exception("the run report could not be built (the metrics and figures above are complete)")
        LOG.info("Inference outputs written to %s (%d figures)", self.out, len(self.fig_list) + len(ctxs) * len(SAMPLE_VIZ))
        return metrics

    # ------------------------------------------------------------------ report extras (what only the model can supply)
    @torch.no_grad()
    def _evidence_pass(self, ds, ids: Sequence[int], view: str = "shear +0.08"):
        self.model.eval()
        for batch in self.data.eval_loader(ds, [int(i) for i in ids]):
            batch = to_device(batch, self.device)
            x, w = batch["image"], batch["width"]
            o = self.model(x, w)
            o2 = self.model(apply_perturbation(PERTURBATIONS[view], x, w), w)
            lp, aux = o["log_probs"].float().cpu().numpy(), o["aux_log_probs"].float().cpu().numpy()
            lp2, lens = o2["log_probs"].float().cpu().numpy(), o["out_lens"].cpu().tolist()
            for b in range(lp.shape[0]):
                L = int(lens[b])
                ids_, st, en, Fm = char_evidence(lp[b, :L], lp2[b, :L], aux[b, :L])
                yield int(batch["index"][b]), L, ids_, st, en, Fm, lp[b, :L]

    def _write_report_extras(self, R: Dict[str, Any], ds) -> None:
        """00_summary/v11_extras.json (+ the PNGs of the reading traces).  The trust rule fitted on TRAIN calibration lines is applied, unchanged,
        to the evaluation lines: the report shows whether it transfers.  Ground truth is read here for REPORTING only."""
        codec, fm = self.data.codec, self.frag_model
        order = sorted(int(i) for i in R["idx"])
        pos = {int(i): k for k, i in enumerate(R["idx"])}
        ref_of = {int(i): g for i, g in zip(R["idx"], R["gt"])}
        pred_of = {int(i): p for i, p in zip(R["idx"], R["pred"])}
        cer_of = {int(i): float(c) for i, c in zip(R["idx"], R["cer_ps"])}
        ex = self.run_extra or {}
        X: Dict[str, Any] = {"version": "v11", "strata": [int(ds.stratum[i]) for i in order], "corpus": [ds.domain_names[int(ds.domain[i])] for i in order]}
        if ex.get("wsd_test_pred") is not None:
            tid = ex["test_ids"]
            wp, we, wc = dict(zip(tid, ex["wsd_test_pred"])), dict(zip(tid, ex["wsd_test_edits"].tolist())), dict(zip(tid, ex["wsd_test_cluster"]))
            X["wsd_pred"], X["wsd_edits"], X["wsd_cluster"] = [ds.display(i, wp[i]) for i in order], [int(we[i]) for i in order], [int(wc[i]) for i in order]
        theta = (self.frag_state or {}).get("theta")
        if fm is not None and theta is not None:
            rows_X, rows_y, rows_ln, rows_st, line_risk = [], [], [], [], {}
            for i, L, ids_, st, en, Fm, _lp in tqdm(self._evidence_pass(ds, order), total=len(order), desc="trust audit", leave=False):
                if not len(ids_):
                    line_risk[i] = 1.0
                    continue
                rk = fm.risk(Fm)
                line_risk[i] = float(rk.mean())
                rows_X.append(np.c_[rk, Fm[:, 0]]), rows_y.append(char_wrong_labels(ref_of[i], codec.decode_ids(ids_))), rows_ln.append(np.full(len(ids_), i))
                rows_st.append(np.full(len(ids_), int(ds.stratum[i])))
            X["risk"] = [line_risk.get(i, 1.0) for i in order]
            if rows_X:
                RX, y, stt = np.concatenate(rows_X), np.concatenate(rows_y).astype(bool), np.concatenate(rows_st)
                th = np.array([cell_theta(theta, c) for c in stt])
                tr = RX[:, 0] <= th
                lines_in = np.concatenate(rows_ln)
                sres = []
                for s_ in sorted(set(stt.tolist())):
                    m_ = stt == s_
                    sres.append({"stratum": int(s_), "lines": int(len(set(lines_in[m_].tolist()))), "chars": int(m_.sum()), "trusted_share": float(tr[m_].mean()),
                                 "err_trusted": float(y[m_ & tr].mean()) if (m_ & tr).any() else None, "err_doubtful": float(y[m_ & ~tr].mean()) if (m_ & ~tr).any() else None})
                X["frag_audit"] = {"target": float((self.frag_state or {}).get("alpha", float("nan"))), "rule_epoch": (self.frag_state or {}).get("epoch"), "view": "shear +0.08",
                                   "chars": int(len(y)), "trusted_share": float(tr.mean()), "err_trusted": float(y[tr].mean()) if tr.any() else None,
                                   "err_doubtful": float(y[~tr].mean()) if (~tr).any() else None, "err_all": float(y.mean()), "auroc_risk": auroc(RX[:, 0], y),
                                   "auroc_conf": auroc(-RX[:, 1], y), "strata": sres}
                with open(os.path.join(self.dirs["08_self_distillation"], "frag_trust_audit.json"), "w") as f:
                    json.dump(json_safe(X["frag_audit"]), f, indent=2)
        nb, rs_ = getattr(self, "_nb_eval", None), getattr(self, "_rescore", None)
        if nb and rs_:
            beta, rho = float(rs_["beta"]), float(rs_["rho"])
            ch_rows = []
            for i, v in nb.items():
                fused = fuse_scores(v["ctc"], v["lm"], v["ntok"], beta, rho)
                k = int(np.argmax(fused))
                eds = [edit_distance(v["ref"], t) for t in v["texts"]]
                ch_rows.append((eds[0] - eds[k], int(i), k, eds, fused, v))
            changed = [r for r in ch_rows if r[2] != 0]
            helped = sorted([r for r in changed if r[0] > 0], key=lambda r: -r[0])[:3]
            hurt = sorted([r for r in changed if r[0] < 0], key=lambda r: r[0])[:2]
            ex_list = []
            for tag, r in [("rescoring helped", r) for r in helped] + [("rescoring hurt", r) for r in hurt]:
                gain, i, k, eds, fused, v = r
                keep = sorted(set(range(min(6, len(v["texts"])))) | {k})
                ex_list.append({"line": i, "tag": tag, "gt": ds.display(i), "chosen": keep.index(k), "gain_edits": int(gain),
                                "items": [{"rank": int(j), "text": ds.display(i, v["texts"][j]), "ctc": float(v["ctc"][j]), "lm": float(v["lm"][j]), "fused": float(fused[j]),
                                           "ed": int(eds[j])} for j in keep]})
            X["nbest"] = {"beta": beta, "rho": rho, "adopted": bool(rs_["adopted"]), "llm": self.cfg.llm.model_name, "lines": len(ch_rows),
                          "changed_share": float(len(changed) / max(len(ch_rows), 1)), "helped_lines": int(sum(1 for r in changed if r[0] > 0)),
                          "hurt_lines": int(sum(1 for r in changed if r[0] < 0)), "net_edits_saved": int(sum(r[0] for r in changed)), "examples": ex_list}
        picks = list(self._pick(R, 9))
        if ex.get("wsd_test_edits") is not None:
            d = ex["edits_greedy"] - ex["wsd_test_edits"]
            tid = ex["test_ids"]
            srt = np.argsort(-d, kind="stable")
            picks += [(int(tid[j]), "WSD helped most") for j in srt[:3] if d[j] > 0] + [(int(tid[j]), "WSD hurt most") for j in srt[::-1][:2] if d[j] < 0]
        seen, plan = set(), []
        for i, tag in picks:
            if i not in seen and i in pos:
                seen.add(i), plan.append((i, tag))
        tdir = os.path.join(self.dirs["12_report"], "08_reading_traces")
        os.makedirs(tdir, exist_ok=True)
        traces = []
        from PIL import Image
        wsd_pred = dict(zip(order, X["wsd_pred"])) if X.get("wsd_pred") else {}
        for i, L, ids_, st, en, Fm, lp in self._evidence_pass(ds, [p[0] for p in plan]):
            tag = next(t for j, t in plan if j == i)
            ink = ds.get_ink(i)
            Image.fromarray((255.0 * (1.0 - np.clip(ink, 0, 1))).astype(np.uint8)).save(os.path.join(tdir, f"line_{i:05d}.png"))
            hyp = codec.decode_ids(ids_)
            wrong = char_wrong_labels(ref_of[i], hyp) if len(ids_) else np.zeros(0)
            rk = fm.risk(Fm) if (fm is not None and len(ids_)) else (1.0 - 1.0 / (1.0 + np.exp(-Fm[:, 0])) if len(ids_) else np.zeros(0))
            th_ = cell_theta(theta, int(ds.stratum[i])) if (theta is not None and fm is not None) else 0.5
            chars = []
            for k in range(len(ids_)):
                fpk = int(st[k] + np.exp(lp[st[k]:en[k]]).max(1).argmax())
                p = np.exp(lp[fpk])
                p2 = p.copy()
                p2[0], p2[ids_[k]] = 0.0, 0.0
                a = int(p2.argmax())
                chars.append({"ch": codec.i2c.get(ids_[k], "?"), "x0": float(st[k] / max(L, 1)), "x1": float(en[k] / max(L, 1)), "risk": float(rk[k]), "p": float(p[ids_[k]]),
                              "alt": codec.i2c.get(a, None) if p2[a] > 1e-3 else None, "palt": float(p2[a]), "trusted": bool(rk[k] <= th_), "wrong": bool(wrong[k])})
            traces.append({"line": i, "tag": tag, "cer": cer_of[i], "gt": ds.display(i), "pr": ds.display(i, pred_of[i]), "pw": wsd_pred.get(i), "rtl": bool(ds.rtl[i]),
                           "image": f"08_reading_traces/line_{i:05d}.png", "chars": chars})
        X["traces"] = traces
        with open(os.path.join(self.dirs["00_summary"], "v11_extras.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(X), f, ensure_ascii=False, allow_nan=False)

    def _html(self, metrics: Dict[str, Any]) -> None:
        def flat(d, p=""):
            for k, v in d.items():
                if k in ("val_cer_grid", "provenance", "rows", "top_confusions", "interpretability"):
                    continue
                if isinstance(v, dict):
                    yield from flat(v, f"{p}{k}.")
                elif isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v) and len(v) <= 4:
                    yield f"{p}{k}", ", ".join(f"{x:.4f}" if isinstance(x, float) else str(x) for x in v)
                elif isinstance(v, bool):
                    yield f"{p}{k}", str(v)
                elif isinstance(v, (int, float, str)):
                    yield f"{p}{k}", v
        fm = lambda v: f"{v:.4f}" if isinstance(v, float) else html.escape(str(v))   # noqa: E731
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{fm(v)}</td></tr>" for k, v in flat(metrics))
        figs = "".join(f"<h3>{html.escape(c)}</h3><img src='../{p}' style='max-width:100%'>" for p, c in self.fig_list)
        banner = ("<div style='background:#b00020;color:#fff;padding:10px;font-weight:bold'>SYNTHETIC SMOKE TEST - fake handwriting - NOT FOR PUBLICATION</div>"
                  if self.data.is_synthetic else "")
        with open(os.path.join(self.dirs["00_summary"], "report.html"), "w", encoding="utf-8") as f:
            f.write("<html><head><meta charset='utf-8'><title>InkLingua metrics</title></head><body style='font-family:sans-serif;max-width:1300px;margin:auto'>"
                    f"{banner}<h1>InkLingua - every number of the evaluation</h1><p>The interactive report is in <a href='../12_report/index.html'>12_report/index.html</a>; "
                    f"the folders 01-11 hold the numbers behind every figure.</p><table border=1 cellpadding=3>{rows}</table>{figs}</body></html>")
