"""
Post-hoc inference options, all label-free at test time, each ADOPTED only if the Bonferroni-adjusted 95 % form-clustered paired-bootstrap
interval of its gain on the `val_gate` lines excludes zero (the test lines only report):
  (0) [VIEW-ENS] posterior averaging over mild re-renderings of the same line,
  (1) [STYLE-ADABN] batch-norm statistics per handwriting-style cluster (transductive),
  (1b) [WSD] writer self-distillation: a few label-free fine-tuning steps per style cluster, guided by the FRAG-RC trust masks (transductive),
  (2) a one-parameter word-gap (space) bonus in the greedy decoder;
plus the structure of the errors (character classes, forms, style bins).
"""
from __future__ import annotations

import copy
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from .data import Collator
from .decoding import decode_with_space_bonus
from .explain import adapt_bn_state, bn_layers, load_bn, snapshot_bn
from .frag import frag_trust_batch
from .model import fragment_distillation
from .pseudo import PERTURBATIONS, apply_perturbation
from .style import STYLE_NAMES, assign_nearest, kmeans_np
from .text import CHAR_CLASSES, align_chars, char_class, edit_distance, paired_bootstrap_delta
from .utils import LOG, to_device


class PosthocMixin:
    # ------------------------------------------------------------------ frame posteriors with optional per-group BN statistics
    @torch.inference_mode()
    def _lp_pass(self, ds, indices: Sequence[int], assign: Optional[Dict[int, int]] = None,
                 states: Optional[Dict[int, Dict[str, Tuple[np.ndarray, np.ndarray]]]] = None, view: Optional[str] = None) -> Dict[int, np.ndarray]:
        """Frame log-posteriors (valid frames only) of the given lines; with `assign` / `states` every line is read by the model that carries
        the batch-norm statistics of its group; the source statistics are always restored."""
        out: Dict[int, np.ndarray] = {}
        groups: Dict[int, List[int]] = {-1: [int(i) for i in indices]}
        if assign is not None:
            groups = defaultdict(list)
            for i in indices:
                groups[assign[int(i)]].append(int(i))
        src = snapshot_bn(self.model)
        try:
            for g, idxs in groups.items():
                if assign is not None and states is not None:
                    load_bn(self.model, states.get(g, src))
                for batch in self.data.eval_loader(ds, idxs):
                    batch = to_device(batch, self.device)
                    x = batch["image"] if view is None else apply_perturbation(PERTURBATIONS[view], batch["image"], batch["width"])
                    o = self.model(x, batch["width"])
                    lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
                    for b in range(lp.shape[0]):
                        out[int(batch["index"][b])] = lp[b, :lens[b]].copy()
        finally:
            load_bn(self.model, src)
        return out

    def _lp_views(self, indices: Sequence[int], ds, views: Sequence[str], cache: Optional[Dict[str, Dict[int, np.ndarray]]] = None) -> Dict[int, np.ndarray]:
        """[VIEW-ENS] frame log-posteriors of the clean line averaged (in PROBABILITY space) with those of re-renderings of the same pixels."""
        def get(v: Optional[str]) -> Dict[int, np.ndarray]:
            if cache is None:
                return self._lp_pass(ds, indices, view=v)
            k = "clean" if v is None else v
            if k not in cache:
                cache[k] = self._lp_pass(ds, indices, view=v)
            return cache[k]
        acc = {i: lp.copy() for i, lp in get(None).items()}
        if not views:
            return acc
        for v in views:
            for i, lp in get(v).items():
                acc[i] = np.logaddexp(acc[i], lp)
        n = float(len(views) + 1)
        return {i: (lp - np.log(n)).astype(np.float32) for i, lp in acc.items()}

    def _tta_assign(self, ds, ids: Sequence[int], k: int) -> Dict[int, int]:
        """Group lines by handwriting style: k-means on the MEASURED descriptors (label-free)."""
        X = ds.style[np.asarray(list(ids), np.int64)].astype(np.float64)
        if len(X) < 2 * max(k, 1):
            return {int(i): 0 for i in ids}
        a = assign_nearest(X, kmeans_np(X, k, seed=0))
        return {int(i): int(c) for i, c in zip(ids, a)}

    @torch.no_grad()
    def _bn_collect(self, ds, assign: Dict[int, int]) -> Dict[int, Tuple[Dict[str, Dict[str, Any]], int]]:
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, g in assign.items():
            groups[g].append(i)
        res: Dict[int, Tuple[Dict[str, Dict[str, Any]], int]] = {}
        layers = bn_layers(self.model)
        for g, idxs in groups.items():
            for _, m in layers:
                m.collect = {}
            try:
                for batch in self.data.eval_loader(ds, idxs):
                    batch = to_device(batch, self.device)
                    self.model(batch["image"], batch["width"])
                res[g] = ({n: m.collect for n, m in layers}, len(idxs))
            finally:
                for _, m in layers:
                    m.collect = None
        return res

    def _edits(self, lps: Dict[int, np.ndarray], ids: Sequence[int], refs: Dict[int, str], space_id: Optional[int], delta: float
               ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        codec = self.data.codec
        ed, rl, preds = [], [], []
        for i in ids:
            t = codec.decode_ids(decode_with_space_bonus(lps[int(i)].astype(np.float32), space_id, delta))
            preds.append(t)
            ed.append(edit_distance(refs[int(i)], t)), rl.append(len(refs[int(i)]))
        return np.array(ed, np.float64), np.maximum(np.array(rl, np.float64), 1.0), preds

    def _posthoc_block(self, ds, R: Dict[str, Any]) -> Dict[str, Any]:
        """Order: view ensemble, style-cluster AdaBN (not combined with the ensemble), writer self-distillation (replaces the previous readings
        if adopted), then the space bonus on top of whatever was adopted.  The final readings go to R["pred_posthoc"]."""
        ic, dm, codec = self.ic, self.data, self.data.codec
        if not ic.posthoc:
            return {}
        val_ids = [int(i) for i in dm.val_gate_idx]
        if len(val_ids) > ic.selective_val_lines:
            val_ids = self._random_subset(val_ids, ic.selective_val_lines, 6)
        test_ids = [int(i) for i in R["idx"]]
        blk = max(1, int(ic.bootstrap_block))
        gv, gt = np.asarray(val_ids) // blk, np.asarray(test_ids) // blk
        refs_v = {i: dm.val_ds.oracle[i] for i in val_ids}
        refs_t = {i: ds.oracle[i] for i in test_ids}
        space_id = codec.c2i.get(" ")
        res: Dict[str, Any] = {"validation_lines": len(val_ids), "test_lines": len(test_ids), "val_role": "val_gate",
                               "rule": "an option is adopted only if the Bonferroni-adjusted (over the grid searched) 95% form-clustered paired-bootstrap "
                                       "interval of its CER gain on the val_gate lines excludes zero; test lines only report"}
        lpv, lpt = self._lp_pass(dm.val_ds, val_ids), self._lp_pass(ds, test_ids)
        ev0, rv0, _ = self._edits(lpv, val_ids, refs_v, space_id, 0.0)
        et0, rt0, pt0 = self._edits(lpt, test_ids, refs_t, space_id, 0.0)
        res["val_CER_base"], res["test_CER_base"] = float(ev0.sum() / rv0.sum()), float(et0.sum() / rt0.sum())
        lpv_c, lpt_c, cur_pred = lpv, lpt, pt0
        ev_cur, et_cur = ev0, et0
        adopted: Dict[str, Any] = {"view_ensemble": [], "tta": False, "wsd": None, "space_bonus": 0.0}
        self._wsd_lp = {}
        self.run_extra = {"test_ids": test_ids, "edits_greedy": et0.copy(), "pred_greedy": list(pt0), "ref_len": rt0.copy()}

        # ---- (0) view ensemble ------------------------------------------------------------------------------------------------
        self._ens_views = ()
        if ic.view_ensemble and ic.view_sets:
            cache_v: Dict[str, Dict[int, np.ndarray]] = {"clean": lpv}
            cands: Dict[str, Any] = {}
            best_e: Tuple[Any, ...] = (float("inf"), None, None, None)
            for vs in ic.view_sets:
                lp_e = self._lp_views(val_ids, dm.val_ds, vs, cache_v)
                e_e, r_e, _ = self._edits(lp_e, val_ids, refs_v, space_id, 0.0)
                cands["+".join(vs)] = float(e_e.sum() / r_e.sum())
                if cands["+".join(vs)] < best_e[0]:
                    best_e = (cands["+".join(vs)], vs, e_e, lp_e)
            cache_v.clear()
            gate_e = paired_bootstrap_delta(best_e[2], ev0, rv0, ic.bootstrap, 7, groups=gv, n_tests=len(ic.view_sets))
            use_e = bool(gate_e["hi_adj"] < 0)
            lp_te = self._lp_views(test_ids, ds, best_e[1], {"clean": lpt})
            e_te, r_te, p_te = self._edits(lp_te, test_ids, refs_t, space_id, 0.0)
            res["view_ensemble"] = {"val_CER_by_view_set": cands, "chosen": list(best_e[1]), "val_delta_vs_base": gate_e, "adopted": use_e,
                                    "test_CER": float(e_te.sum() / r_te.sum()), "test_delta_vs_base": paired_bootstrap_delta(e_te, et0, rt0, ic.bootstrap, 8, groups=gt),
                                    "note": "clean line + re-renderings of the same pixels; frame posteriors averaged in probability space; label-free"}
            LOG.info("[VIEW-ENS] views %s: val CER %.4f -> %.4f (adjusted hi %.4f) -> %s | test CER %.4f -> %.4f", "+".join(best_e[1]), res["val_CER_base"],
                     best_e[0], gate_e["hi_adj"], "ADOPTED" if use_e else "not adopted", res["test_CER_base"], res["view_ensemble"]["test_CER"])
            if use_e:
                self._ens_views = tuple(best_e[1])
                adopted["view_ensemble"] = list(best_e[1])
                lpv_c, lpt_c, cur_pred, ev_cur, et_cur = best_e[3], lp_te, p_te, best_e[2], e_te
            del lp_te

        # ---- (1) style-cluster AdaBN ---------------------------------------------------------------------------------------------
        if ic.tta and bn_layers(self.model) and not self._ens_views:
            src = snapshot_bn(self.model)
            av, at = self._tta_assign(dm.val_ds, val_ids, ic.tta_clusters), self._tta_assign(ds, test_ids, ic.tta_clusters)
            accv, acct = self._bn_collect(dm.val_ds, av), self._bn_collect(ds, at)
            grid: Dict[str, Any] = {}
            best: Tuple[Any, ...] = (float("inf"), None, None, None)
            for tau in ic.tta_shrink:
                states = {g: adapt_bn_state(src, a, n, tau) if n >= ic.tta_min_lines else src for g, (a, n) in accv.items()}
                lp_a = self._lp_pass(dm.val_ds, val_ids, av, states)
                e_a, r_a, _ = self._edits(lp_a, val_ids, refs_v, space_id, 0.0)
                grid[str(tau)] = float(e_a.sum() / r_a.sum())
                if grid[str(tau)] < best[0]:
                    best = (grid[str(tau)], tau, e_a, lp_a)
            tau_b, e_b, lp_b = best[1], best[2], best[3]
            gate = paired_bootstrap_delta(e_b, ev0, rv0, ic.bootstrap, 1, groups=gv, n_tests=len(ic.tta_shrink))
            states_t = {g: adapt_bn_state(src, a, n, tau_b) if n >= ic.tta_min_lines else src for g, (a, n) in acct.items()}
            lp_t = self._lp_pass(ds, test_ids, at, states_t)
            e_t, r_t, p_t = self._edits(lp_t, test_ids, refs_t, space_id, 0.0)
            use = bool(gate["hi_adj"] < 0)
            res["tta"] = {"clusters": ic.tta_clusters, "val_CER_by_shrinkage_tau": grid, "chosen_tau": tau_b, "val_delta_vs_base": gate, "adopted": use,
                          "test_CER": float(e_t.sum() / r_t.sum()), "test_delta_vs_base": paired_bootstrap_delta(e_t, et0, rt0, ic.bootstrap, 2, groups=gt),
                          "test_lines_per_cluster": {str(g): n for g, (_, n) in acct.items()},
                          "note": "transductive and label-free: batch-norm statistics of each style cluster from the unlabeled test lines (AdaBN, shrunk)"}
            LOG.info("TTA (style-cluster AdaBN, tau=%s): val CER %.4f -> %.4f -> %s | test CER %.4f -> %.4f", tau_b, res["val_CER_base"], best[0],
                     "ADOPTED" if use else "not adopted", res["test_CER_base"], res["tta"]["test_CER"])
            if use:
                adopted["tta"] = True
                lpv_c, lpt_c, cur_pred, ev_cur, et_cur = lp_b, lp_t, p_t, e_b, e_t

        # ---- (1b) writer self-distillation ---------------------------------------------------------------------------------------
        if ic.wsd and getattr(self, "frag_model", None) is not None and getattr(self, "frag_state", None) is not None and ic.wsd_steps:
            k_t = max(2, int(ic.wsd_clusters))
            k_v = int(max(2, min(k_t, round(len(val_ids) / max(len(test_ids) / k_t, 1.0)))))
            av, at = self._tta_assign(dm.val_ds, val_ids, k_v), self._tta_assign(ds, test_ids, k_t)
            t_wsd = time.time()
            reads_v, info_v = self._wsd_reads(dm.val_ds, val_ids, av, ic.wsd_steps, lpv, seed=11)
            reads_t, info_t = self._wsd_reads(ds, test_ids, at, ic.wsd_steps, lpt, seed=13)
            cand, best_w = {}, (float("inf"), None, None)
            for s_ in sorted(int(s) for s in ic.wsd_steps):
                e_w, r_w, _ = self._edits(reads_v[s_], val_ids, refs_v, space_id, 0.0)
                cand[str(s_)] = float(e_w.sum() / r_w.sum())
                if cand[str(s_)] < best_w[0]:
                    best_w = (cand[str(s_)], s_, e_w)
            s_b = int(best_w[1])
            gate_w = paired_bootstrap_delta(best_w[2], ev_cur, rv0, ic.bootstrap, 21, groups=gv, n_tests=len(cand))
            use_w = bool(gate_w["hi_adj"] < 0)
            e_tw, r_tw, p_tw = self._edits(reads_t[s_b], test_ids, refs_t, space_id, 0.0)
            rep_w = paired_bootstrap_delta(e_tw, et_cur, rt0, ic.bootstrap, 22, groups=gt)

            def by_cluster(assign, ids, e_new, e_old, r_):
                out_ = {}
                for g_ in sorted(set(assign.values())):
                    m_ = np.array([assign[int(i)] == g_ for i in ids])
                    out_[str(g_)] = {"lines": int(m_.sum()), "CER_before": float(e_old[m_].sum() / max(r_[m_].sum(), 1)),
                                     "CER_after": float(e_new[m_].sum() / max(r_[m_].sum(), 1))}
                return out_

            res["wsd"] = {"val_CER_by_steps": cand, "chosen_steps": s_b, "val_delta_vs_before": gate_w, "adopted": use_w, "test_CER": float(e_tw.sum() / r_tw.sum()),
                          "test_delta_vs_before": rep_w, "clusters": {"val_gate": k_v, "test": k_t},
                          "val_by_cluster": by_cluster(av, val_ids, best_w[2], ev_cur, rv0), "test_by_cluster": by_cluster(at, test_ids, e_tw, et_cur, rt0),
                          "val_group_info": {str(g): v for g, v in info_v.items()}, "test_group_info": {str(g): v for g, v in info_t.items()},
                          "seconds": float(time.time() - t_wsd),
                          "note": "transductive and label-free: each cluster's copy of the model moves for a few steps on the unlabeled images of that cluster "
                                  "only, guided by the FRAG-RC trust masks; val_gate lines are adapted the same way before the gain is measured"}
            LOG.info("[WSD] steps %s: val_gate CER %.4f -> %.4f (adjusted hi %.4f) -> %s | test CER %.4f -> %.4f (%.0fs)", s_b, float(ev_cur.sum() / rv0.sum()),
                     best_w[0], gate_w["hi_adj"], "ADOPTED" if use_w else "not adopted", float(et_cur.sum() / rt0.sum()), res["wsd"]["test_CER"], res["wsd"]["seconds"])
            self.run_extra.update({"wsd_test_pred": list(p_tw), "wsd_test_edits": e_tw.copy(), "wsd_test_cluster": [int(at[int(i)]) for i in test_ids]})
            if use_w:
                adopted["wsd"] = s_b
                self._ens_views = ()
                self._wsd_lp = {id(dm.val_ds): reads_v[s_b], id(ds): reads_t[s_b]}
                lpv_c, lpt_c, cur_pred, ev_cur, et_cur = reads_v[s_b], reads_t[s_b], p_tw, best_w[2], e_tw
            del reads_v, reads_t
        elif ic.wsd:
            res["wsd"] = {"adopted": False, "skipped": "no FRAG-RC rule in the checkpoint (train with cfg.train.frag.enabled=True and calibration lines)"}

        # ---- (2) space bonus ------------------------------------------------------------------------------------------------------
        if space_id is not None and ic.space_bonus_grid:
            grid2: Dict[str, float] = {}
            best2 = (float("inf"), 0.0, ev_cur)
            for dlt in ic.space_bonus_grid:
                e_d, r_d, _ = self._edits(lpv_c, val_ids, refs_v, space_id, float(dlt))
                grid2[str(dlt)] = float(e_d.sum() / r_d.sum())
                if grid2[str(dlt)] < best2[0]:
                    best2 = (grid2[str(dlt)], float(dlt), e_d)
            gate2 = paired_bootstrap_delta(best2[2], ev_cur, rv0, ic.bootstrap, 3, groups=gv, n_tests=len(ic.space_bonus_grid))
            e_s, r_s, p_s = self._edits(lpt_c, test_ids, refs_t, space_id, best2[1])
            use2 = bool(best2[1] != 0.0 and gate2["hi_adj"] < 0)
            res["space_bonus"] = {"val_CER_by_bonus": grid2, "chosen": best2[1], "val_delta_vs_before": gate2, "adopted": use2,
                                  "test_CER": float(e_s.sum() / r_s.sum()), "test_delta_vs_before": paired_bootstrap_delta(e_s, et_cur, rt0, ic.bootstrap, 4, groups=gt),
                                  "test_space_count": {"reference": int(sum(refs_t[i].count(" ") for i in test_ids)), "before": int(sum(p.count(" ") for p in cur_pred)),
                                                       "with_bonus": int(sum(p.count(" ") for p in p_s))}}
            LOG.info("space bonus: val CER %.4f -> %.4f at +%.1f nats -> %s | test CER %.4f -> %.4f", float(ev_cur.sum() / rv0.sum()), best2[0], best2[1],
                     "ADOPTED" if use2 else "not adopted", float(et_cur.sum() / rt0.sum()), float(e_s.sum() / r_s.sum()))
            if use2:
                adopted["space_bonus"] = best2[1]
                cur_pred, et_cur = p_s, e_s
        res["adopted"] = adopted
        self.run_extra.update({"final_pred": list(cur_pred), "final_edits": et_cur.copy()})
        res["test_CER_final"] = float(et_cur.sum() / rt0.sum())
        res["test_delta_final_vs_base"] = paired_bootstrap_delta(et_cur, et0, rt0, ic.bootstrap, 5, groups=gt)
        R["pred_posthoc"] = {i: p for i, p in zip(test_ids, cur_pred)}
        return res

    # ------------------------------------------------------------------ [WSD]
    @torch.inference_mode()
    def _read_with(self, model: nn.Module, ds, lines: Sequence[int]) -> Dict[int, np.ndarray]:
        res: Dict[int, np.ndarray] = {}
        model.eval()
        for batch in self.data.eval_loader(ds, [int(i) for i in lines]):
            batch = to_device(batch, self.device)
            o = model(batch["image"], batch["width"])
            lp, lens = o["log_probs"].cpu().numpy(), o["out_lens"].cpu().tolist()
            for b in range(lp.shape[0]):
                res[int(batch["index"][b])] = lp[b, :lens[b]].copy()
        return res

    def _wsd_reads(self, ds, ids: Sequence[int], assign: Dict[int, int], steps_list: Sequence[int], base_lp: Dict[int, np.ndarray], seed: int = 0
                   ) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[str, float]]]:
        """For every style cluster a COPY of the model is fine-tuned for a few steps on the cluster's own UNLABELED images: a strong-view student
        is distilled from the frozen model's clean-view posteriors, the FRAG-RC trust masks decide which frames are sharpened.  Only the CTC head,
        the final LayerNorm, the output FiLM and the last encoder layer move.  Returns {steps: {line: log-posteriors}} and per-group diagnostics."""
        ic, dev, codec, fs = self.ic, self.device, self.data.codec, self.frag_state
        steps_list = sorted(int(s) for s in steps_list)
        out: Dict[int, Dict[int, np.ndarray]] = {s: dict(base_lp) for s in steps_list}
        info: Dict[int, Dict[str, float]] = {}
        groups: Dict[int, List[int]] = defaultdict(list)
        for i in ids:
            groups[assign[int(i)]].append(int(i))
        views = ("gauss noise", "blur", "thicker ink", "thinner ink", "shear +0.08", "shear -0.08", "low contrast")
        collate = Collator(codec)
        self.model.eval()
        for g, lines in sorted(groups.items()):
            if len(lines) < ic.wsd_min_lines:
                info[g] = {"n": len(lines), "adapted": 0.0}
                continue
            rng = np.random.RandomState(seed + 31 * int(g) + 7)
            stu = copy.deepcopy(self.model).eval()
            for p in stu.parameters():
                p.requires_grad_(False)
            params = []
            for m_ in [stu.ctc_head, stu.final_ln, stu.layers[-1]] + ([stu.film_out] if stu.film_out is not None else []):
                for p in m_.parameters():
                    p.requires_grad_(True)
                    params.append(p)
            opt = torch.optim.AdamW(params, lr=ic.wsd_lr, weight_decay=0.0)
            order, ptr, tf = list(rng.permutation(lines)), 0, []
            for step in range(1, steps_list[-1] + 1):
                if ptr + ic.wsd_batch > len(order):
                    order, ptr = list(rng.permutation(lines)), 0
                sel, ptr = order[ptr:ptr + ic.wsd_batch], ptr + ic.wsd_batch
                items = []
                for i in sel:
                    it = ds[int(i)]
                    it["text"], it["weight"], it["oracle"] = "", 0.0, ""          # the label never enters the batch
                    items.append(it)
                batch = to_device(collate(items), dev)
                x, w = batch["image"], batch["width"]
                v1, v2 = rng.choice(len(views), 2, replace=False)
                with torch.no_grad():
                    o_w = self.model(x, w)
                    o_2 = self.model(apply_perturbation(PERTURBATIONS[views[v2]], x, w), w)
                    B = x.size(0)
                    trust, _nc, _nt = frag_trust_batch(o_w["log_probs"].float().cpu().numpy(), o_w["aux_log_probs"].float().cpu().numpy(),
                                                       o_2["log_probs"].float().cpu().numpy(), o_w["out_lens"].cpu().numpy(), batch["stratum"].cpu().numpy(),
                                                       np.ones(B, bool), self.frag_model, fs["theta"], self.cfg.train.frag.guard_frames)
                o_s = stu(apply_perturbation(PERTURBATIONS[views[v1]], x, w), w)
                loss = fragment_distillation(o_s["log_probs"], o_w["log_probs"].float(), o_w["out_lens"], torch.from_numpy(trust).to(dev),
                                             torch.ones(B, dtype=torch.bool, device=dev), ic.wsd_tau, ic.wsd_anchor)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                tf.append(float(trust.sum() / max(float(sum(int(l) for l in o_w["out_lens"].cpu().numpy())), 1.0)))
                if step in steps_list:
                    for i, lp in self._read_with(stu, ds, lines).items():
                        out[step][i] = lp
            info[g] = {"n": len(lines), "adapted": 1.0, "trusted_frame_share": float(np.mean(tf)) if tf else float("nan")}
            del stu, opt
        return out, info

    # ------------------------------------------------------------------ structure of the errors
    def _error_structure_block(self, R: Dict[str, Any]) -> Dict[str, Any]:
        """Where do the errors come from?  Character classes (Unicode categories), form-level clustering and the measured style descriptors."""
        n = len(R["gt"])
        pos = self._random_subset(range(n), self.ic.error_analysis_lines, 5)
        ref_cnt, err_cnt, ins_cnt = Counter(), Counter(), Counter()
        sub_n = case_only = 0
        sp = {"deleted": 0, "inserted": 0, "substituted_away": 0}
        n_edits = 0
        for k in pos:
            gt, pr = R["gt"][k], R["pred"][k]
            for op, ri, hi in align_chars(gt, pr):
                if ri is not None:
                    ref_cnt[char_class(gt[ri])] += 1
                if op == "match":
                    continue
                n_edits += 1
                if op == "sub":
                    err_cnt[char_class(gt[ri])] += 1
                    sub_n += 1
                    case_only += int(gt[ri].lower() == pr[hi].lower() and gt[ri] != pr[hi])
                    sp["substituted_away"] += int(gt[ri] == " ")
                elif op == "del":
                    err_cnt[char_class(gt[ri])] += 1
                    sp["deleted"] += int(gt[ri] == " ")
                else:
                    ins_cnt[char_class(pr[hi])] += 1
                    sp["inserted"] += int(pr[hi] == " ")
        tot_ref = max(sum(ref_cnt.values()), 1)
        M: Dict[str, Any] = {"lines_analysed": len(pos),
                             "by_character_class": {c: {"share_of_reference_chars": ref_cnt[c] / tot_ref,
                                                        "error_rate_on_reference_chars": (err_cnt[c] / ref_cnt[c]) if ref_cnt[c] else float("nan"),
                                                        "inserted_chars": int(ins_cnt[c])} for c in CHAR_CLASSES},
                             "space_edits": {**sp, "share_of_all_edits": float(sum(sp.values()) / max(n_edits, 1))},
                             "case_only_share_of_substitutions": float(case_only / max(sub_n, 1))}
        order = np.argsort(np.asarray(R["idx"]))
        e_o, l_o = np.asarray(R["ed"], np.float64)[order], np.maximum(np.asarray(R["reflen"], np.float64)[order], 1.0)
        c_o = e_o / l_o
        blk = np.arange(len(c_o)) // 20
        bc = np.array([e_o[blk == b].sum() / l_o[blk == b].sum() for b in np.unique(blk)])
        blk30 = np.arange(len(c_o)) // 30
        ub = np.unique(blk30)
        grand = c_o.mean()
        ss_b = sum(((blk30 == b).sum()) * (c_o[blk30 == b].mean() - grand) ** 2 for b in ub)
        ss_t = float(((c_o - grand) ** 2).sum())
        M["form_structure"] = {
            "block_size": 20, "block_CER_quantiles_10_25_50_75_90_99": [float(v) for v in np.percentile(bc, [10, 25, 50, 75, 90, 99])],
            "worst_block_first_line_index": int(np.asarray(R["idx"])[order][int(np.argmax(bc)) * 20]), "worst_block_CER": float(bc.max()),
            "lag1_autocorrelation_of_line_CER": float(np.corrcoef(c_o[:-1], c_o[1:])[0, 1]) if len(c_o) > 2 and c_o.std() > 0 else float("nan"),
            "between_block_variance_share_30_line_blocks": float(ss_b / ss_t) if ss_t > 0 else float("nan"),
            "same_share_expected_without_structure": float((len(ub) - 1) / max(len(c_o) - 1, 1)),
            "note": "consecutive evaluation lines are stored form by form: a share far above the 'expected' value means errors cluster by form / writer"}
        sty = np.asarray(R["style_raw"], np.float64)
        ed, rl = np.asarray(R["ed"], np.float64), np.maximum(np.asarray(R["reflen"], np.float64), 1.0)
        bins: Dict[str, Any] = {}
        for j, name in enumerate(STYLE_NAMES):
            v = sty[:, j]
            ok = np.isfinite(v)
            if ok.sum() < 50:
                continue
            q = np.quantile(v[ok], np.linspace(0, 1, 6))
            rows = []
            for b in range(5):
                m = ok & (v >= q[b]) & ((v <= q[b + 1]) if b == 4 else (v < q[b + 1]))
                rows.append({"range": [float(q[b]), float(q[b + 1])], "n": int(m.sum()), "CER": float(ed[m].sum() / rl[m].sum()) if m.any() else float("nan")})
            rk_v, rk_c = np.argsort(np.argsort(v[ok])), np.argsort(np.argsort((ed / rl)[ok]))
            rho = float(np.corrcoef(rk_v, rk_c)[0, 1]) if rk_v.std() > 0 and rk_c.std() > 0 else float("nan")
            bins[name] = {"quintiles": rows, "spearman_with_line_CER": rho}
        M["cer_by_style_quintile"] = bins
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
        cls_rate = [M["by_character_class"][c]["error_rate_on_reference_chars"] * 100 for c in CHAR_CLASSES]
        ax[0].bar(CHAR_CLASSES, np.nan_to_num(cls_rate), color="indianred")
        for x_, c in enumerate(CHAR_CLASSES):
            ax[0].text(x_, np.nan_to_num(cls_rate[x_]), f"{100 * M['by_character_class'][c]['share_of_reference_chars']:.1f}% of chars", ha="center", va="bottom", fontsize=7)
        ax[0].set_ylabel("error rate on reference characters (%)"), ax[0].set_title("which characters fail?", fontsize=9)
        ax[1].plot(bc * 100, lw=1.2), ax[1].axhline(100 * float(e_o.sum() / l_o.sum()), color="k", ls="--", lw=1)
        ax[1].set_xlabel("block of 20 consecutive lines (form order)"), ax[1].set_ylabel("block CER (%)")
        ax[1].set_title(f"errors cluster by form / writer (lag-1 corr {M['form_structure']['lag1_autocorrelation_of_line_CER']:.2f})", fontsize=9)
        for name, b in bins.items():
            ax[2].plot(range(1, 6), [r["CER"] * 100 for r in b["quintiles"]], "o-", label=f"{name} (rho {b['spearman_with_line_CER']:+.2f})")
        ax[2].set_xlabel("quintile of the measured descriptor (1 = lowest)"), ax[2].set_ylabel("CER (%)"), ax[2].legend(fontsize=7)
        ax[2].set_title("CER by handwriting style", fontsize=9)
        self._fig("03_error_structure/error_structure.png", "Where the errors come from: character classes, forms, handwriting style", fig)
        return M
