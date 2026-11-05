"""
The training loop: curriculum schedules, the EMA teacher, FRAG-RC calibration refreshes, the legacy line-level pseudo-labels, checkpoint
selection on val_sel, early stopping, resume (warm restart instead of an lr jump) and warm starts from another checkpoint.
"""
from __future__ import annotations

import glob
import json
import math
import os
import random
import signal
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from .alphabet import CharGroups, extend_ctc_heads
from .config import Config
from .curves import plot_training_curves
from .data import HandwritingDataModule, HandwritingDataset
from .decoding import greedy_decode, seq_confidence
from .frag import FRAG_FEATURES, FragModel, cell_theta, char_evidence, char_wrong_labels, fit_trust_rule, frag_trust_batch
from .llm import LLMEngine, fuse_scores
from .model import ClosedFormLeakage, GPUAugment, InkLinguaNet, ModelEMA, compute_losses
from .pseudo import PERTURBATIONS, RELIABILITY_FEATURES, PseudoLabelBank, SelectionRule, apply_perturbation, eval_noise, reliability_features, reliability_views
from .riskcontrol import auroc
from .style import NUM_DESC, STYLE_NAMES
from .text import cer, detokenize, edit_distance, grapheme_clusters
from .utils import LOG, Meter, atomic_save, json_safe, to_device, torch_load

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


def warm_start(model: InkLinguaNet, path: str, codec, strict_alphabet: bool = False) -> Dict[str, Any]:
    """Load the weights of an InkLingua checkpoint (best.pt / last.pt / adapted.pt) into `model`, whose output layer has already been built
    for `codec`.  If the checkpoint's alphabet is a PREFIX of the new one (the data module built the codec with base_units), the new units are
    appended as tied copies ([ALPHA-SPLIT]); otherwise the alphabets must be identical."""
    st = torch_load(path)
    weights = st["weights"] if "weights" in st else st["ema"]
    old_units = list(st.get("units") or grapheme_clusters(st["chars"]))
    new_units = list(codec.units)
    info = {"path": path, "epoch": st.get("epoch"), "old_alphabet": len(old_units), "new_alphabet": len(new_units), "appended_units": 0}
    if old_units == new_units:
        model.load_state_dict(weights)
        return info
    if strict_alphabet or new_units[:len(old_units)] != old_units:
        raise RuntimeError(f"warm start: the checkpoint alphabet ({len(old_units)} units) differs from the current one ({len(new_units)}); "
                           "build the data module with base_units=<checkpoint units> (mode 'finetune' does this) or set init_strict_alphabet=False "
                           "with an extending alphabet")
    V0 = len(old_units) + 1
    # build the network at the OLD size, load, then grow the heads function-preservingly
    for name in ("ctc_head", "aux_head"):
        lin = getattr(model, name)
        setattr(model, name, nn.Linear(lin.in_features, V0).to(lin.weight.device))
    model.num_classes = V0
    model.load_state_dict(weights)
    groups = CharGroups(old_units, new_units)
    extend_ctc_heads(model, groups)
    info.update({"appended_units": len(new_units) - len(old_units), "variant_groups": groups.n_variant_groups, "orphans": len(groups.orphans),
                 "orphan_units": "".join(new_units[i - 1] for i in groups.orphans)})
    return info


class Trainer:
    def __init__(self, cfg: Config, data: HandwritingDataModule, model: InkLinguaNet, llm: LLMEngine, device: torch.device):
        self.cfg, self.data, self.llm, self.device = cfg, data, llm, device
        self.tc, self.pc, self.fc = cfg.train, cfg.train.pseudo, cfg.train.frag
        for v in self.pc.cons_views:
            if v not in PERTURBATIONS:
                raise ValueError(f"unknown consistency view '{v}'; choose from {list(PERTURBATIONS)}")
        if self.pc.consistency and not self.pc.cons_views:
            raise ValueError("PseudoConfig.consistency=True needs at least one view in cons_views")
        self.model = model.to(device)
        self.init_info: Optional[Dict[str, Any]] = None
        if self.tc.init_from:
            self.init_info = warm_start(self.model, self.tc.init_from, data.codec, self.tc.init_strict_alphabet)
            LOG.info("warm start from %s (epoch %s): alphabet %d -> %d units (+%d appended as tied copies)", self.tc.init_from, self.init_info["epoch"],
                     self.init_info["old_alphabet"], self.init_info["new_alphabet"], self.init_info["appended_units"])
        self.ema = ModelEMA(self.model, self.tc.ema_decay)
        self.aug = GPUAugment(self.tc, data.style_mean, data.style_std)
        ds = data.train_ds
        self.shadow = bool(self.pc.shadow_calibration)
        self.bank = PseudoLabelBank(len(ds), ~ds.labeled & (~ds.calib | self.shadow), ds.oracle, np.ceil(ds.width / 4).astype(np.int64),
                                    calib=ds.calib if self.shadow else None)
        self.prev_cov: Optional[float] = None
        self.cov_hist: List[float] = []
        self.leak = ClosedFormLeakage(cfg.model.d_model, NUM_DESC, self.tc.leak_queue, self.tc.leak_ridge, self.tc.leak_slack, data.leak_active,
                                      np.clip(data.text_baseline_r2, 0.0, 1.0), device, self.tc.leak_refit_every)
        self.use_amp = bool(self.tc.amp and device.type == "cuda")
        self.amp_dtype = torch.float32
        if self.use_amp:   # bf16 only where it is native (compute capability >= 8); fp16 + GradScaler elsewhere
            self.amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        decay, no_decay = [], []
        for n, p in self.model.named_parameters():
            (decay if p.ndim >= 2 else no_decay).append(p)
        peak = self.tc.lr * (self.tc.init_lr_scale if self.tc.init_from else 1.0)
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": self.tc.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
                                     lr=peak, betas=(0.9, 0.98))
        self.steps_per_epoch = int(math.ceil(len(ds) / cfg.data.batch_size))
        self.total_steps = self.tc.epochs * self.steps_per_epoch
        self.cycles: List[Dict[str, float]] = [{"start": 0, "end": self.total_steps, "peak": peak, "warm": self.tc.warmup_steps, "lo": 0.0}]
        self.global_step, self.start_epoch = 0, 0
        self.history: Dict[str, List[float]] = {"epoch": []}
        self.best_cer, self.best_from, self.best_epoch = float("inf"), "student", 0
        self.ref_cer, self.since_best, self.stopped_at = float("inf"), 0, None
        self.prev_val_preds: Optional[List[str]] = None
        self.prev_val_cer: Optional[np.ndarray] = None
        self.skipped = 0
        self.interrupted, self.completed_epochs = False, 0
        self._resume_partial: Optional[Dict[str, Any]] = None       # set by _try_resume when last.pt was written in the middle of an epoch
        self._last_ckpt = time.time()
        self._stop_signal: Optional[int] = None
        self._phase_secs: Dict[str, List[float]] = {"refresh": [], "eval": []}
        # ---- FRAG-RC state: reliability model + one threshold per cell (re-fitted every `refresh_every` epochs)
        self.frag_state: Optional[Dict[str, Any]] = None
        self.frag_model: Optional[FragModel] = None
        self.stratum_w = np.ones(max(1, data.n_strata))
        if self.fc.enabled and self.pc.enabled:
            LOG.warning('FRAG-RC and the legacy line-level pseudo-labels are both enabled: both write into the loss (use Config.ablation("legacy_pl") '
                        'for the legacy baseline alone)')
        if self.fc.enabled and len(data.calib_idx) == 0:
            LOG.warning('FRAG-RC needs held-out calibration lines (cfg.data.calib_fraction > 0): without them the trust rule is never fitted '
                        'and only the consistency loss acts on the unlabeled lines')

    # ------------------------------------------------------------------ schedules
    def _lr(self, step: int) -> float:
        tc, cyc = self.tc, self.cycles[-1]
        for c in self.cycles:
            if step < c["end"]:
                cyc = c
                break
        t = step - cyc["start"]
        if t < cyc["warm"]:
            return float(cyc["lo"] + (cyc["peak"] - cyc["lo"]) * (t + 1) / max(cyc["warm"], 1))
        p = (t - cyc["warm"]) / max(1.0, cyc["end"] - cyc["start"] - cyc["warm"])
        return float(cyc["peak"] * (tc.min_lr_ratio + (1 - tc.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))))

    def _extend_schedule(self, old_total: int) -> None:
        """cfg.train.epochs changed between the checkpoint and this run: shorter -> the last cycle is cut; longer -> a NEW cycle is appended
        (warm restart to restart_lr_scale * lr after a finished cycle, continuation from the current lr after an interrupted one)."""
        tc, new_total = self.tc, self.total_steps
        last = self.cycles[-1]
        if new_total == last["end"]:
            return
        if new_total < last["end"]:
            last["end"] = max(new_total, self.global_step + 1)
            LOG.warning("cfg.train.epochs decreased: the last lr cycle now ends at step %d", last["end"])
            return
        if tc.extend_policy == "error":
            raise RuntimeError("cfg.train.epochs was increased on resume and cfg.train.extend_policy='error'")
        lr_now = self._lr(self.global_step)
        finished = self.global_step >= last["end"]
        if not finished:
            last["end"] = self.global_step
        peak = min(tc.restart_lr_scale * tc.lr, tc.lr) if finished else lr_now
        warm = tc.restart_warmup_steps if finished else 0
        self.cycles.append({"start": self.global_step, "end": new_total, "peak": float(peak), "warm": warm, "lo": float(lr_now)})
        LOG.warning("cfg.train.epochs changed (%d -> %d steps): new %s lr cycle from step %d, peak lr %.2e (lr now %.2e)", old_total, new_total,
                    "warm-restart" if finished else "continuation", self.global_step, peak, lr_now)

    def _ramp(self, step: int, start_ep: float, ramp_ep: float) -> float:
        spe = self.steps_per_epoch
        return float(np.clip((step - start_ep * spe) / max(1.0, ramp_ep * spe), 0.0, 1.0))

    def _aux_scale(self, step: int) -> float:
        return self._ramp(step, self.tc.aux_start_epoch, self.tc.aux_ramp_epochs)

    def _vq_lambda(self, step: int) -> float:
        return self._ramp(step, self.tc.vq_start_epoch, self.tc.vq_anneal_epochs)

    @property
    def best_path(self) -> str:
        return os.path.join(self.cfg.ckpt_dir, "best.pt")

    # ------------------------------------------------------------------ one epoch
    def _epoch_pool(self, epoch: int) -> np.ndarray:
        """Lines visited in one epoch: every line that has a recognition target (labeled + selected pseudo-labels) plus a random subset of the
        remaining unlabeled / calibration lines (images only: style / masking / VQ / consistency / FRAG-RC).  Labeled lines of harder cells are
        repeated ([STRATA]); with several corpora, `domain_balance` repeats lines of the smaller ones."""
        ds, tc, dc = self.data.train_ds, self.tc, self.cfg.data
        rng = np.random.RandomState(self.cfg.run.seed * 1000 + epoch)
        act = np.flatnonzero(ds.tgt_weight > 0)
        rest = np.flatnonzero(ds.tgt_weight <= 0)
        fc = self.fc
        ratio = fc.unlabeled_ratio if fc.enabled else tc.unlabeled_aux_ratio
        n_aux = len(rest) if len(act) == 0 else min(len(rest), int(round(ratio * len(act))))
        aux = rng.choice(rest, n_aux, replace=False) if n_aux > 0 else np.zeros(0, np.int64)
        lab = np.flatnonzero(ds.labeled & (ds.tgt_weight > 0))
        extra = np.zeros(len(lab))
        if fc.enabled and fc.stratum_emphasis > 0 and self.frag_state is not None and len(self.stratum_w) == self.data.n_strata:
            extra += self.stratum_w[np.clip(ds.stratum[lab], 0, len(self.stratum_w) - 1)] - 1.0
        n_dom = len(self.data.specs)
        if dc.domain_balance > 0 and n_dom > 1 and len(lab):
            cnt = np.bincount(ds.domain[lab], minlength=n_dom).astype(np.float64)
            wd = (cnt.max() / np.maximum(cnt, 1.0)) ** float(np.clip(dc.domain_balance, 0.0, 1.0))
            extra += wd[ds.domain[lab]] - 1.0
        if extra.any():
            k = np.floor(extra).astype(np.int64) + (rng.rand(len(lab)) < (extra - np.floor(extra))).astype(np.int64)
            act = np.concatenate([act, np.repeat(lab, np.maximum(k, 0))]).astype(np.int64)
        pool = np.concatenate([act, aux]).astype(np.int64)
        need = self.steps_per_epoch * self.cfg.data.batch_size
        reps = int(math.ceil(need / max(len(pool), 1)))
        return np.concatenate([rng.permutation(pool) for _ in range(reps)])[:need]

    # ------------------------------------------------------------------ stopping and step checkpoints
    def _deadline_left(self) -> float:
        return float("inf") if self.tc.deadline_unix <= 0 else self.tc.deadline_unix - time.time()

    def _phase_fits(self, phase: str) -> bool:
        """Whether a phase of typical duration (refresh / end-of-epoch evaluation, measured on this run) still fits before the deadline."""
        past = self._phase_secs[phase]
        return self._deadline_left() > 1.2 * (float(np.median(past)) if past else 0.0) + 60.0

    def _must_stop(self) -> bool:
        return self._stop_signal is not None or self._deadline_left() < 90.0

    def save_step_checkpoint(self, epoch: int, step: int, batches: List[List[int]], meter: Meter, rec: Dict[str, float]) -> None:
        """last.pt in the middle of epoch `epoch` (0-based) after `step` of its batches: everything the step loop needs to continue exactly where
        it stopped (the epoch's batch list, the running means, the refresh metrics of the epoch, every random state)."""
        st = self._state(epoch)
        st["partial"] = {"epoch": int(epoch), "step": int(step), "batches": batches, "meter": meter.state(), "rec": dict(rec)}
        atomic_save(st, os.path.join(self.cfg.ckpt_dir, "last.pt"))
        self._last_ckpt = time.time()

    def _install_signal_handlers(self):
        """SIGTERM (a scheduler / the experiment plan stopping the run) -> finish the current step, write a step checkpoint, stop."""
        if threading.current_thread() is not threading.main_thread():
            return None

        def handler(signum, _frame):
            self._stop_signal = int(signum)
        try:
            return signal.signal(signal.SIGTERM, handler)
        except (ValueError, OSError):
            return None

    def _train_epoch(self, epoch: int, rec: Optional[Dict[str, float]] = None, resume: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, float]]:
        """One epoch of optimisation steps.  Returns the running means, or None when the run had to stop in the middle (deadline, SIGTERM,
        KeyboardInterrupt): a step checkpoint has then been written and the next start continues with the next batch."""
        self.model.train()
        rec = dict(rec or {})
        if resume is not None:
            batches, start, meter = resume["batches"], int(resume["step"]), Meter.from_state(resume["meter"])
            LOG.info("continuing epoch %d at step %d of %d", epoch + 1, start, len(batches))
        else:
            batches, start, meter = self.data.epoch_batches(self._epoch_pool(epoch), epoch), 0, Meter()
        loader = self.data.train_loader(batches=batches[start:])
        self.device
        keys = ("total", "ctc", "aux_ctc", "style_reg", "style_cls", "leak", "cons", "decor", "vq", "mask", "frag")
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{self.tc.epochs}", leave=False, initial=start, total=len(batches))
        step = start
        try:
            for batch in bar:
                self._train_step(batch, keys, meter)
                step += 1
                if self._must_stop():
                    self.save_step_checkpoint(epoch, step, batches, meter, rec)
                    why = f"signal {self._stop_signal}" if self._stop_signal is not None else "the deadline"
                    LOG.warning("[STOP] %s: step checkpoint at epoch %d, step %d/%d -- the next start continues from there", why, epoch + 1, step, len(batches))
                    self.interrupted = True
                    return None
                if time.time() - self._last_ckpt >= 60.0 * self.tc.ckpt_every_minutes:
                    self.save_step_checkpoint(epoch, step, batches, meter, rec)
                bar.set_postfix(loss=f"{meter.mean('total'):.3f}", ctc=f"{meter.mean('ctc'):.3f}", refresh=False)
        except KeyboardInterrupt:
            self.save_step_checkpoint(epoch, step, batches, meter, rec)
            LOG.warning("[STOP] interrupted by the user: step checkpoint at epoch %d, step %d/%d", epoch + 1, step, len(batches))
            raise
        self._epoch_batches, self._epoch_meter = batches, meter
        return meter.means()

    def _train_step(self, batch: Dict[str, Any], keys: Tuple[str, ...], meter: Meter) -> None:
        """One optimisation step.  Two host synchronisations per step: the finiteness test of the loss and ONE transfer of every logged value."""
        dev, tc = self.device, self.tc
        teacher = self.ema.module.eval()
        batch = to_device(batch, dev)
        lr = self._lr(self.global_step)
        for g in self.opt.param_groups:
            g["lr"] = lr
        x_s, x_w, wd, st_t, st_m, cl_ok = self.aug(batch["image"], batch["style"], batch["style_ok"], batch["width"], batch["min_width"])
        aux, vql = self._aux_scale(self.global_step), self._vq_lambda(self.global_step)
        t_lp, t_out, frag = None, None, None
        use_frag = self.fc.enabled and self.frag_model is not None and tc.loss.frag > 0
        if (tc.loss.consistency > 0 or use_frag) and aux > 0:
            with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                t_out = teacher(x_w, wd)                     # no_grad, not inference_mode: these tensors enter the student's loss
            t_lp = t_out["log_probs"].float()
        if use_frag and t_out is not None:
            frag = self._frag_batch(t_out, x_s, wd, (~batch["labeled"]) & (batch["weight"] <= 0), batch["stratum"])
        with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
            out = self.model(x_s, wd, vq_lambda=vql, mask_prob=self.cfg.model.mask_prob * aux)
        L = compute_losses(out, batch, tc.loss, st_t, st_m, cl_ok, True, aux_scale=aux, vq_scale=vql, teacher_lp=t_lp, leak=self.leak, frag=frag)
        self.opt.zero_grad(set_to_none=True)
        if not torch.isfinite(L["total"]):
            self.skipped += 1
            return
        self.scaler.scale(L["total"]).backward()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), tc.grad_clip)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.ema.update(self.model, self.global_step)
        self.global_step += 1

        B = batch["image"].size(0)
        ps, lab = L["_ctc_ps"].detach().float(), batch["labeled"]
        pl = (~lab) & (batch["weight"] > 0)
        leak = self.leak.last_vals if self.leak.last_vals is not None else ps.new_zeros(0)
        head = torch.stack([L[k].detach().float() for k in keys] + [gn.detach().float(), out["ppl_low"].detach().float(), out["ppl_high"].detach().float(),
                            (ps * lab).sum(), lab.sum().float(), (ps * pl).sum(), pl.sum().float()])
        v = torch.cat([head, leak.float(), ps, pl.float()]).cpu().numpy()          # the one host transfer of the step
        nk = len(keys)
        for k, x in zip(keys, v[:nk]):
            meter.add(k, float(x), B)
        lc_ = tc.loss
        meter.add("task_total", float(v[1] + lc_.style_reg * v[3] + lc_.style_cls * v[4]), B)
        meter.add("grad_norm", float(v[nk]), 1)
        meter.add("ppl_low", float(v[nk + 1]), 1)
        meter.add("ppl_high", float(v[nk + 2]), 1)
        s_lab, n_lab, s_pl, n_pl = (float(x) for x in v[nk + 3:nk + 7])
        o = nk + 7
        for j, x in zip(self.leak.active if len(leak) else [], v[o:o + len(leak)]):
            meter.add(f"leak_r2_{j}", float(x), 1)
        o += len(leak)
        ps_h, pl_h = v[o:o + B], v[o + B:o + 2 * B] > 0.5
        if frag is not None:
            meter.add("frag_trust", frag["trusted_frames"] / max(frag["frames"], 1), 1)
            meter.add("frag_char_trust", frag["n_trusted_chars"] / max(frag["n_chars"], 1), 1)
        if n_lab > 0:
            meter.add("ctc_labeled", s_lab / n_lab, int(n_lab))
        if n_pl > 0:
            meter.add("ctc_pseudo", s_pl / n_pl, int(n_pl))
            self.bank.update_loss(batch["index"][torch.from_numpy(pl_h).to(batch["index"].device)].cpu().tolist(), ps_h[pl_h].tolist())

    # ------------------------------------------------------------------ evaluation
    @torch.inference_mode()
    def evaluate(self, model: nn.Module, ds: HandwritingDataset, indices: Optional[Sequence[int]] = None, noise: bool = False) -> Dict[str, Any]:
        model.eval()
        loader = self.data.eval_loader(ds, indices)
        n = len(ds)
        preds: List[Optional[str]] = [None] * n
        conf = np.full(n, np.nan)
        ed_ps, ref_ps = np.full(n, np.nan), np.full(n, np.nan)
        meter, ed_w, ref_w = Meter(), 0, 0
        s_abs, s_cnt = np.zeros(NUM_DESC), np.zeros(NUM_DESC)
        dev, codec = self.device, self.data.codec
        for batch in loader:
            batch = to_device(batch, dev)
            x = apply_perturbation(eval_noise, batch["image"], batch["width"]) if noise else batch["image"]
            with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = model(x, batch["width"])
            B = x.size(0)
            sm = batch["style_ok"].float()
            L = compute_losses(out, batch, self.tc.loss, batch["style"], sm, torch.ones(B, dtype=torch.bool, device=dev), False)
            for k in ("total", "ctc", "style_reg", "style_cls"):
                meter.add(k, float(L[k]), B)
            err = ((out["style_reg"] - batch["style"]).abs() * sm).cpu().numpy()
            s_abs += err.sum(0)
            s_cnt += sm.cpu().numpy().sum(0)
            valid_cl = batch["cluster"] >= 0
            if bool(valid_cl.any()):
                meter.add("style_acc", float((out["style_cls"].argmax(1) == batch["cluster"])[valid_cl].float().mean()), int(valid_cl.sum()))
            lp, lens = out["log_probs"].cpu().numpy(), out["out_lens"].cpu().tolist()
            for b in range(B):
                ids, cf, _ = greedy_decode(lp[b, :lens[b]])
                text, i = codec.decode_ids(ids), int(batch["index"][b])
                ref = batch["oracle"][b]
                preds[i], conf[i] = text, seq_confidence(cf)
                ed_ps[i], ref_ps[i] = edit_distance(ref, text), len(ref)
                ed_w, ref_w = ed_w + edit_distance(ref.split(), text.split()), ref_w + len(ref.split())
        res = dict(meter.means())
        done = np.isfinite(ed_ps)
        res.update({"cer": float(ed_ps[done].sum() / max(ref_ps[done].sum(), 1)), "wer": ed_w / max(ref_w, 1),
                    "style_mae": float(s_abs.sum() / max(s_cnt.sum(), 1.0)), "style_mae_desc": s_abs / np.maximum(s_cnt, 1.0),
                    "preds": preds, "conf": conf, "ed_ps": ed_ps, "ref_ps": ref_ps, "cer_ps": ed_ps / np.maximum(ref_ps, 1)})
        return res

    # ------------------------------------------------------------------ legacy pseudo-labelling (line level)
    @torch.inference_mode()
    def _teacher_read(self, idxs: np.ndarray, keep_lp: bool = False, desc: str = "teacher") -> Dict[str, Any]:
        """The EMA teacher reads the given train lines (clean view + the perturbed views) and describes every reading with the label-free
        reliability features."""
        pc, dev, codec = self.pc, self.device, self.data.codec
        teacher = self.ema.module.eval()
        views = reliability_views(pc)
        n_gate = len(pc.cons_views) if pc.consistency else 0
        r: Dict[str, Any] = {"idx": [], "text": [], "conf": [], "cons": [], "X": [], "lp": {}}
        for batch in tqdm(self.data.eval_loader(self.data.train_ds, idxs), desc=desc, leave=False):
            batch = to_device(batch, dev)
            with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                out = teacher(batch["image"], batch["width"])
            lp, lens = out["log_probs"].cpu().numpy(), out["out_lens"].cpu().tolist()
            B = lp.shape[0]
            reads = []
            for b in range(B):
                ids, cf, fr = greedy_decode(lp[b, :lens[b]])
                reads.append((ids, cf, fr, codec.decode_ids(ids)))
            cons_v = np.zeros((B, len(views)))
            for vi, v in enumerate(views):
                xv = apply_perturbation(PERTURBATIONS[v], batch["image"], batch["width"])
                with torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
                    ov = teacher(xv, batch["width"])
                lpv, lnv = ov["log_probs"].cpu().numpy(), ov["out_lens"].cpu().tolist()
                for b in range(B):
                    cons_v[b, vi] = cer(reads[b][3], codec.decode_ids(greedy_decode(lpv[b, :lnv[b]])[0]))
            for b in range(B):
                ids, cf, fr, text = reads[b]
                r["idx"].append(int(batch["index"][b]))
                r["text"].append(text)
                r["conf"].append(seq_confidence(cf))
                r["cons"].append(float(cons_v[b, :n_gate].mean()) if n_gate else float("nan"))
                r["X"].append(reliability_features(lp[b, :lens[b]], ids, cf, fr, cons_v[b]))
                if keep_lp:
                    r["lp"][int(batch["index"][b])] = lp[b, :lens[b]].copy()
        r["X"] = np.array(r["X"]) if r["X"] else np.zeros((0, len(RELIABILITY_FEATURES)))
        return r

    def _save_source_calibration(self, X: np.ndarray, ed: np.ndarray, rl: np.ndarray, epoch: int) -> None:
        """checkpoints/source_calibration.npz: the SOURCE side of the transfer-calibrated risk control of the zero-shot adapter."""
        try:
            os.makedirs(self.cfg.ckpt_dir, exist_ok=True)
            np.savez(os.path.join(self.cfg.ckpt_dir, "source_calibration.npz"), X=np.asarray(X, np.float64), ed=np.asarray(ed, np.float64),
                     rl=np.asarray(rl, np.float64), epoch=int(epoch), chars=self.data.codec.chars, features=np.array(RELIABILITY_FEATURES))
        except Exception as e:
            LOG.warning("could not write source_calibration.npz (%s)", e)

    def _ref_cov(self) -> Optional[float]:
        return float(np.median(self.cov_hist)) if self.cov_hist else None

    @torch.no_grad()
    def refresh_pseudo_labels(self, epoch: int) -> Dict[str, float]:
        """[RC-PL v2] risk-controlled line-level pseudo-labels with SHADOW calibration (the v9 method, kept as a baseline)."""
        pc, bank, ds = self.pc, self.bank, self.data.train_ds
        bank.begin_round()
        use_llm = bool(pc.llm_refine and self.llm.can_score())
        rng = np.random.RandomState(self.cfg.run.seed + 17 * epoch)
        rule: Optional[SelectionRule] = None
        info: Dict[str, float] = {}
        cal_idx = self.data.calib_idx
        rc = None
        if len(cal_idx) >= pc.min_calib:
            rc = self._teacher_read(cal_idx, keep_lp=use_llm and self.shadow, desc="calibration lines")
            ref = [ds.reference(i) for i in rc["idx"]]
            ed = np.array([edit_distance(a, b) for a, b in zip(ref, rc["text"])], np.float64)
            rl = np.array([len(a) for a in ref], np.float64)
            rule = SelectionRule.fit(rc["X"], ed, rl, pc, seed=self.cfg.run.seed + epoch, prev_cov=self.prev_cov, ref_cov=self._ref_cov())
            if pc.save_source_calibration:
                self._save_source_calibration(rc["X"], ed, rl, epoch)
            info = {"rc_" + k: v for k, v in rule.info.items()}
            info["rc_theta"] = float(rule.theta if rule.theta is not None else float("nan"))
            info["rc_theta_llm"] = float(rule.theta_llm if rule.theta_llm is not None else float("nan"))
            self.prev_cov = float(rule.info["calib_coverage"])
            if rule.info["calib_cov_raw"] * rule.info["calib_n"] >= pc.min_selected_calib:
                self.cov_hist = (self.cov_hist + [self.prev_cov])[-int(pc.cov_ref_window):]
        else:
            LOG.warning("only %d calibration lines (< min_calib=%d): FALLBACK to the sequence-confidence threshold %.2f -- expect noisy pseudo-labels",
                        len(cal_idx), pc.min_calib, pc.fallback_tau_conf)
        theta = None if rule is None else (rule.theta if rule.theta is not None else float("-inf"))
        theta_llm = None if rule is None else (rule.theta_llm if rule.theta_llm is not None else float("-inf"))
        use_llm = use_llm and rule is not None and np.isfinite(theta_llm)
        reservoir: List[Tuple[int, np.ndarray]] = []
        n_border = 0
        un = self._teacher_read(self.data.unlabeled_idx, keep_lp=use_llm, desc="pseudo-labelling")
        risk = rule.predict(un["X"]) if rule is not None else np.full(len(un["idx"]), np.nan)
        bank.update_predictions(un["idx"], un["text"], un["conf"], un["cons"], risk)
        pools = [(un, risk)]
        if self.shadow and rc is not None and rule is not None:
            bank.update_predictions(rc["idx"], rc["text"], rc["conf"], rc["cons"], rule.oof)
            pools.append((rc, rule.oof))
        if use_llm:
            for src, rk in pools:
                for k, i in enumerate(src["idx"]):
                    if i not in src["lp"]:
                        continue
                    gate = (not pc.consistency) or (np.isfinite(src["cons"][k]) and src["cons"][k] <= pc.tau_cons)
                    if gate and theta < rk[k] <= theta_llm:
                        n_border += 1
                        if len(reservoir) < pc.llm_budget:
                            reservoir.append((i, src["lp"][i]))
                        else:
                            j = rng.randint(0, n_border)
                            if j < pc.llm_budget:
                                reservoir[j] = (i, src["lp"][i])
        if use_llm and reservoir:
            from .decoding import ctc_prefix_beam_search
            self._llm_rescue([(i, ctc_prefix_beam_search(l, pc.llm_beam)) for i, l in reservoir], n_border)
        bank.select(pc, theta, theta_llm)
        ds.set_pseudo(bank.train_text, bank.weight, include_calib=self.shadow)
        d = bank.diagnostics()
        d.update(info)
        d["pl_fallback"] = float(rule is None)
        LOG.info("pseudo-labels (%s): selected=%d (LLM-rescued %d) coverage=%.2f | selected CER vs hidden GT=%.4f exact=%.3f | all-unlabeled CER=%.4f",
                 "risk-controlled" if rule is not None else "FALLBACK", int(d.get("pl_selected", 0)), int(d.get("pl_llm_selected", 0)),
                 d.get("pl_coverage", 0), d.get("pl_sel_cer", float("nan")), d.get("pl_sel_exact", float("nan")), d.get("pl_all_cer", float("nan")))
        return d

    def _llm_rescue(self, border: List[Tuple[int, list]], n_border_total: int) -> None:
        pc, codec, bank, ds = self.pc, self.data.codec, self.bank, self.data.train_ds
        raw, det = [], []
        for i, nb in border:
            for ids, _lp in nb:
                t = codec.decode_ids(ids)
                raw.append(t), det.append(detokenize(ds.display(i, t)) if ds.detok(i) else ds.display(i, t))
        langs = [ds.language(i) for i, nb in border for _ in nb]
        lm_lp, n_tok = np.zeros(len(det)), np.zeros(len(det))
        for lang in sorted(set(langs)):
            ix = [k for k, l in enumerate(langs) if l == lang]
            a, b = self.llm.score_texts([det[k] for k in ix], lang)
            lm_lp[ix], n_tok[ix] = a, b
        pos, changed, agreed = 0, 0, 0
        for i, nb in border:
            K = len(nb)
            fused = fuse_scores(np.array([lp for _, lp in nb]), lm_lp[pos:pos + K], n_tok[pos:pos + K], pc.llm_beta, pc.llm_rho)
            best = int(np.argmax(fused))
            lm_best, ntk_best = lm_lp[pos + best], max(n_tok[pos + best], 1)
            top_t, best_t = raw[pos], raw[pos + best]
            pos += K
            if lm_best / ntk_best < pc.llm_min_tok_lp or not best_t:
                continue
            if best == 0:
                bank.llm_ok[i], bank.llm_text[i] = True, top_t
                agreed += 1
            elif edit_distance(best_t, top_t) / max(len(top_t), 1) <= pc.llm_max_edit_ratio:
                bank.llm_ok[i], bank.llm_text[i] = True, best_t
                changed += 1
        LOG.info("LLM rescoring: %d of %d borderline lines -> %d confirmed, %d corrected", len(border), n_border_total, agreed, changed)

    # ------------------------------------------------------------------ [FRAG-RC]
    def _frag_set_state(self, st: Optional[Dict[str, Any]]) -> None:
        self.frag_state = st
        self.frag_model = FragModel.from_state(st["model"]) if st else None

    def _teacher_pair(self, x_w: torch.Tensor, x_s: torch.Tensor, wd: torch.Tensor):
        dev, teacher = self.device, self.ema.module.eval()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
            return teacher(x_w, wd), teacher(x_s, wd)

    @torch.inference_mode()
    def _frag_collect(self, idxs: np.ndarray, ref_fn, passes: int, seed: int) -> Dict[str, np.ndarray]:
        """Evidence (+ correctness against `ref_fn(line)`) of every emitted character of the given train lines, as the training loop will see
        them: teacher on a weak and a strong augmentation of the same pixels."""
        ds, dev, codec = self.data.train_ds, self.device, self.data.codec
        X, Y, LN, ST = [], [], [], []
        for p in range(passes):
            with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else []):
                torch.manual_seed(seed + 101 * p)
                for batch in self.data.eval_loader(ds, idxs):
                    batch = to_device(batch, dev)
                    x_s, x_w, wd, _a, _b, _c = self.aug(batch["image"], batch["style"], batch["style_ok"], batch["width"], batch["min_width"])
                    o_w, o_s = self._teacher_pair(x_w, x_s, wd)
                    lp_w, aux_w = o_w["log_probs"].float().cpu().numpy(), o_w["aux_log_probs"].float().cpu().numpy()
                    lp_s, lens = o_s["log_probs"].float().cpu().numpy(), o_w["out_lens"].cpu().numpy()
                    for b in range(lp_w.shape[0]):
                        i, L = int(batch["index"][b]), int(lens[b])
                        ids, st, en, Fm = char_evidence(lp_w[b, :L], lp_s[b, :L], aux_w[b, :L])
                        if not len(ids):
                            continue
                        y = char_wrong_labels(ref_fn(i), codec.decode_ids(ids))
                        X.append(Fm), Y.append(y), LN.append(np.full(len(ids), i)), ST.append(np.full(len(ids), int(ds.stratum[i])))
        if not X:
            return {"X": np.zeros((0, len(FRAG_FEATURES))), "y": np.zeros(0), "line": np.zeros(0, np.int64), "strat": np.zeros(0, np.int64)}
        return {"X": np.concatenate(X), "y": np.concatenate(Y).astype(np.float64), "line": np.concatenate(LN), "strat": np.concatenate(ST)}

    def refresh_frag(self, epoch: int) -> Dict[str, float]:
        """Re-calibrate the character-level trust rule on the held-out calibration lines: evidence of every emitted character (true text used
        ONLY to mark characters wrong) -> cross-fitted model -> one threshold per cell -> diagnostics on a sample of UNLABELED lines."""
        fc, ds, S = self.fc, self.data.train_ds, self.data.n_strata
        cal_idx = self.data.calib_idx
        info: Dict[str, float] = {}
        if len(cal_idx) < 20:
            LOG.warning("[FRAG-RC] only %d calibration lines: no trust rule (set cfg.data.calib_fraction > 0)", len(cal_idx))
            return info
        cal = self._frag_collect(cal_idx, ds.reference, fc.calib_passes, self.cfg.run.seed + epoch)
        X, y, ln, stt = cal["X"], cal["y"], cal["line"], cal["strat"]
        if len(y) < 2 * fc.min_chars:
            LOG.warning("[FRAG-RC] %d calibration characters (< %d): no trust rule yet", len(y), 2 * fc.min_chars)
            return info
        model, theta, oof, info = fit_trust_rule(X, y, ln, stt, S, fc.target_err, fc.risk_z, fc.min_chars, fc.folds, self.cfg.run.seed + 7 * epoch)
        self._frag_set_state({"model": model.state(), "theta": [float(t) for t in theta], "alpha": fc.target_err, "epoch": int(epoch), "n_strata": S,
                              "features": list(FRAG_FEATURES)})
        info["frag_auroc_risk"] = auroc(oof, y.astype(bool))
        info["frag_auroc_conf"] = auroc(-X[:, 0], y.astype(bool))
        info["frag_cal_chars"] = float(len(y))
        info["frag_cal_char_err"] = float(y.mean())
        errs = np.array([info[f"frag_cal_err_all_s{s}"] for s in range(S)])
        if np.isfinite(errs).all() and errs.mean() > 0 and fc.stratum_emphasis > 0:
            self.stratum_w = np.clip((errs / errs.mean()) ** fc.stratum_emphasis, 1.0, 3.0)
        un = self.data.unlabeled_idx
        if len(un):                                            # diagnostics on unlabeled lines (hidden text = diagnostics only)
            rng = np.random.RandomState(self.cfg.run.seed + 7 * epoch)
            sub = np.sort(rng.choice(un, min(fc.diag_lines, len(un)), replace=False))
            u = self._frag_collect(sub, lambda i: ds.oracle[i], 1, self.cfg.run.seed + 5 * epoch)
            if len(u["y"]):
                rk = model.risk(u["X"])
                keep = rk <= np.array([cell_theta(theta, c) for c in u["strat"]])
                info["frag_unl_cov"] = float(keep.mean())
                info["frag_unl_err_sel"] = float(u["y"][keep].mean()) if keep.any() else float("nan")
                info["frag_unl_err_all"] = float(u["y"].mean())
                for s in range(S):
                    ms = (u["strat"] == s) & keep
                    info[f"frag_unl_err_sel_s{s}"] = float(u["y"][ms].mean()) if ms.any() else float("nan")
                    info[f"frag_unl_cov_s{s}"] = float(keep[u["strat"] == s].mean()) if (u["strat"] == s).any() else float("nan")
        LOG.info("[FRAG-RC] epoch %d | %d calibration chars (error %.3f), AUROC risk %.3f vs peak prob %.3f | per cell: coverage %s, error of trusted "
                 "chars %s (target %.3f) | UNLABELED sample (hidden text): trusted %.2f of chars at error %.4f (all chars %.4f) | emphasis %s", epoch, len(y),
                 y.mean(), info["frag_auroc_risk"], info["frag_auroc_conf"], [round(info[f"frag_cal_cov_s{s}"], 2) for s in range(S)],
                 [round(info[f"frag_cal_err_sel_s{s}"], 4) for s in range(S)], fc.target_err, info.get("frag_unl_cov", float("nan")),
                 info.get("frag_unl_err_sel", float("nan")), info.get("frag_unl_err_all", float("nan")), np.round(self.stratum_w, 2).tolist())
        return info

    def _frag_batch(self, t_out: Dict[str, Any], x_s: torch.Tensor, wd: torch.Tensor, rows: torch.Tensor, stratum: torch.Tensor) -> Optional[Dict[str, Any]]:
        """Trust masks of the unlabeled rows of a batch (teacher: weak view from `t_out`, strong view read here, inter-CTC head)."""
        dev, fc = self.device, self.fc
        ridx = torch.nonzero(rows).flatten()
        if ridx.numel() == 0 or self.frag_model is None:
            return None
        teacher = self.ema.module.eval()
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=self.amp_dtype, enabled=self.use_amp):
            s_out = teacher(x_s[ridx], wd[ridx])
        lp_w, aux_w = t_out["log_probs"].float().cpu().numpy(), t_out["aux_log_probs"].float().cpu().numpy()
        lp_s = np.zeros_like(lp_w)
        ls = s_out["log_probs"].float().cpu().numpy()
        lp_s[ridx.cpu().numpy(), :ls.shape[1]] = ls[:, :lp_s.shape[1]]
        lens = t_out["out_lens"].cpu().numpy()
        trust, n_ch, n_tr = frag_trust_batch(lp_w, aux_w, lp_s, lens, stratum.cpu().numpy(), rows.cpu().numpy(), self.frag_model,
                                             self.frag_state["theta"], fc.guard_frames)
        valid_rows = int(sum(lens[b] for b in ridx.cpu().numpy()))
        return {"trust": torch.from_numpy(trust).to(dev), "rows": rows, "tau": fc.sharpen, "anchor": fc.anchor_weight, "n_chars": n_ch,
                "n_trusted_chars": n_tr, "trusted_frames": int(trust.sum()), "frames": valid_rows}

    # ------------------------------------------------------------------ checkpoints
    def _compat(self) -> Dict[str, Any]:
        c = self.cfg
        return json_safe({"units": self.data.codec.units, "model": c.model.__dict__, "img_h": c.data.img_height, "n_clusters": self.data.n_clusters,
                          "labeled_fraction": c.data.labeled_fraction, "calib_fraction": c.data.calib_fraction,
                          "shadow_calibration": c.train.pseudo.shadow_calibration, "seed": c.run.seed, "n_train": len(self.data.train_ds),
                          "batch_size": c.data.batch_size, "frag_enabled": c.train.frag.enabled, "n_strata": self.data.n_strata,
                          "val_gate_fraction": c.data.val_gate_fraction, "val_block": c.data.val_block, "mixline_prob": c.train.mixline_prob,
                          "labeled_lines_sha1": self.data.provenance.get("labeled_lines_sha1"),
                          "calibration_lines_sha1": self.data.provenance.get("calibration_lines_sha1"), "source": c.data.source,
                          "sources": list(c.data.sources), "init_from": c.train.init_from})

    def _data_meta(self) -> Dict[str, Any]:
        """Everything inference needs about the data that produced the weights."""
        return {"chars": self.data.codec.chars, "units": list(self.data.codec.units), "style_mean": self.data.style_mean, "style_std": self.data.style_std,
                "n_clusters": self.data.n_clusters, "n_strata": int(self.data.n_strata),
                "strata_cuts": {str(k): np.asarray(v).tolist() for k, v in self.data.strata_cuts.items()}, "domains": self.data.domain_names}

    def _state(self, completed_epochs: int) -> Dict[str, Any]:
        rng = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "py": random.getstate()}
        if torch.cuda.is_available():
            rng["cuda"] = torch.cuda.get_rng_state_all()
        return {"epoch": completed_epochs, "global_step": self.global_step, "model": self.model.state_dict(), "ema": self.ema.module.state_dict(),
                "opt": self.opt.state_dict(), "scaler": self.scaler.state_dict(), "history": self.history, "best_cer": self.best_cer,
                "best_from": self.best_from, "best_epoch": self.best_epoch, "ref_cer": self.ref_cer, "since_best": self.since_best,
                "stopped_at": self.stopped_at, "cycles": self.cycles, "frag_state": self.frag_state, "stratum_w": np.asarray(self.stratum_w),
                "bank": self.bank.state_dict(), "prev_cov": self.prev_cov, "cov_hist": list(self.cov_hist), "prev_val_preds": self.prev_val_preds,
                "prev_val_cer": self.prev_val_cer, "cfg": self.cfg.to_dict(), "compat": self._compat(), "rng": rng, "leak": self.leak.state_dict(),
                "skipped": self.skipped, "phase_secs": self._phase_secs, **self._data_meta()}

    def save_checkpoint(self, completed_epochs: int) -> None:
        d = self.cfg.ckpt_dir
        st = self._state(completed_epochs)
        atomic_save(st, os.path.join(d, "last.pt"))
        self._last_ckpt = time.time()
        if completed_epochs % self.tc.save_every == 0:
            atomic_save(st, os.path.join(d, f"epoch_{completed_epochs:03d}.pt"))
            old = sorted(glob.glob(os.path.join(d, "epoch_*.pt")))
            for p in old[:-self.tc.keep_last] if self.tc.keep_last > 0 else []:
                os.remove(p)

    def _save_best(self, completed_epochs: int, which: str, cer_val: float) -> None:
        src = self.ema.module if which == "ema" else self.model
        atomic_save({"weights": src.state_dict(), "from": which, "epoch": completed_epochs, "val_cer": cer_val, "cfg": self.cfg.to_dict(),
                     "frag_state": self.frag_state, "compat": self._compat(), **self._data_meta()}, self.best_path)

    def find_latest_checkpoint(self) -> Optional[str]:
        d = self.cfg.ckpt_dir
        cands = [os.path.join(d, "last.pt")] + sorted(glob.glob(os.path.join(d, "epoch_*.pt")), reverse=True)
        return next((p for p in cands if os.path.exists(p)), None)

    def _try_resume(self) -> int:
        """Resume from the newest readable checkpoint.  A checkpoint made with a different model / data / seed / labeled split is REFUSED."""
        if not self.tc.resume:
            return 0
        d, mine = self.cfg.ckpt_dir, self._compat()
        cands = [os.path.join(d, "last.pt")] + sorted(glob.glob(os.path.join(d, "epoch_*.pt")), reverse=True)
        for p in cands:
            if not os.path.exists(p):
                continue
            try:
                st = torch_load(p, map_location="cpu")
            except Exception as e:
                LOG.warning("checkpoint %s is unreadable (%s) - trying an older one", p, e)
                continue
            theirs = st.get("compat")
            diff = [k for k in mine if (theirs or {}).get(k) != mine[k]] if theirs else ["<no compatibility record>"]
            if diff:
                raise RuntimeError(f"checkpoint {p} was produced by a different setup (differs in: {diff}).  Delete {d}, change cfg.run.name or set "
                                   f"cfg.train.resume=False.")
            self.model.load_state_dict(st["model"])
            self.ema.module.load_state_dict(st["ema"])
            self.opt.load_state_dict(st["opt"])
            self.scaler.load_state_dict(st["scaler"])
            self.history, self.best_cer, self.best_from = st["history"], st["best_cer"], st["best_from"]
            self.best_epoch, self.ref_cer = st.get("best_epoch", 0), st.get("ref_cer", float("inf"))
            self.since_best, self.stopped_at = st.get("since_best", 0), st.get("stopped_at")
            self.global_step, self.prev_val_preds = st["global_step"], st.get("prev_val_preds")
            self.prev_val_cer = st.get("prev_val_cer")
            self.bank.load_state_dict(st["bank"])
            self._frag_set_state(st.get("frag_state"))
            if st.get("stratum_w") is not None:
                self.stratum_w = np.asarray(st["stratum_w"], dtype=np.float64)
            self.prev_cov = st.get("prev_cov")
            self.cov_hist = list(st.get("cov_hist", []))
            if self.bank.sel.any():
                self.data.train_ds.set_pseudo(self.bank.train_text, self.bank.weight, self.shadow)
            torch.set_rng_state(st["rng"]["torch"].cpu())
            np.random.set_state(st["rng"]["numpy"])
            random.setstate(st["rng"]["py"])
            if "cuda" in st["rng"] and torch.cuda.is_available():
                try:
                    torch.cuda.set_rng_state_all(st["rng"]["cuda"])
                except Exception:
                    pass
            if st.get("leak") is not None:
                self.leak.load_state_dict(st["leak"])
            self.skipped = int(st.get("skipped", 0))
            self._phase_secs = st.get("phase_secs") or self._phase_secs
            self._resume_partial = st.get("partial")
            if st.get("cycles"):
                self.cycles = st["cycles"]
            old_total = int(self.cycles[-1]["end"])
            if old_total != self.total_steps:
                self._extend_schedule(old_total)
                if self.stopped_at is not None:
                    self.stopped_at, self.since_best = None, 0
            part = self._resume_partial
            LOG.info("RESUMED from %s (completed epochs: %d%s, global step %d, best val_sel CER %.4f at epoch %d, lr now %.2e)", p, st["epoch"],
                     f" + {part['step']}/{len(part['batches'])} steps of epoch {part['epoch'] + 1}" if part else "",
                     self.global_step, self.best_cer, self.best_epoch, self._lr(self.global_step))
            return int(st["epoch"])
        return 0

    # ------------------------------------------------------------------ main loop
    def _record(self, epoch: int, rec: Dict[str, float]) -> None:
        H = self.history
        H["epoch"].append(epoch)
        n = len(H["epoch"])
        for k, v in rec.items():
            if k not in H:
                H[k] = [float("nan")] * (n - 1)
            H[k].append(float(v))
        for k in H:
            if len(H[k]) < n:
                H[k].append(float("nan"))

    @staticmethod
    def _sub_cer(e: Dict[str, Any], ix) -> float:
        return float(np.nansum(e["ed_ps"][ix]) / max(np.nansum(e["ref_ps"][ix]), 1))

    def fit(self) -> None:
        cfg, tc, data = self.cfg, self.tc, self.data
        self.start_epoch = self._try_resume()
        if self.stopped_at is not None:
            LOG.info("Training had stopped early at epoch %d (no val_sel-CER gain for %d epochs); raise cfg.train.epochs to continue.", self.stopped_at, tc.patience)
            return
        if self.start_epoch >= tc.epochs:
            LOG.info("Training already finished (%d epochs).", self.start_epoch)
            return
        LOG.info("Training: %d epochs x %d steps | params %.2fM | device %s | amp=%s (%s) | labeled %d / calibration %d / unlabeled %d | corpora %s",
                 tc.epochs, self.steps_per_epoch, sum(p.numel() for p in self.model.parameters()) / 1e6, self.device, self.use_amp,
                 str(self.amp_dtype).replace("torch.", ""), len(data.labeled_idx), len(data.calib_idx), len(data.unlabeled_idx), data.domain_names)
        n_unl = len(data.unlabeled_idx)
        nan = float("nan")
        self.completed_epochs = self.start_epoch
        old_handler = self._install_signal_handlers()
        try:
            self._fit_epochs(n_unl, nan)
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGTERM, old_handler)
        if not self.interrupted:
            LOG.info("Training done. Best val_sel CER %.4f (%s, epoch %d). Curves: %s", self.best_cer, self.best_from, self.best_epoch, cfg.curve_dir)
            self._final_source_calibration()

    def _stop_between_phases(self, epoch: int, step: int, batches: Optional[List[List[int]]], meter: Optional[Meter], rec: Dict[str, float],
                             phase: str) -> bool:
        """Before a long phase (refresh, end-of-epoch evaluation): stop now with a step checkpoint if it would not finish before the deadline."""
        if self._stop_signal is None and self._phase_fits(phase):
            return False
        if batches is None:
            self.save_checkpoint(epoch)                       # nothing of this epoch done yet: the epoch-boundary checkpoint is the resume point
        else:
            self.save_step_checkpoint(epoch, step, batches, meter, rec)
        LOG.warning("[STOP] the %s of epoch %d would not finish before the deadline: checkpoint written, the next start continues there", phase, epoch + 1)
        self.interrupted = True
        return True

    def _fit_epochs(self, n_unl: int, nan: float) -> None:
        cfg, tc, pc, data = self.cfg, self.tc, self.pc, self.data
        for epoch in range(self.start_epoch, tc.epochs):
            t0 = time.time()
            part, self._resume_partial = self._resume_partial, None
            if part is not None and int(part["epoch"]) == epoch:
                rec = dict(part.get("rec") or {})                # the refresh of this epoch already happened before the stop
            else:
                part, rec = None, {}
                refresh = ((pc.enabled and n_unl > 0 and epoch >= pc.warmup_epochs and (epoch - pc.warmup_epochs) % pc.refresh_every == 0) or
                           (self.fc.enabled and n_unl > 0 and len(data.calib_idx) and epoch >= self.fc.warmup_epochs
                            and (epoch - self.fc.warmup_epochs) % self.fc.refresh_every == 0))
                if refresh and self._stop_between_phases(epoch, 0, None, None, rec, "refresh"):
                    return
                tr0 = time.time()
                if pc.enabled and n_unl > 0 and epoch >= pc.warmup_epochs and (epoch - pc.warmup_epochs) % pc.refresh_every == 0:
                    rec.update(self.refresh_pseudo_labels(epoch))
                if self.fc.enabled and n_unl > 0 and len(data.calib_idx) and epoch >= self.fc.warmup_epochs and (epoch - self.fc.warmup_epochs) % self.fc.refresh_every == 0:
                    rec.update(self.refresh_frag(epoch))
                if refresh:
                    self._phase_secs["refresh"].append(time.time() - tr0)
            tr = self._train_epoch(epoch, rec, part)
            if tr is None:
                return
            if self._stop_between_phases(epoch, len(self._epoch_batches), self._epoch_batches, self._epoch_meter, rec, "eval"):
                return
            te0 = time.time()
            if "total" not in tr:
                LOG.warning("epoch %d: every step was skipped for a non-finite loss (%d skipped so far)", epoch + 1, self.skipped)
            ev = self.evaluate(self.model, data.val_ds)
            ev_ema = self.evaluate(self.ema.module, data.val_ds)
            sel_i, gate_i = data.val_sel_idx, data.val_gate_idx
            sel_st, sel_em = self._sub_cer(ev, sel_i), self._sub_cer(ev_ema, sel_i)
            ev_tr = self.evaluate(self.model, data.train_ds, data.train_eval_idx)
            if self.prev_val_preds is not None:               # inertia metrics (Hsu et al.)
                rec["val_stability"] = float(np.mean([a == b for a, b in zip(ev["preds"], self.prev_val_preds)]))
                rec["val_nfr"] = float(np.nanmean(ev["cer_ps"] > self.prev_val_cer + 1e-9))
            self.prev_val_preds, self.prev_val_cer = list(ev["preds"]), ev["cer_ps"].copy()
            sub = list(range(min(tc.noise_eval_samples, len(data.val_ds))))
            ev_n = self.evaluate(self.model, data.val_ds, sub, noise=True)
            rec["val_noise_cons"] = 1.0 - float(np.mean([cer(ev["preds"][i], ev_n["preds"][i]) for i in sub]))
            rec["val_noise_gap"] = ev_n["cer"] - self._sub_cer(ev, sub)
            self._phase_secs["eval"].append(time.time() - te0)
            g = tr.get
            rec.update({"lr": self._lr(self.global_step), "lr_rel": self._lr(self.global_step) / self.tc.lr, "vq_lambda": self._vq_lambda(self.global_step),
                        "aux_scale": self._aux_scale(self.global_step), "val_sel_cer": sel_st, "val_ema_sel_cer": sel_em,
                        "val_gate_cer": self._sub_cer(ev, gate_i), "val_ema_gate_cer": self._sub_cer(ev_ema, gate_i),
                        "train_frag": g("frag", nan), "train_frag_trust": g("frag_trust", nan), "train_frag_char_trust": g("frag_char_trust", nan),
                        "train_total": g("total", nan), "train_task_total": g("task_total", nan), "train_ctc": g("ctc", nan),
                        "train_style_reg": g("style_reg", nan), "train_ppl_low": g("ppl_low", nan), "train_ppl_high": g("ppl_high", nan),
                        "train_leak": g("leak", nan), "train_cons": g("cons", nan), "train_decor": g("decor", nan),
                        "train_ctc_labeled": g("ctc_labeled", nan), "train_ctc_pseudo": g("ctc_pseudo", nan), "train_mask": g("mask", nan),
                        "grad_norm": g("grad_norm", nan), "val_total": ev["total"], "val_ctc": ev["ctc"], "val_cer": ev["cer"], "val_wer": ev["wer"],
                        "val_style_mae": ev["style_mae"], "val_style_acc": ev.get("style_acc", nan), "val_ema_total": ev_ema["total"],
                        "val_ema_cer": ev_ema["cer"], "val_ema_wer": ev_ema["wer"], "train_eval_total": ev_tr["total"], "train_eval_ctc": ev_tr["ctc"],
                        "train_eval_cer": ev_tr["cer"], "train_eval_wer": ev_tr["wer"], "train_eval_style_mae": ev_tr["style_mae"],
                        "train_eval_style_acc": ev_tr.get("style_acc", nan)})
            if len(data.specs) > 1:                           # per-corpus validation CER (same checkpoint; informational)
                for d_, name in enumerate(data.domain_names):
                    m_ = np.flatnonzero(data.val_ds.domain == d_)
                    if len(m_):
                        rec[f"val_cer_{name}"] = self._sub_cer(ev, m_)
            for j, n_ in enumerate(STYLE_NAMES):
                rec[f"val_style_mae_{n_}"] = float(ev["style_mae_desc"][j])
                if f"leak_r2_{j}" in tr:
                    rec[f"leak_r2_{n_}"] = tr[f"leak_r2_{j}"]
            self._record(epoch, rec)

            which = "ema" if sel_em <= sel_st else "student"
            cur = min(sel_em, sel_st)
            if cur < self.best_cer:
                self.best_cer, self.best_from, self.best_epoch = cur, which, epoch + 1
                self._save_best(epoch + 1, which, cur)
            if cur < self.ref_cer - tc.min_delta:
                self.ref_cer, self.since_best = cur, 0
            else:
                self.since_best += 1
            stop = tc.patience > 0 and (epoch + 1) >= tc.min_epochs and self.since_best >= tc.patience
            if stop:
                self.stopped_at = epoch + 1
            self.save_checkpoint(epoch + 1)
            with open(os.path.join(cfg.curve_dir, "history.json"), "w") as f:
                json.dump(json_safe(self.history), f)
            last = epoch + 1 == tc.epochs or stop
            plot_training_curves(self.history, cfg.curve_dir, individual=last or (epoch + 1) % tc.save_every == 0)
            LOG.info("epoch %3d/%d | %.0fs | train loss %.3f | val CER %.4f (EMA %.4f) [sel %.4f / gate %.4f] WER %.4f | train-subset CER %.4f | style MAE %.3f "
                     "acc %.3f | VQ ppl %.0f/%.0f | lr %.1e | best(val_sel) %.4f (%s, ep %d)%s", epoch + 1, tc.epochs, time.time() - t0, g("total", nan),
                     ev["cer"], ev_ema["cer"], sel_st, rec["val_gate_cer"], ev["wer"], ev_tr["cer"], ev["style_mae"], ev.get("style_acc", nan),
                     g("ppl_low", nan), g("ppl_high", nan), self._lr(self.global_step), self.best_cer, self.best_from, self.best_epoch,
                     f" | skipped steps {self.skipped}" if self.skipped else "")
            self.completed_epochs = epoch + 1
            if stop:
                LOG.info("EARLY STOP at epoch %d: no val_sel-CER gain >= %.4f for %d epochs (best %.4f at epoch %d).", epoch + 1, tc.min_delta, tc.patience,
                         self.best_cer, self.best_epoch)
                break

    def _final_source_calibration(self) -> None:
        """checkpoints/source_calibration.npz from the BEST weights when training used FRAG-RC (the legacy path writes it at every refresh): the
        source side of the zero-shot adapter's transfer-calibrated risk control.  Without it the adapter could not select pseudo-labels."""
        p = os.path.join(self.cfg.ckpt_dir, "source_calibration.npz")
        if os.path.exists(p) or len(self.data.calib_idx) < 20 or not os.path.exists(self.best_path) or not self.pc.save_source_calibration:
            return
        try:
            st = torch_load(self.best_path)
            self.ema.module.load_state_dict(st["weights"])
            rc = self._teacher_read(self.data.calib_idx, desc="source calibration lines")
            ref = [self.data.train_ds.reference(i) for i in rc["idx"]]
            ed = np.array([edit_distance(a, b) for a, b in zip(ref, rc["text"])], np.float64)
            rl = np.array([len(a) for a in ref], np.float64)
            self._save_source_calibration(rc["X"], ed, rl, int(st.get("epoch", -1)))
            LOG.info("source calibration written (%d calibration lines, CER %.4f)", len(ed), ed.sum() / max(rl.sum(), 1))
        except Exception:
            LOG.exception("could not write source_calibration.npz")
