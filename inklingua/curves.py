"""Training curves (train vs valid, FRAG-RC trust, legacy pseudo-labels, inertia, leakage, VQ)."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .style import STYLE_NAMES


def _plot_series(ax, H: Dict[str, List[float]], series: Sequence[Tuple[str, str]], title: str, ylabel: str = "", logy: bool = False) -> None:
    ep = np.asarray(H.get("epoch", []), float) + 1
    drawn = False
    for key, label in series:
        if key in H:
            y = np.asarray(H[key], float)
            m = np.isfinite(y)
            if m.any():
                n = min(len(ep), len(y))
                m = m[:n]
                ax.plot(ep[:n][m], y[:n][m], marker="o", ms=3, lw=1.6, label=label)
                drawn = True
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if logy and drawn:
        ax.set_yscale("log")
    if drawn:
        ax.legend(fontsize=7)


def plot_training_curves(H: Dict[str, List[float]], out_dir: str, individual: bool = True) -> None:
    os.makedirs(out_dir, exist_ok=True)
    leak_series = [(f"leak_r2_{n}", f"R2 of the optimal probe: {n}") for n in STYLE_NAMES]
    cells = sorted({k.split("frag_cal_cov_s")[1] for k in H if k.startswith("frag_cal_cov_s")}, key=lambda s: int(s))
    panels = [
        ("loss", "Loss: train vs valid", "loss",
         [("train_task_total", "train task loss (running, augmented)"), ("train_total", "train full objective (all terms)"),
          ("train_eval_total", "train subset (eval mode)"), ("val_total", "valid"), ("val_ema_total", "valid (EMA)")], False),
        ("ctc", "CTC loss (per char): train vs valid", "CTC",
         [("train_ctc", "train running"), ("train_eval_ctc", "train subset (eval)"), ("val_ctc", "valid")], True),
        ("cer", "CER: train vs valid (val_sel selects the checkpoint, val_gate never does)", "CER",
         [("train_eval_cer", "train subset"), ("val_sel_cer", "val_sel (student)"), ("val_ema_sel_cer", "val_sel (EMA teacher)"),
          ("val_gate_cer", "val_gate (student)"), ("val_cer", "all validation (student)")], True),
        ("wer", "WER: train vs valid", "WER",
         [("train_eval_wer", "train subset"), ("val_wer", "valid (student)"), ("val_ema_wer", "valid (EMA teacher)")], True),
        ("style", "Connected task - style MAE (std. units, measured entries only)", "MAE / loss",
         [("train_eval_style_mae", "MAE train subset"), ("val_style_mae", "MAE valid"), ("train_style_reg", "reg loss (running, SES targets)")], False),
        ("style_acc", "Connected task - style-cluster accuracy", "accuracy", [("train_eval_style_acc", "train subset"), ("val_style_acc", "valid")], False),
        ("frag_trust", "[FRAG-RC] trusted share in the training batches + fragment loss", "value",
         [("train_frag_char_trust", "trusted share of emitted characters"), ("train_frag_trust", "trusted share of frames"),
          ("train_frag", "fragment distillation loss")], False),
        ("frag_cov", "[FRAG-RC] coverage of the trust rule per cell (calibration chars) and on unlabeled lines", "coverage",
         [(f"frag_cal_cov_s{s}", f"calibration, cell {s}") for s in cells] + [("frag_unl_cov", "unlabeled sample (all cells)")], False),
        ("frag_err", "[FRAG-RC] realised error of the TRUSTED characters (unlabeled sample, hidden text; diagnostic)", "error rate",
         [(f"frag_unl_err_sel_s{s}", f"trusted chars, cell {s}") for s in cells] + [("frag_unl_err_sel", "trusted chars, all cells"),
                                                                                    ("frag_unl_err_all", "ALL chars of the sample"),
                                                                                    ("frag_cal_char_err", "all calibration chars")], False),
        ("pseudo", "Legacy pseudo-labels (checked against hidden GT, diagnostics only)", "value",
         [("pl_coverage", "coverage of unlabeled"), ("pl_sel_exact", "selected: exact-match rate"), ("pl_sel_cer", "selected: CER vs hidden GT"),
          ("pl_all_cer", "all unlabeled: CER"), ("pl_stable_frac", "stable fraction")], False),
        ("pl_rank", "Does the reliability score rank the errors? (AUROC)", "AUROC",
         [("pl_auroc_error_detection", "learned line risk (unlabeled)"), ("pl_auroc_seq_conf", "CTC sequence confidence (unlabeled)"),
          ("rc_calib_auroc_error_detection", "learned line risk (calibration)"), ("frag_auroc_risk", "FRAG-RC character risk (calibration)"),
          ("frag_auroc_conf", "character peak probability (calibration)")], False),
        ("cov", "Legacy pseudo-labels: accepted coverage", "coverage",
         [("rc_calib_cov_raw", "calibration lines: raw rule"), ("rc_calib_coverage", "calibration lines: after guard"),
          ("shadow_coverage", "shadow lines"), ("pl_coverage", "unlabeled lines")], False),
        ("inertia", "Model inertia (Hsu et al.)", "value",
         [("val_stability", "prediction stability vs prev. epoch"), ("val_nfr", "negative-flip rate"),
          ("val_noise_cons", "noise consistency (1-CER clean/noisy)"), ("val_noise_gap", "noise CER gap")], False),
        ("vq", "VQ codebook perplexity", "perplexity", [("train_ppl_low", "stroke codebook"), ("train_ppl_high", "component codebook")], False),
        ("lr", "Learning rate / schedules", "value", [("lr_rel", "lr / peak lr"), ("vq_lambda", "quantisation strength"), ("aux_scale", "aux-task curriculum")], False),
        ("leak", "Content / style separation and consistency losses", "loss",
         [("train_leak", "[CFL] leakage penalty"), ("train_cons", "teacher-student frame consistency"), ("train_decor", "cross-covariance")], False),
        ("leak_r2", "[CFL] style readable from the content stream (R2 of the optimal linear probe)", "R2", leak_series, False),
    ]
    ncol = 3
    nrow = int(math.ceil(len(panels) / ncol))
    fig_all, axes = plt.subplots(nrow, ncol, figsize=(17, 4.0 * nrow))
    for (name, title, yl, series, logy), ax in zip(panels, axes.ravel()):
        _plot_series(ax, H, series, title, yl, logy)
        if individual:
            f, a = plt.subplots(figsize=(6.5, 4.2))
            _plot_series(a, H, series, title, yl, logy)
            f.tight_layout()
            f.savefig(os.path.join(out_dir, f"curve_{name}.png"), dpi=130)
            plt.close(f)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(out_dir, "curves_all.png"), dpi=110)
    plt.close(fig_all)
