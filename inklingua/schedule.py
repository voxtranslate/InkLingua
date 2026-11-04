"""
Training budget from the data ("epochs set according to the complexity of the task").

The rule is anchored on the one corpus for which there is evidence: on IAM-line (6,482 training lines, batch 16 -> 406 steps per epoch) the
v9 / v10 runs used 100 epochs = 40,600 optimisation steps, and the validation CER plateaued between epoch 30 and 77.  Every other corpus gets

    S(d) = S_IAM * sqrt(N_train(d) / N_IAM) * k_script(d) * k_era(d)          (total optimisation steps, upper bound)
    epochs(d) = clip( ceil(S(d) / steps_per_epoch(d)), 20, 150 )

* sqrt(N): more data needs more steps, but sub-linearly (each epoch over a larger corpus already carries more distinct evidence);
* k_script: alphabet size and cursivity -- Latin / Cyrillic / Greek 1.0, Hebrew 1.2, Arabic and Devanagari 1.3 (cursive, joined forms,
  diacritics), Han 1.8 (thousands of output units);
* k_era: 1.15 for historical archival hands (abbreviations, unstable letter forms), 1.0 for modern writing;
* for a corpus mix, the factors are averaged with the share of training lines of each corpus.

Every epoch-denominated quantity scales with the epoch budget E, with the fractions that the IAM defaults already had (so IAM reproduces
them exactly): patience 0.25 E (>= 10), minimum epochs 0.30 E, FRAG-RC warm-up 0.20 E and refresh 0.04 E (>= 2), legacy pseudo-label warm-up
0.25 E and refresh 0.03 E (>= 2), auxiliary-loss start 0.03 E and ramp 0.05 E, quantisation start 0.03 E and anneal 0.04 E, checkpoint
interval 0.05 E; lr warm-up 1.5 % of S (100..1500 steps).  Early stopping on val_sel makes E an upper bound, not a target.  The budget is
identical for every variant of a corpus (labeled_only, full, ablations ...): the comparison is at equal optimisation budget.
"""
from __future__ import annotations

import math
from typing import Dict, Sequence

REF_LINES = 6482
REF_STEPS = 40600
MIN_EPOCHS, MAX_EPOCHS = 20, 150
SCRIPT_FACTOR = {"Latin": 1.0, "Cyrillic": 1.0, "Greek": 1.0, "Hebrew": 1.2, "Arabic": 1.3, "Devanagari": 1.3, "Han": 1.8}
ERA_FACTOR = {"modern": 1.0, "historical": 1.15}


def complexity_factor(specs: Sequence, lines_per_spec: Sequence[int]) -> float:
    tot = max(float(sum(lines_per_spec)), 1.0)
    return float(sum((n / tot) * SCRIPT_FACTOR.get(s.script, 1.0) * ERA_FACTOR.get(getattr(s, "era", "modern"), 1.0)
                     for s, n in zip(specs, lines_per_spec)))


def budget(n_train: int, factor: float, batch_size: int) -> Dict[str, int]:
    """The schedule for a training set of n_train lines (all lines of the epoch pool: labeled, calibration and unlabeled)."""
    steps_total = REF_STEPS * math.sqrt(max(n_train, 1) / REF_LINES) * factor
    spe = int(math.ceil(max(n_train, 1) / batch_size))
    E = int(min(MAX_EPOCHS, max(MIN_EPOCHS, math.ceil(steps_total / spe))))
    r = lambda f, lo=1: max(lo, int(round(f * E)))  # noqa: E731
    return {"epochs": E, "steps_per_epoch": spe, "planned_steps": E * spe, "budget_steps": int(round(steps_total)),
            "patience": r(0.25, 10), "min_epochs": r(0.30), "frag_warmup": r(0.20), "frag_refresh": r(0.04, 2),
            "pl_warmup": r(0.25), "pl_refresh": r(0.03, 2), "aux_start": r(0.03), "aux_ramp": r(0.05), "vq_start": r(0.03),
            "vq_anneal": r(0.04), "save_every": r(0.05), "warmup_steps": int(min(1500, max(100, round(0.015 * steps_total))))}


def apply_budget(cfg, b: Dict[str, int]) -> None:
    tc = cfg.train
    vq_off = tc.vq_start_epoch >= 10 ** 5          # the no_vq / ctc_only variants keep quantisation off: never re-enable it
    tc.epochs, tc.patience, tc.min_epochs = b["epochs"], b["patience"], b["min_epochs"]
    tc.frag.warmup_epochs, tc.frag.refresh_every = b["frag_warmup"], b["frag_refresh"]
    tc.pseudo.warmup_epochs, tc.pseudo.refresh_every = b["pl_warmup"], b["pl_refresh"]
    tc.aux_start_epoch, tc.aux_ramp_epochs, tc.vq_start_epoch, tc.vq_anneal_epochs = b["aux_start"], b["aux_ramp"], b["vq_start"], b["vq_anneal"]
    tc.save_every, tc.warmup_steps = b["save_every"], b["warmup_steps"]
    if vq_off:
        tc.vq_start_epoch = 10 ** 6
