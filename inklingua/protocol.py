"""[PROTOCOL] which split is used for what, and a log of every look at an evaluation split."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict

import numpy as np

from .utils import json_safe


def write_protocol(cfg, data) -> Dict[str, Any]:
    """protocol/protocol.json.  The TEST split decides nothing and is looked at once per checkpoint (every look is logged); the
    VALIDATION split is cut into two disjoint halves of whole form blocks -- `val_sel` picks the checkpoint / stops training, `val_gate`
    is the only data on which any post-hoc option, temperature, threshold or LLM weight is tuned; hidden train labels are read only by
    diagnostics."""
    d = os.path.join(cfg.run_dir, "protocol")
    os.makedirs(d, exist_ok=True)
    sha = lambda a: hashlib.sha1(np.asarray(a, np.int64).tobytes()).hexdigest()  # noqa: E731
    p = {
        "roles": {
            "train: labeled lines": "supervised CTC loss (text used)",
            "train: calibration lines (labeled, text held out)": "text used ONLY to measure the character error of the trust rule (FRAG-RC); "
                                                                 "their IMAGES join the unlabeled pool (shadow calibration)",
            "train: unlabeled lines": "images only; their hidden text is read by DIAGNOSTICS only (never by a loss, a selection or a threshold)",
            "validation: val_sel": "checkpoint selection (EMA vs student, best epoch) and early stopping",
            "validation: val_gate": "every post-hoc decision: view ensemble, style AdaBN, writer self-distillation, space bonus, LLM weights, "
                                    "temperature, [?] marks.  Never used to select the checkpoint.",
            "test": "reported once per checkpoint; nothing is tuned, selected or calibrated on it"},
        "val_sel_lines": int(len(data.val_sel_idx)), "val_gate_lines": int(len(data.val_gate_idx)),
        "val_sel_sha1": sha(data.val_sel_idx), "val_gate_sha1": sha(data.val_gate_idx),
        "val_split_enabled": bool(0.0 < cfg.data.val_gate_fraction < 1.0), "val_block_lines": int(cfg.data.val_block),
        "labeled_fraction": cfg.data.labeled_fraction, "calibration_fraction": cfg.data.calib_fraction,
        "frag_rc": bool(cfg.train.frag.enabled), "legacy_pseudo_labels": bool(cfg.train.pseudo.enabled),
        "corpora": data.domain_names, "n_strata": int(data.n_strata),
        "transductive_options": ["style AdaBN (reads unlabeled evaluation lines)", "writer self-distillation WSD (reads unlabeled evaluation lines)"],
        "external_knowledge": "the LLM was trained on web text that may contain the corpus a benchmark is drawn from: report LLM-rescored numbers "
                              "as 'with an external language model'",
        "seed": cfg.run.seed, "provenance": json_safe(data.provenance)}
    with open(os.path.join(d, "protocol.json"), "w", encoding="utf-8") as f:
        json.dump(json_safe(p), f, indent=2, ensure_ascii=False)
    return p


def log_test_look(cfg, ckpt: Dict[str, Any], split: str) -> int:
    """Append one line to protocol/test_looks.jsonl for every evaluation run; returns how many looks at `split` there have been."""
    d = os.path.join(cfg.run_dir, "protocol")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "test_looks.jsonl")
    h = None
    try:
        with open(ckpt.get("path", ""), "rb") as f:
            h = hashlib.sha1(f.read(1 << 22)).hexdigest()
    except Exception:
        pass
    with open(path, "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "split": split, "checkpoint": ckpt.get("path"),
                            "checkpoint_sha1_first4MB": h, "epoch": ckpt.get("epoch")}) + "\n")
    with open(path) as f:
        return sum(1 for line in f if json.loads(line).get("split") == split)
