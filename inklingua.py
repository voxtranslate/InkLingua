#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
InkLingua -- style-disentangled, uncertainty-aware handwriting-to-translation with fragment-level risk-controlled self-distillation.

This file is the entry point; the implementation lives in the `inklingua/` package (one module per question, see inklingua/__init__.py and
README.md).  There is no argparse: every option is a dataclass default in inklingua/config.py, or is set in a notebook cell:

    from inklingua import Config, main
    main()                                        # the default IAM-line experiment
    main(Config.smoke())                          # a synthetic plumbing test
    cfg = Config(); cfg.run.mode = "suite"; main(cfg)                    # baselines + ablations x seeds -> suite_summary.md
    main(Config.multi(("iam", "rimes", "khatt")))                        # several corpora of the registry at once
    cfg = Config(); cfg.run.mode = "finetune"; cfg.train.init_from = "<run>/checkpoints/best.pt"; cfg.data.sources = ("rimes",); main(cfg)
"""
from inklingua import Config, main  # noqa: F401

if __name__ == "__main__":
    main()
