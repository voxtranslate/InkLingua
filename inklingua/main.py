"""Entry point.  `main(cfg)` dispatches on cfg.run.mode; `python inklingua.py` runs the default IAM experiment."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence

from .config import Config
from .data import HandwritingDataModule
from .llm import LLMEngine
from .model import InkLinguaNet
from .protocol import write_protocol
from .style import NUM_DESC
from .utils import LOG, repository_selftests, resolve_device, resolve_llm_device, set_seed, setup_logging


def run_single(cfg: Config, base_units: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """Train (resumable) and / or evaluate one configuration."""
    from . import __version__
    from .inferencer import Inferencer
    from .trainer import Trainer
    cfg.make_dirs()
    setup_logging(cfg)
    set_seed(cfg.run.seed)
    cfg.save()
    device = resolve_device(cfg.run.device)
    LOG.info("InkLingua v%s | run dir: %s | device: %s | seed %d | corpora %s", __version__, cfg.run_dir, device, cfg.run.seed, cfg.data.sources)
    if cfg.run.self_test:
        repository_selftests().run_self_tests()
        set_seed(cfg.run.seed)
    data = HandwritingDataModule(cfg, base_units=base_units).prepare()
    if not cfg.llm.source_language:
        cfg.llm.source_language = data.primary_language
    if cfg.train.auto_schedule and cfg.run.do_train:
        import json as _json
        from .schedule import apply_budget, budget, complexity_factor
        lines = [int((data.train_ds.domain == d).sum()) for d in range(len(data.specs))]
        b = budget(len(data.train_ds), complexity_factor(data.specs, lines), cfg.data.batch_size)
        b["complexity_factor"] = complexity_factor(data.specs, lines)
        apply_budget(cfg, b)
        cfg.save()
        with open(os.path.join(cfg.run_dir, "schedule.json"), "w") as f:
            _json.dump(b, f, indent=2)
        LOG.info("[SCHEDULE] %d training lines, complexity %.2f -> %d epochs (%d steps/epoch, budget %d steps), patience %d, FRAG warm-up %d",
                 len(data.train_ds), b["complexity_factor"], b["epochs"], b["steps_per_epoch"], b["budget_steps"], b["patience"], b["frag_warmup"])
    write_protocol(cfg, data)
    llm_device = resolve_llm_device(cfg.llm.device, device)
    LOG.info("recogniser on %s, LLM on %s", device, llm_device)
    llm = LLMEngine(cfg.llm, llm_device)
    model = InkLinguaNet(cfg.model, data.codec.num_classes, NUM_DESC, data.n_clusters, cfg.data.img_height)
    best, metrics = None, None
    if cfg.run.do_train:
        trainer = Trainer(cfg, data, model, llm, device)
        trainer.fit()
        if trainer.interrupted:
            LOG.warning("training interrupted by the time budget at epoch %d: resumable, evaluation postponed", trainer.completed_epochs)
            return {"interrupted": True, "completed_epochs": trainer.completed_epochs}
        best = trainer.best_path if os.path.exists(trainer.best_path) else None
    if cfg.run.do_infer:
        infer = Inferencer(cfg, data, llm, device)
        infer.load(best)
        metrics = infer.run()
        LOG.info("All done. Open %s", os.path.join(cfg.infer_dir, "12_report", "index.html"))
    return metrics


def main(cfg: Optional[Config] = None) -> Optional[Dict[str, Any]]:
    cfg = cfg or Config()
    mode = cfg.run.mode
    if mode == "suite":
        from .suite import run_suite
        return run_suite(cfg)
    if mode == "suite_summary":
        from .suite import run_suite
        return run_suite(cfg, train=False)
    if mode == "finetune":
        from .adapt import run_finetune
        return run_finetune(cfg)
    if mode == "adapt":
        from .adapt import run_adapt
        return run_adapt(cfg)
    if mode == "adapt_curve":
        from .adapt import run_adapt_curve
        return run_adapt_curve(cfg)
    if mode == "source_calib":
        from .adapt import run_source_calib
        run_source_calib(cfg)
        return None
    if mode == "report":
        from .report import build_run_report
        cfg.make_dirs()
        setup_logging(cfg)
        build_run_report(cfg.infer_dir, cfg.run_dir, curve_dir=cfg.curve_dir)
        return None
    if mode == "smoke":
        return repository_selftests().run_smoke()
    if mode != "single":
        raise ValueError(f"unknown cfg.run.mode '{mode}'")
    return run_single(cfg)
