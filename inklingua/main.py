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
from .utils import LOG, resolve_device, resolve_llm_device, set_seed, setup_logging


def run_single(cfg: Config, base_units: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """Train (resumable) and / or evaluate one configuration."""
    from . import __version__
    from .inferencer import Inferencer
    from .selftest import run_self_tests
    from .trainer import Trainer
    cfg.make_dirs()
    setup_logging(cfg)
    set_seed(cfg.run.seed)
    cfg.save()
    device = resolve_device(cfg.run.device)
    LOG.info("InkLingua v%s | run dir: %s | device: %s | seed %d | corpora %s", __version__, cfg.run_dir, device, cfg.run.seed, cfg.data.sources)
    if cfg.run.self_test:
        run_self_tests()
        set_seed(cfg.run.seed)
    data = HandwritingDataModule(cfg, base_units=base_units).prepare()
    if not cfg.llm.source_language:
        cfg.llm.source_language = data.primary_language
    write_protocol(cfg, data)
    llm_device = resolve_llm_device(cfg.llm.device, device)
    LOG.info("recogniser on %s, LLM on %s", device, llm_device)
    llm = LLMEngine(cfg.llm, llm_device)
    model = InkLinguaNet(cfg.model, data.codec.num_classes, NUM_DESC, data.n_clusters, cfg.data.img_height)
    best, metrics = None, None
    if cfg.run.do_train:
        trainer = Trainer(cfg, data, model, llm, device)
        trainer.fit()
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
        from .selftest import run_smoke
        return run_smoke()
    if mode != "single":
        raise ValueError(f"unknown cfg.run.mode '{mode}'")
    return run_single(cfg)
