"""Generic utilities: seeding, strict JSON, logging, devices, atomic checkpoints, running means."""
from __future__ import annotations

import logging
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, Optional

import numpy as np
import torch

LOG = logging.getLogger("inklingua")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False        # variable line widths: autotuning would only add noise


def json_safe(o: Any) -> Any:
    """Recursively convert numpy types and replace NaN / inf by null (strict JSON)."""
    if isinstance(o, dict):
        return {str(k): json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_safe(v) for v in o]
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        o = o.item()
    if isinstance(o, np.ndarray):
        return json_safe(o.tolist())
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    return o


_LOG_STACK = []


def setup_logging(cfg, replace: bool = True) -> None:
    """Log to stdout and to <run_dir>/run.log.  The v10 self-tests called main() for a synthetic run, which replaced the
    handlers of the REAL run with handlers pointing into a temporary directory that was then deleted; here the previous handlers
    are remembered (`push`) and `restore_logging()` puts them back."""
    os.makedirs(cfg.run_dir, exist_ok=True)
    if replace:
        _LOG_STACK.append((list(LOG.handlers), LOG.level))
    LOG.setLevel(getattr(logging, cfg.run.log_level.upper(), logging.INFO))
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    fh = logging.FileHandler(os.path.join(cfg.run_dir, "run.log"))
    fh.setFormatter(fmt)
    LOG.addHandler(sh)
    LOG.addHandler(fh)
    LOG.propagate = False


def restore_logging() -> None:
    """Undo the last setup_logging (used after a nested run such as the smoke test)."""
    if not _LOG_STACK:
        return
    handlers, level = _LOG_STACK.pop()
    for h in list(LOG.handlers):
        LOG.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    for h in handlers:
        LOG.addHandler(h)
    LOG.setLevel(level)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_llm_device(name: str, main_device: torch.device) -> torch.device:
    """Device of the LLM: "auto" puts it on the second GPU when two are available (the recogniser keeps the first), else next to the recogniser."""
    if name == "auto":
        if main_device.type == "cuda" and torch.cuda.device_count() > 1:
            other = 1 if (main_device.index or 0) == 0 else 0
            return torch.device(f"cuda:{other}")
        return main_device
    return torch.device(name)


def atomic_save(obj: Any, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def torch_load(path: str, map_location="cpu") -> Any:
    """Our own checkpoints contain numpy arrays / lists -> weights_only=False is intended."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # very old torch
        return torch.load(path, map_location=map_location)


def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}


class Meter:
    """Weighted running means for a dict of scalars."""

    def __init__(self):
        self.sum: Dict[str, float] = defaultdict(float)
        self.cnt: Dict[str, float] = defaultdict(float)

    def add(self, key: str, value: Optional[float], n: float = 1.0) -> None:
        if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
            return
        self.sum[key] += float(value) * n
        self.cnt[key] += n

    def mean(self, key: str, default: float = float("nan")) -> float:
        return self.sum[key] / self.cnt[key] if self.cnt[key] > 0 else default

    def means(self) -> Dict[str, float]:
        return {k: self.mean(k) for k in self.sum}
