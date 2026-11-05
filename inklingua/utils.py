"""Generic utilities: seeding, strict JSON, logging, devices, atomic checkpoints, running means."""
from __future__ import annotations

import logging
import math
import os
import warnings
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
    quiet_third_party()


_NOISY_LOGGERS = ("httpx", "httpx2", "httpcore", "httpcore2", "urllib3", "requests", "filelock", "fsspec", "huggingface_hub", "datasets", "transformers", "accelerate",
                  "PIL", "matplotlib", "h5py", "numexpr")
_HUB_READY = {"done": False}


def quiet_third_party() -> None:
    """Keep the experiment log about the experiment: no per-request HTTP lines (httpx logs every HEAD / GET at INFO, and a notebook
    that configured the root logger at INFO printed them all), no download progress bars, no telemetry, and the Hugging Face token
    resolved once (see `resolve_hf_token`) instead of a warning repeated by every library that talks to the Hub."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:
        pass
    try:
        import datasets as _hfd
        _hfd.utils.logging.set_verbosity_warning()
        _hfd.utils.logging.disable_progress_bar()
    except Exception:
        pass
    for name in _NOISY_LOGGERS:                                         # after the imports: huggingface_hub resets its level on import
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)        # its token hint is logged once below, as an INFO line
    if not _HUB_READY["done"]:
        _HUB_READY["done"] = True
        how = resolve_hf_token()
        LOG.info("Hugging Face Hub: %s", how)


def resolve_hf_token() -> str:
    """HF_TOKEN from the environment, else from a Kaggle secret named HF_TOKEN (Add-ons -> Secrets), exported for every library and
    subprocess.  Without a token the public corpora still download (lower rate limits)."""
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return "authenticated (HF_TOKEN from the environment)"
    try:
        from kaggle_secrets import UserSecretsClient          # only exists on Kaggle
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            return "authenticated (Kaggle secret HF_TOKEN)"
    except Exception:
        pass
    return "unauthenticated (public datasets only; set HF_TOKEN, or add a Kaggle secret named HF_TOKEN, for higher rate limits)"


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


def _numpy_safe_globals() -> list:
    """The numpy objects our checkpoints contain (random states, small arrays): allowlisted for torch.load(weights_only=True)."""
    import importlib
    g: list = [np.ndarray, np.dtype]
    for mod in ("numpy._core.multiarray", "numpy.core.multiarray"):
        try:
            m = importlib.import_module(mod)
            g += [m._reconstruct, m.scalar]
        except Exception:
            pass
    try:                                                        # numpy >= 1.25: one class per dtype (numpy.dtypes.Int8DType, ...)
        import numpy.dtypes as npd
        g += [getattr(npd, n) for n in dir(npd) if n.endswith("DType") and isinstance(getattr(npd, n), type)]
    except Exception:
        pass
    return g


def torch_load(path: str, map_location="cpu", trusted: bool = False) -> Any:
    """torch.load with weights_only=True (no arbitrary code from a pickle) and the numpy types of our checkpoints allowlisted.  Unrestricted
    unpickling is used only when the caller says the file is trusted (or INKLINGUA_TRUSTED_CHECKPOINTS=1), e.g. a checkpoint of an older version."""
    try:
        with torch.serialization.safe_globals(_numpy_safe_globals()):
            return torch.load(path, map_location=map_location, weights_only=True)
    except Exception as e:
        if not (trusted or os.environ.get("INKLINGUA_TRUSTED_CHECKPOINTS") == "1"):
            raise RuntimeError(f"{path} cannot be loaded safely (weights_only=True): {str(e).splitlines()[0][:200]}.  If you trust this file, "
                               "set INKLINGUA_TRUSTED_CHECKPOINTS=1.") from e
        LOG.warning("loading %s with weights_only=False (trusted file)", path)
        return torch.load(path, map_location=map_location, weights_only=False)


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

    def state(self) -> Dict[str, Dict[str, float]]:
        return {"sum": dict(self.sum), "cnt": dict(self.cnt)}

    @classmethod
    def from_state(cls, st: Optional[Dict[str, Dict[str, float]]]) -> "Meter":
        m = cls()
        if st:
            m.sum.update(st["sum"])
            m.cnt.update(st["cnt"])
        return m
