#!/usr/bin/env python3
"""
Build  notebooks/InkLingua_Kaggle_T4x2.ipynb : ONE code cell, ONE entry point, that runs the complete experiment plan of the article
(inklingua/experiments.py) on real data on a Kaggle "GPU T4 x2" session, and continues where it stopped in the next session.

The cell embeds the `inklingua/` package as a zip archive (base64, with its sha256), so the notebook needs no repository access; the readable
source is the repository, and the archive's sha256 is recorded in the outputs (paper/REPRODUCIBILITY.md).  Re-run after changing the package:

    python tools/build_kaggle_notebook.py
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "inklingua")
OUT = os.path.join(ROOT, "notebooks", "InkLingua_Kaggle_T4x2.ipynb")

HEADER = r'''# ======================================================================================================================
#  InkLingua - the complete experiment plan of the article.  ONE cell, ONE entry point (main() at the bottom).
#  Kaggle: Settings -> Accelerator "GPU T4 x2", Internet "On".  Then "Save Version" -> "Save & Run All (Commit)".
#  Optional: Add-ons -> Secrets -> a secret named HF_TOKEN (a Hugging Face read token): faster downloads, higher rate limits.
# ======================================================================================================================
#
#  WHAT RUNS (real data only: the 12 corpora of the registry, pinned Hub commits; no synthetic data, no dummy model, no smoke test)
#    For every corpus, one after the other (IAM first, then from the smallest corpus to the largest), and for every seed:
#      core      labeled_only | ctc_only | legacy_pl (v9 line-level pseudo-labels) | full (InkLingua) | full_supervised (100 % labels)
#      ablations 14 training ablations of InkLingua (FRAG-RC, strata, sharpening, MixLine, style task, SES, VQ, FiLM, leakage penalty,
#                frame consistency, masked codes, inter-CTC, layer drop + CTC label smoothing, horizontal stretch)
#      budget    labeled_only and full at 5 % and 10 % labels (30 % and 100 % come from the core runs)
#    Then one model trained on all corpora at once (joint), and the zero-shot adaptation of the IAM model to French and Catalan.
#    Inference-only options (writer self-distillation, view ensemble, AdaBN, word-gap bonus, LLM rescoring) are measured inside every run
#    and gated on val_gate; they are not retrained.  The epoch budget of each corpus follows its size, script and era
#    (inklingua/schedule.py; IAM reproduces the published defaults: 100 epochs).
#
#  HOW IT RUNS
#    Two runs train in parallel, one per T4; each run is its own process (GPU memory is freed between runs).  Every training writes a
#    step checkpoint at least every 10 minutes and when it is stopped, and resumes EXACTLY at that step (same batches, same random state).
#    Each corpus is downloaded once (pinned commit, offline afterwards) and decoded once in parallel into a line cache shared by all its
#    runs; the caches of unfinished corpora are saved with the output and restored next session (no new download / decoding).
#    The session stops cleanly before Kaggle's 12-hour limit; everything finished is in /kaggle/working/inklingua_runs:
#      paper/RESULTS.md (+ tables/*.tex|md|csv, figures/, REPRODUCIBILITY.md, STATUS.md) - rebuilt after every finished run
#      <corpus>/<variant>_lfXXX_sNNNN/  - each run: config, schedule, curves, checkpoints, protocol log, inference/test/00..12 folders
#
#  TO CONTINUE IN THE NEXT SESSION
#    "Add Input" -> the Output of the previous version of THIS notebook, put its path in RESUME_FROM below, Save & Run All again.
#    Finished runs are skipped, interrupted runs continue from the step where they stopped.  The full plan is long (see the estimate printed at the
#    start; it is replaced by measured rates as soon as runs finish): ~70 training runs per corpus.  PLAN_OVERRIDES can narrow it.
# ======================================================================================================================

RESUME_FROM = ""                 # e.g. "/kaggle/input/inklingua-experiment-plan" (the previous version's output); "" on the first session
PLAN_OVERRIDES = {}              # optional, e.g. {"seeds": (1337,)} or {"corpora": ("iam", "khatt")} or {"order": "tier"}; {} = the full plan
'''

BODY = r'''
# ----------------------------------------------------------------------------------------------------- nothing to edit below this line
import base64, hashlib, io, os, subprocess, sys, time, urllib.request, zipfile

SESSION_START = time.time()
WORK_DIR = "/kaggle/working/inklingua_runs" if os.path.isdir("/kaggle/working") else os.path.abspath("inklingua_runs")
CACHE_DIR = "/tmp/inklingua_cache"


def _log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _ensure_packages():
    import importlib.util
    need = {"editdistance": "editdistance", "cv2": "opencv-python-headless", "pyarrow": "pyarrow", "huggingface_hub": "huggingface_hub",
            "transformers": "transformers", "accelerate": "accelerate", "matplotlib": "matplotlib", "tqdm": "tqdm", "datasets": "datasets",
            "sentencepiece": "sentencepiece"}
    import torch
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
        raise RuntimeError(f"torch {torch.__version__} is too old (>= 2.5 needed): choose a recent Kaggle environment")
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        _log(f"installing {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _preflight():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError("no GPU: set Settings -> Accelerator to 'GPU T4 x2'")
    _log("GPUs: " + " | ".join(out.splitlines()))
    try:
        urllib.request.urlopen("https://huggingface.co/api/datasets/Teklia/IAM-line", timeout=20).read(64)
    except Exception as e:
        raise RuntimeError(f"the Hugging Face Hub is not reachable ({e}): set Settings -> Internet to 'On'") from e
    free = int(subprocess.run(["df", "-B1", "--output=avail", "/tmp"], capture_output=True, text=True).stdout.split()[-1])
    _log(f"free disk for downloads and caches (/tmp): {free / 2**30:.0f} GB (largest corpus: ~5.3 GB of parquet)")


def _install_code():
    blob = base64.b64decode(_CODE_ZIP_B64)
    sha = hashlib.sha256(blob).hexdigest()
    if sha != _CODE_SHA256:
        raise RuntimeError("the embedded code archive is corrupted (sha256 mismatch)")
    code_root = os.path.join(WORK_DIR, "_plan", "code")
    if os.path.isdir(os.path.join(code_root, "inklingua")):
        import shutil
        shutil.rmtree(os.path.join(code_root, "inklingua"))
    os.makedirs(code_root, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        z.extractall(code_root)
    with open(os.path.join(WORK_DIR, "_plan", "code_sha256.txt"), "w") as f:
        f.write(sha + "\n")
    sys.path.insert(0, code_root)
    _log(f"InkLingua code installed in {code_root} (archive sha256 {sha[:16]}...)")
    return code_root


def main():
    _ensure_packages()
    _preflight()
    os.makedirs(os.path.join(WORK_DIR, "_plan"), exist_ok=True)
    code_root = _install_code()
    from inklingua import __version__
    from inklingua.experiments import PlanConfig, run_plan
    from inklingua.utils import quiet_third_party, resolve_hf_token
    quiet_third_party()                                  # no per-request HTTP lines, no progress bars
    _log(f"Hugging Face Hub: {resolve_hf_token()}")      # HF_TOKEN (env or Kaggle secret) is exported to every run process
    used_h = (time.time() - SESSION_START) / 3600.0
    plan = PlanConfig(work_dir=WORK_DIR, cache_dir=CACHE_DIR, resume_from=RESUME_FROM, **PLAN_OVERRIDES)
    plan.session_hours = max(0.5, plan.session_hours - used_h)
    _log(f"InkLingua v{__version__} | experiment plan | resume_from={RESUME_FROM or '-'} | overrides={PLAN_OVERRIDES or '-'}")
    res = run_plan(plan, code_root=code_root, log=_log)
    results = os.path.join(WORK_DIR, "paper", "RESULTS.md")
    if os.path.exists(results):
        try:
            from IPython.display import Markdown, display
            display(Markdown(open(results, encoding="utf-8").read()))
        except Exception:
            print(open(results, encoding="utf-8").read())
    _log(f"{res['progress']} | paper outputs: {res['paper']} | to continue: add this version's output as input and set RESUME_FROM")
'''

FOOTER = r'''

if __name__ == "__main__":
    main()
'''


def code_archive() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for fn in sorted(os.listdir(PKG)):
            if fn.endswith(".py"):
                info = zipfile.ZipInfo(f"inklingua/{fn}", date_time=(2026, 1, 1, 0, 0, 0))     # fixed timestamps: a reproducible archive
                info.compress_type = zipfile.ZIP_DEFLATED
                with open(os.path.join(PKG, fn), "rb") as f:
                    z.writestr(info, f.read())
    return buf.getvalue()


def build() -> str:
    blob = code_archive()
    b64 = base64.b64encode(blob).decode()
    lines = [b64[i:i + 120] for i in range(0, len(b64), 120)]
    embed = ("\n# ---------------------------------------------------------------------- the InkLingua package (zip, base64; source: the repository)\n"
             f'_CODE_SHA256 = "{hashlib.sha256(blob).hexdigest()}"\n_CODE_ZIP_B64 = (\n' + "\n".join(f'    "{l}"' for l in lines) + "\n)\n")
    src = HEADER + BODY + embed + FOOTER
    nb = {"cells": [{"cell_type": "code", "metadata": {"trusted": True}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}],
          "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True, "language": "python",
                                  "sourceType": "notebook", "isGpuEnabled": True}}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    return OUT


if __name__ == "__main__":
    p = build()
    print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB, 1 cell; code archive {len(code_archive()) / 1024:.0f} KB)")
