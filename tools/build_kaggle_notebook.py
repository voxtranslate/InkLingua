#!/usr/bin/env python3
"""
Build the single-file Kaggle notebook  notebooks/InkLingua_Kaggle_T4x2.ipynb  from the package sources.

The notebook embeds every module of `inklingua/` as a `%%writefile` cell (one module per cell, collapsed section), so it runs on a
Kaggle "GPU T4 x2" session WITHOUT cloning the repository.  Re-run this script after changing the package so that the notebook
stays in sync:

    python tools/build_kaggle_notebook.py
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "inklingua")
OUT = os.path.join(ROOT, "notebooks", "InkLingua_Kaggle_T4x2.ipynb")

MODULE_ORDER = ["__init__", "config", "utils", "text", "decoding", "style", "datasets_registry", "data", "model", "riskcontrol", "llm",
                "pseudo", "frag", "alphabet", "protocol", "curves", "trainer", "render", "explain", "posthoc", "interpret", "inferencer",
                "adapt", "report", "suite", "selftest", "main"]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str, collapsed: bool = False) -> dict:
    meta = {"jupyter": {"source_hidden": True}} if collapsed else {}
    return {"cell_type": "code", "metadata": meta, "execution_count": None, "outputs": [], "source": text.strip("\n").splitlines(keepends=True)}


INTRO = r"""
# InkLingua on Kaggle (GPU T4 x2)

Style-disentangled, uncertainty-aware **handwriting -> text -> translation** with fragment-level risk-controlled self-distillation
(FRAG-RC), writer self-distillation, an LLM in the loop and a validation protocol in which the test split never decides anything.

This notebook is **self-contained**: section 2 writes the `inklingua/` package to disk (one cell per module), the rest configures and
runs it.  It is generated from the repository by `tools/build_kaggle_notebook.py`; the repository is the source of truth.

**How the two T4s are used.**  The recogniser trains on `cuda:0`; the LLM (Qwen2.5-1.5B, fp16, ~3 GB) sits on `cuda:1` for
translation and N-best rescoring.  T4s have no bf16, so training uses fp16 autocast with a gradient scaler (handled automatically).

**Before you run.**
1. Settings -> Accelerator: *GPU T4 x2*.  Settings -> Internet: *On* (the dataset and the LLM are downloaded on first use).
   Offline alternative: attach a Kaggle dataset with `train/validation/test.parquet` and set `DATA_SOURCE = "local"`.
2. A full IAM run (100 epochs, 30 % labels) takes a few hours on a T4.  Kaggle sessions are limited to ~9-12 h: the run **resumes**
   from `/kaggle/working/inklingua_runs/<run>/checkpoints/last.pt` if you re-run the notebook, so save the working directory
   (Save Version with outputs) and continue in a new session if needed.
3. Start with `QUICK_TEST = True` (a few minutes) to see every output once, then switch it off.

| section | what it does |
|---|---|
| 1 | environment: packages, GPUs |
| 2 | the package source (collapsed; one cell per module) |
| 3 | configuration knobs: mode, corpora, label budget, LLM |
| 4 | fast self-tests (~30 s) |
| 5 | train + evaluate (resumable) |
| 6 | results: headline numbers, curves, the interactive report, key figures |
| 7 | optional: suite (baselines + ablations), multi-corpus, fine-tuning, zero-shot adaptation |
| 8 | export the run directory as a zip |

Honesty note: nothing here has been validated on real IAM yet.  Whether unlabeled data helps at all is exactly what the **suite**
(section 7) measures: `labeled_only` vs `full` over several seeds.  IAM is distributed for non-commercial research.
"""

SETUP = r"""
# ---- 1. Environment -------------------------------------------------------------------------------------------------------
import os, subprocess, sys, importlib

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

_missing = [p for p, m in (("editdistance", "editdistance"), ("opencv-python-headless", "cv2"), ("datasets", "datasets"),
                           ("transformers", "transformers"), ("accelerate", "accelerate"), ("sentencepiece", "sentencepiece"),
                           ("sacremoses", "sacremoses"), ("matplotlib", "matplotlib"), ("tqdm", "tqdm"))
            if importlib.util.find_spec(m) is None]
if _missing:
    print("installing:", _missing)
    _pip(*_missing)

import torch
print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} {p.name}  {p.total_memory / 2**30:.1f} GB  compute {p.major}.{p.minor}")
os.makedirs("inklingua", exist_ok=True)
ON_KAGGLE = os.path.isdir("/kaggle/working")
WORK_DIR = "/kaggle/working/inklingua_runs" if ON_KAGGLE else "./inklingua_runs"
print("work dir:", WORK_DIR)
"""

KNOBS = r"""
# ---- 3. Configuration knobs ------------------------------------------------------------------------------------------------
# Everything else keeps the defaults of inklingua/config.py (documented there).

MODE = "single"            # "smoke"  : synthetic plumbing test, no download (minutes)
                           # "single" : train + evaluate ONE configuration (the default experiment)
                           # "suite"  : baselines + ablations x seeds -> suite_summary.md  (what a paper needs; long)
                           # "finetune": continue a checkpoint (INIT_FROM) on the corpora in SOURCES
                           # "report" : rebuild the report of an existing run from its files
RUN_NAME = "inklingua_iam"
SOURCES = ("iam",)         # registry keys; several = multi-corpus training, e.g. ("iam", "rimes") or ("iam", "khatt")
                           # registry: iam rimes belfort esposalles himanis home_alcar newseye_de norhand1 norhand3 popp casia khatt
LABELED_FRACTION = 0.30    # share of the training lines whose transcription is used
EPOCHS = 100
SEED = 1337
DATA_SOURCE = "hf"         # "hf" (download, Internet on) | "local" (LOCAL_DIR with train/validation/test .parquet)
LOCAL_DIR = ""             # e.g. "/kaggle/input/iam-line-parquet"
USE_LLM = True             # Qwen on the second GPU: translation + N-best rescoring.  False -> "dummy" backend (no download)
LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_LANGUAGE = "fr"
SUITE_PRESET = "quick"     # for MODE="suite": "quick" (labeled_only, full) | "paper" (8 variants) | "all"
SUITE_SEEDS = (1337, 1338, 1339)
INIT_FROM = ""             # for MODE="finetune": path of best.pt / last.pt / adapted.pt
QUICK_TEST = True          # True: 1,200 train / 300 val / 300 test lines, 6 epochs, small LLM sample -> ~20 min dry run of every output
"""

BUILD_CFG = r"""
# ---- 3b. Build the Config from the knobs ------------------------------------------------------------------------------------
from inklingua import Config, main
from inklingua.config import SUITE_PRESETS

if MODE == "smoke":
    cfg = Config.smoke(os.path.join(WORK_DIR, "smoke"))
else:
    cfg = Config()
    cfg.run.name, cfg.run.work_dir, cfg.run.seed, cfg.run.mode = RUN_NAME, WORK_DIR, SEED, MODE
    cfg.data.sources, cfg.data.labeled_fraction, cfg.data.source = tuple(SOURCES), LABELED_FRACTION, DATA_SOURCE
    if DATA_SOURCE == "local":
        cfg.data.local_dir = LOCAL_DIR
    cfg.train.epochs = EPOCHS
    cfg.llm.backend = "causal" if USE_LLM else "dummy"
    cfg.llm.model_name, cfg.llm.target_language = LLM_NAME, TARGET_LANGUAGE
    cfg.llm.device = "auto"                 # second GPU when available
    cfg.run.suite_preset, cfg.run.suite_seeds = SUITE_PRESET, tuple(SUITE_SEEDS)
    if MODE == "finetune":
        cfg.train.init_from = INIT_FROM
    if QUICK_TEST:
        cfg.run.name = RUN_NAME + "_quick"
        cfg.data.limit_train, cfg.data.limit_val, cfg.data.limit_test = 1200, 300, 300
        cfg.train.epochs, cfg.train.min_epochs, cfg.train.patience = 6, 0, 0
        cfg.train.frag.warmup_epochs, cfg.train.frag.refresh_every, cfg.train.frag.min_chars = 2, 2, 100
        cfg.train.aux_start_epoch, cfg.train.aux_ramp_epochs, cfg.train.vq_start_epoch, cfg.train.vq_anneal_epochs = 1, 1, 1, 1
        cfg.infer.n_translate, cfg.infer.rescore_val_lines, cfg.infer.selective_val_lines = 24, 150, 300
        cfg.infer.pl_diag_samples, cfg.infer.bootstrap, cfg.infer.wsd_steps = 300, 500, (10, 20)
        cfg.infer.probe_lines, cfg.infer.uncertainty_lines, cfg.infer.longline_pairs = 200, 200, 100
    cfg.run.self_test = True                # the fast invariant tests run before the data is touched (~30 s)

print("mode:", MODE, "| run dir:", cfg.run_dir, "| corpora:", cfg.data.sources, "| labeled:", cfg.data.labeled_fraction,
      "| epochs:", cfg.train.epochs, "| LLM:", cfg.llm.backend, cfg.llm.model_name if cfg.llm.backend != "dummy" else "")
if cfg.run.mode == "suite":
    print("suite variants:", SUITE_PRESETS[cfg.run.suite_preset], "x seeds", cfg.run.suite_seeds)
"""

SELFTEST = r"""
# ---- 4. Fast self-tests (CPU, no data) ---------------------------------------------------------------------------------------
import logging, sys
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s", stream=sys.stdout, force=True)
from inklingua.selftest import run_self_tests
run_self_tests()
"""

RUN = r"""
# ---- 5. Train + evaluate (resumable: re-running this cell continues from the last checkpoint) ---------------------------------
metrics = main(cfg)
if isinstance(metrics, dict) and "CER" in metrics:
    ci = metrics.get("CER_ci95") or [float("nan"), float("nan")]
    print(f"\nCER {100 * metrics['CER']:.2f} %  [{100 * ci[0]:.2f}, {100 * ci[1]:.2f}]  |  WER {100 * metrics['WER']:.2f} %  |  "
          f"exact lines {100 * metrics['exact_line_accuracy']:.1f} %  |  system CER (validated options) "
          f"{100 * metrics.get('CER_system_final', metrics.get('CER_final_with_validated_posthoc', metrics['CER'])):.2f} %")
"""

RESULTS = r"""
# ---- 6. Results ----------------------------------------------------------------------------------------------------------------
import glob, json
from IPython.display import HTML, IFrame, Image, Markdown, display

run_dir = cfg.run_dir if cfg.run.mode != "suite" else None
if run_dir and os.path.isdir(os.path.join(run_dir, "inference", cfg.infer.split)):
    inf = os.path.join(run_dir, "inference", cfg.infer.split)
    M = json.load(open(os.path.join(inf, "00_summary", "metrics.json")))
    rows = [("CER (greedy)", M["CER"]), ("CER 95 % CI", M["CER_ci95"]), ("WER", M["WER"]), ("exact lines", M["exact_line_accuracy"]),
            ("CER + validated post-hoc", M.get("CER_final_with_validated_posthoc")), ("system CER (incl. LLM if adopted)", M.get("CER_system_final")),
            ("looks at this split so far", M.get("evaluation_looks_so_far"))]
    display(Markdown("### Headline numbers\n\n| metric | value |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows)))
    if M.get("per_corpus"):
        display(Markdown("### Per corpus\n\n| corpus | lines | CER | WER |\n|---|---|---|---|\n" + "\n".join(
            f"| {k} | {v['lines']} | {100 * v['CER']:.2f} % | {100 * v['WER']:.2f} % |" for k, v in M["per_corpus"].items())))
    ph = M.get("posthoc") or {}
    if ph:
        display(Markdown("### Post-hoc options (adopted only on a significant val_gate gain)\n\n| option | adopted | test CER |\n|---|---|---|\n" + "\n".join(
            f"| {k} | {ph[k].get('adopted')} | {100 * ph[k]['test_CER']:.2f} % |" for k in ("view_ensemble", "tta", "wsd", "space_bonus")
            if isinstance(ph.get(k), dict) and ph[k].get("test_CER") is not None)))
    display(Markdown("### Training curves"))
    display(Image(os.path.join(run_dir, "curves", "curves_all.png"), width=1400))
    display(Markdown("### Where the errors come from / uncertainty / explanations"))
    for rel in ("03_error_structure/error_structure.png", "02_uncertainty/calibration.png", "02_uncertainty/uncertainty_decomposition.png",
                "10_explanations/layer_probes.png", "07_robustness/robustness_profile.png", "11_translation/translation_ablation.png"):
        p = os.path.join(inf, rel)
        if os.path.exists(p):
            display(Markdown(f"`{rel}`")), display(Image(p, width=1100))
    dash = sorted(glob.glob(os.path.join(inf, "10_explanations", "samples", "*", "dashboard.png")))
    for p in dash[:2]:
        display(Markdown(f"`{os.path.relpath(p, inf)}`")), display(Image(p, width=1100))
    rep = os.path.join(inf, "12_report", "index.html")
    if os.path.exists(rep):
        display(Markdown("### The interactive report (also saved as `12_report/index.html`)"))
        display(IFrame(src=os.path.relpath(rep, os.getcwd()), width="100%", height=900))
elif cfg.run.mode == "suite":
    p = os.path.join(cfg.run.work_dir, f"{cfg.run.name}_suite", "suite_summary.md")
    display(Markdown(open(p).read()) if os.path.exists(p) else Markdown("no suite summary yet"))
else:
    print("no inference folder yet for", run_dir)
"""

OPTIONAL = r"""
# ---- 7. Optional recipes (set RUN_* to True and run) ------------------------------------------------------------------------
RUN_SUITE, RUN_MULTI, RUN_FINETUNE = False, False, False

if RUN_SUITE:         # the decisive experiment: does the unlabeled data help?  labeled_only vs full (preset "quick"), >= 3 seeds
    s = Config(); s.run.name, s.run.work_dir, s.run.mode = RUN_NAME, WORK_DIR, "suite"
    s.run.suite_preset, s.run.suite_seeds = "quick", (1337, 1338, 1339)
    s.llm.backend = "causal" if USE_LLM else "dummy"
    s.train.epochs = 60                       # validation CER is flat after ~30 epochs in every IAM run so far
    summary = main(s)

if RUN_MULTI:         # English + French + Arabic (right-to-left) at once; per-corpus FRAG-RC cells and per-corpus metrics
    m_ = Config.multi(("iam", "rimes", "khatt")); m_.run.work_dir = WORK_DIR
    m_.data.domain_balance = 0.5              # visit the smaller corpora more often
    m_.llm.backend = "causal" if USE_LLM else "dummy"
    metrics_multi = main(m_)

if RUN_FINETUNE:      # continue the IAM checkpoint on another corpus; the alphabet grows without changing the model's behaviour
    f_ = Config(); f_.run.name, f_.run.work_dir, f_.run.mode = RUN_NAME + "_ft_esposalles", WORK_DIR, "finetune"
    f_.train.init_from = os.path.join(WORK_DIR, RUN_NAME, "checkpoints", "best.pt")
    f_.data.sources, f_.train.epochs = ("esposalles",), 30
    f_.llm.backend = "causal" if USE_LLM else "dummy"
    metrics_ft = main(f_)

# Zero-shot adaptation to a FOLDER of line images in a new language (no or few labels):
#   a = Config(); a.run.name, a.run.work_dir, a.run.mode = RUN_NAME, WORK_DIR, "adapt"
#   a.adapt.lines_dir = "/kaggle/input/my-lines"; a.adapt.charset = "éèàùç"; a.adapt.target_languages = ("en", "fr")
#   a.adapt.anchors_file = "/kaggle/input/my-lines/anchors.tsv"   # optional "file<TAB>text"
#   main(a)                                   # -> <run>/adapt/adapt_results.csv, adapted.pt, adapt_metrics.json
"""

EXPORT = r"""
# ---- 8. Export ---------------------------------------------------------------------------------------------------------------
import shutil
if ON_KAGGLE and os.path.isdir(cfg.run_dir):
    zip_path = shutil.make_archive(os.path.join("/kaggle/working", os.path.basename(cfg.run_dir) + "_outputs"), "zip",
                                   root_dir=cfg.run_dir, base_dir=".")
    print("zipped run directory ->", zip_path, f"({os.path.getsize(zip_path) / 2**20:.1f} MB)")
    print("checkpoints stay in", cfg.ckpt_dir, "(best.pt / last.pt); keep /kaggle/working to resume in a new session")
"""


def build() -> str:
    cells = [md(INTRO), md("## 1. Environment"), code(SETUP), md(
        "## 2. The package source\n\nOne cell per module, written to `./inklingua/`.  Collapsed on purpose: read the repository for the code; "
        "run the cells (Run -> Run all cells of this section).")]
    for name in MODULE_ORDER:
        with open(os.path.join(PKG, name + ".py"), encoding="utf-8") as f:
            src = f.read()
        cells.append(code(f"%%writefile inklingua/{name}.py\n{src}", collapsed=True))
    cells += [md("## 3. Configuration"), code(KNOBS), code(BUILD_CFG), md("## 4. Self-tests"), code(SELFTEST),
              md("## 5. Train and evaluate"), code(RUN), md("## 6. Results"), code(RESULTS),
              md("## 7. Optional: suite, multi-corpus, fine-tuning, zero-shot adaptation"), code(OPTIONAL), md("## 8. Export"), code(EXPORT)]
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True, "language": "python",
                                  "sourceType": "notebook", "isGpuEnabled": True}}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return OUT


if __name__ == "__main__":
    p = build()
    print("wrote", p, f"({os.path.getsize(p) / 1024:.0f} KB, {sum(1 for _ in MODULE_ORDER)} module cells)")
