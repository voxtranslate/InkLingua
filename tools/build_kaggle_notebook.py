#!/usr/bin/env python3
"""
Build  notebooks/InkLingua_Kaggle_T4x2.ipynb : ONE readable code cell that contains ALL of InkLingua -- every class and function of the
`inklingua/` package, in dependency order, one section per module -- followed by the notebook entry point that runs the complete experiment
plan of the article on a Kaggle "GPU T4 x2" session (and continues where it stopped in the next session).

How the single cell works
  * Executed in the notebook, the cell defines everything and calls `notebook_main()`.
  * Every training / evaluation runs in its own process (one per GPU, memory freed between runs).  The cell therefore writes ITS OWN
    SOURCE to  <work dir>/_plan/inklingua_notebook.py  and starts  `python inklingua_notebook.py {run|adapt|prepare|gpucheck} spec result`;
    executed that way, the same code runs one worker job instead of the plan (see the end of the cell).
  * The repository package stays the source of truth (tests, history); this file is generated from it.  Only the repository self-tests
    (inklingua/selftest.py, synthetic data) are not part of the notebook.

Re-run after changing the package:   python tools/build_kaggle_notebook.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import textwrap
from typing import Dict, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(ROOT, "inklingua")
OUT = os.path.join(ROOT, "notebooks", "InkLingua_Kaggle_T4x2.ipynb")
EXCLUDED = {"__init__", "selftest"}            # __init__: re-exports only (its docstring becomes the table of contents); selftest: repository tests
BAR = "# " + "=" * 138

# ======================================================================================================================================
# the text of the cell around the package code
# ======================================================================================================================================
HEADER = r'''# ======================================================================================================================================
#  InkLingua -- the complete experiment plan of the article, in ONE readable cell.
#
#  Kaggle: Settings -> Accelerator "GPU T4 x2", Internet "On".  Then "Save Version" -> "Save & Run All (Commit)".
#  Optional: Add-ons -> Secrets -> a secret named HF_TOKEN (a Hugging Face read token): faster downloads, higher rate limits.
#
#  WHAT RUNS (real data only: the corpora of the registry at pinned Hub commits; no synthetic data, no dummy model)
#    For every corpus, one after the other (IAM first, then from the smallest corpus to the largest), and for every seed:
#      core      labeled_only | ctc_only | legacy_pl (line-level pseudo-labels) | full (InkLingua) | full_supervised (100 % labels)
#      ablations 14 training ablations of InkLingua
#      budget    labeled_only and full at 5 % and 10 % labels
#    Then one model trained on all corpora at once (joint), and the zero-shot adaptation of the IAM model to French and Catalan.
#    The epoch budget of each corpus follows its size, script and era (section "schedule"; IAM reproduces the published defaults).
#
#  HOW IT RUNS
#    1. every GPU is checked by a few training steps of the real model on random tensors (drivers / CUDA / cuDNN / memory);
#    2. every corpus is downloaded once (pinned commit, offline afterwards) and decoded once, in a CPU-only process, into a line cache
#       shared by all its runs; the caches of unfinished corpora are saved with the output and restored in the next session;
#    3. two runs train in parallel, one per T4, each in its own process; every training writes a step checkpoint at least every
#       10 minutes and when it is stopped, and resumes EXACTLY at that step (same batches, same random state);
#    4. a failed run shows the end of its log here; after 4 failures in a row with no finished run the session stops (systematic error).
#    Everything is in /kaggle/working/inklingua_runs:  paper/RESULTS.md (+ tables, figures, REPRODUCIBILITY.md, STATUS.md), one folder per run.
#
#  TO CONTINUE IN THE NEXT SESSION
#    "Add Input" -> the Output of the previous version of THIS notebook, put its path in RESUME_FROM below, Save & Run All again.
#    Cost: the full plan is ~2,300 T4 GPU-hours (~100 sessions).  PLAN_OVERRIDES narrows it, e.g. {"seeds": (1337,)} (one third);
#    {"dry_run": True} only checks the GPUs and prepares the first corpus (minutes) -- a good first session.
#
#  CONTENTS (one section per component, in reading order)
{toc}
# ======================================================================================================================================
from __future__ import annotations

'''

SETTINGS = r'''

# ======================================================================================================================================
#  SETTINGS -- the only lines to edit
# ======================================================================================================================================
RESUME_FROM = ""                 # e.g. "/kaggle/input/inklingua-experiment-plan" (the previous version's output); "" on the first session
PLAN_OVERRIDES: Dict[str, Any] = {}   # e.g. {"seeds": (1337,)}, {"corpora": ("iam", "khatt")}, {"order": "tier"}, {"dry_run": True}; {} = full plan
'''

FOOTER = r'''

# ======================================================================================================================================
#  NOTEBOOK ENTRY POINT
# ======================================================================================================================================
SESSION_START = time.time()
WORK_DIR = "/kaggle/working/inklingua_runs" if os.path.isdir("/kaggle/working") else os.path.abspath("inklingua_runs")
CACHE_DIR = "/tmp/inklingua_cache"


def _log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def _ensure_packages() -> None:
    """torch >= 2.5 is required (weights_only checkpoints, torch.amp); the other packages are installed when missing."""
    import importlib.util
    import subprocess
    if tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) < (2, 5):
        raise RuntimeError(f"torch {torch.__version__} is too old (>= 2.5 needed): choose a recent Kaggle environment")
    need = {"editdistance": "editdistance", "pyarrow": "pyarrow", "huggingface_hub": "huggingface_hub", "transformers": "transformers",
            "accelerate": "accelerate", "datasets": "datasets", "sentencepiece": "sentencepiece"}
    missing = [pip for mod, pip in need.items() if importlib.util.find_spec(mod) is None]
    if missing:
        _log(f"installing {missing}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)


def _preflight() -> None:
    import subprocess
    import urllib.request
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    if not out:
        raise RuntimeError("no GPU: set Settings -> Accelerator to 'GPU T4 x2'")
    _log("GPUs: " + " | ".join(out.splitlines()))
    try:
        urllib.request.urlopen("https://huggingface.co/api/datasets/Teklia/IAM-line", timeout=20).read(64)
    except Exception as e:
        raise RuntimeError(f"the Hugging Face Hub is not reachable ({e}): set Settings -> Internet to 'On'") from e
    free = shutil.disk_usage("/tmp").free
    _log(f"free disk for downloads and caches (/tmp): {free / 2 ** 30:.0f} GB (largest corpus: ~5.3 GB of parquet)")


def _write_own_source() -> str:
    """The run processes execute this very cell as a script: write its source next to the plan (and record its sha256)."""
    import hashlib
    import linecache
    src = "".join(linecache.getlines(notebook_main.__code__.co_filename))
    if "def notebook_main" not in src:
        try:
            from IPython import get_ipython
            src = get_ipython().user_ns["In"][-1]          # IPython fallback: the text of the cell being executed
        except Exception as e:
            raise RuntimeError("cannot read the source of this cell (run it in a Jupyter / Kaggle notebook)") from e
    path = os.path.join(WORK_DIR, "_plan", "inklingua_notebook.py")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(src)
    sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    with open(os.path.join(WORK_DIR, "_plan", "code_sha256.txt"), "w") as f:
        f.write(sha + "\n")
    _log(f"code of this cell written to {path} ({src.count(chr(10)):,} lines, sha256 {sha[:16]}...)")
    return path


def notebook_main() -> None:
    global WORKER_ENTRY
    _ensure_packages()
    _preflight()
    os.makedirs(os.path.join(WORK_DIR, "_plan"), exist_ok=True)
    WORKER_ENTRY = [_write_own_source()]                    # run processes execute this cell's code (see worker_command)
    quiet_third_party()                                     # no per-request HTTP lines, no progress bars
    _log(f"Hugging Face Hub: {resolve_hf_token()}")         # HF_TOKEN (environment or Kaggle secret) is exported to every run process
    plan = PlanConfig(work_dir=WORK_DIR, cache_dir=CACHE_DIR, resume_from=RESUME_FROM, **PLAN_OVERRIDES)
    plan.session_hours = max(0.5, plan.session_hours - (time.time() - SESSION_START) / 3600.0)
    _log(f"InkLingua v{__version__} | experiment plan | resume_from={RESUME_FROM or '-'} | overrides={PLAN_OVERRIDES or '-'}")
    res = run_plan(plan, code_root=os.path.join(WORK_DIR, "_plan"), log=_log)
    results = os.path.join(WORK_DIR, "paper", "RESULTS.md")
    if os.path.exists(results):
        try:
            from IPython.display import Markdown, display
            display(Markdown(open(results, encoding="utf-8").read()))
        except Exception:
            print(open(results, encoding="utf-8").read())
    _log(f"{res['progress']} | paper outputs: {res['paper']} | to continue: add this version's output as input and set RESUME_FROM")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] in WORKER_COMMANDS:     # a run process started by the plan:  <file> run spec.json result.json
        worker_main(sys.argv[1:])
    else:                                                          # the notebook cell
        notebook_main()
'''


# ======================================================================================================================================
# flattening the package
# ======================================================================================================================================
READING_ORDER = ("utils", "config", "datasets_registry", "text", "style", "decoding", "alphabet", "data", "model", "riskcontrol", "frag", "pseudo",
                 "llm", "curves", "render", "protocol", "explain", "posthoc", "interpret", "inferencer", "report", "trainer", "schedule", "main",
                 "adapt", "suite", "paper", "experiments")


def module_order() -> List[str]:
    """Modules in reading order (data -> model -> training -> evaluation -> experiment plan), constrained so that a module always comes after
    every module it imports at top level."""
    deps: Dict[str, Set[str]] = {}
    for fn in sorted(os.listdir(PKG)):
        name = fn[:-3]
        if not fn.endswith(".py") or name in EXCLUDED:
            continue
        tree = ast.parse(open(os.path.join(PKG, fn), encoding="utf-8").read())
        deps[name] = {n.module.split(".")[0] for n in tree.body if isinstance(n, ast.ImportFrom) and n.level == 1 and n.module} - EXCLUDED
    order: List[str] = []
    while len(order) < len(deps):
        rank = {m: (READING_ORDER.index(m) if m in READING_ORDER else len(READING_ORDER), m) for m in deps}
        ready = sorted((m for m in deps if m not in order and deps[m] <= set(order)), key=rank.get)
        if not ready:
            raise RuntimeError(f"import cycle among {sorted(set(deps) - set(order))}")
        order.append(ready[0])
    return order


def _is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name)
            and node.test.left.id == "__name__")


SEEN_TOP_BLOCKS: Set[str] = set()                      # identical top-level try-blocks (the tqdm import of several modules) appear once


def flatten_module(name: str, imports: Dict[str, Set[Tuple[str, ...]]]) -> Tuple[str, str]:
    """(docstring, body) of one module with its imports removed (absolute ones are collected in `imports`, package-relative ones vanish:
    every name lives in the one namespace of the cell).  Formatting and comments are kept line for line."""
    src = open(os.path.join(PKG, name + ".py"), encoding="utf-8").read()
    lines = src.splitlines()
    tree = ast.parse(src)
    drop: Set[int] = set()                                   # 0-based line numbers to remove
    insert_pass: Dict[int, str] = {}
    doc = ast.get_docstring(tree) or ""

    def span(n: ast.AST) -> range:
        return range(n.lineno - 1, n.end_lineno)

    for n in tree.body:
        if isinstance(n, ast.Expr) and isinstance(getattr(n, "value", None), ast.Constant) and isinstance(n.value.value, str) and n is tree.body[0]:
            drop.update(span(n))
        elif isinstance(n, ast.ImportFrom) and n.module == "__future__":
            drop.update(span(n))
        elif isinstance(n, ast.ImportFrom) and n.level > 0:
            drop.update(span(n))
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                imports.setdefault("from", set()).add((n.module, a.name, a.asname or ""))
            drop.update(span(n))
        elif isinstance(n, ast.Import):
            for a in n.names:
                imports.setdefault("import", set()).add((a.name, a.asname or ""))
            drop.update(span(n))
        elif _is_main_guard(n):
            drop.update(span(n))
        elif isinstance(n, ast.Try):
            key = ast.dump(n)
            if key in SEEN_TOP_BLOCKS:
                drop.update(span(n))
            SEEN_TOP_BLOCKS.add(key)
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute)
              and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == "matplotlib" and n.value.func.attr == "use"):
            drop.update(span(n))                             # the backend is chosen once, at the top of the cell
    # package-relative imports inside functions (lazy imports): the names are global in the cell
    for parent in ast.walk(tree):
        for field in ("body", "orelse", "finalbody", "handlers"):
            block = getattr(parent, field, None)
            if not isinstance(block, list) or not block or not all(isinstance(x, ast.stmt) for x in block):
                continue
            rel = [x for x in block if isinstance(x, ast.ImportFrom) and x.level > 0 and x.col_offset > 0]
            for x in rel:
                drop.update(span(x))
            if rel and len(rel) == len(block):
                insert_pass[rel[0].lineno - 1] = " " * rel[0].col_offset + "pass"
    out = []
    for i, line in enumerate(lines):
        if i in insert_pass:
            out.append(insert_pass[i])
        if i not in drop:
            out.append(line)
    body = re.sub(r"\n{4,}", "\n\n\n", "\n".join(out)).strip("\n")
    return doc, body


def render_imports(imports: Dict[str, Set[Tuple[str, ...]]]) -> str:
    std = {"__future__", "ast", "base64", "collections", "concurrent", "contextlib", "copy", "csv", "dataclasses", "fcntl", "functools", "glob",
           "hashlib", "html", "importlib", "io", "itertools", "json", "linecache", "logging", "math", "multiprocessing", "os", "pickle", "platform",
           "random", "re", "shutil", "signal", "statistics", "string", "subprocess", "sys", "tempfile", "textwrap", "threading", "time", "traceback",
           "types", "typing", "unicodedata", "urllib", "warnings", "zipfile"}
    plain: Dict[str, List[str]] = {"std": [], "third": []}
    for mod, alias in sorted(imports.get("import", set())):
        if mod.split(".")[0] == "matplotlib":
            continue
        plain["std" if mod.split(".")[0] in std else "third"].append(f"import {mod}" + (f" as {alias}" if alias else ""))
    froms: Dict[str, Set[str]] = {}
    for mod, nm, alias in imports.get("from", set()):
        froms.setdefault(mod, set()).add(nm + (f" as {alias}" if alias else ""))
    for mod, names in sorted(froms.items()):
        stmt = f"from {mod} import " + ", ".join(sorted(names, key=str.lower))
        if len(stmt) > 140:
            stmt = f"from {mod} import (" + ", ".join(sorted(names, key=str.lower)) + ")"
        plain["std" if mod.split(".")[0] in std else "third"].append(stmt)
    return ("\n".join(sorted(set(plain["std"]))) + "\n\nimport matplotlib\n\nmatplotlib.use(\"Agg\")                       # figures are written to files\n"
            "import matplotlib.pyplot as plt  # noqa: E402\n" + "\n".join(sorted(set(plain["third"]))) + "\n")


# ======================================================================================================================================
# the notebook is self-contained: section names instead of file names, nothing that points to the repository
# ======================================================================================================================================
TITLES = {"utils": "Utilities", "config": "Configuration", "datasets_registry": "Dataset registry", "text": "Text and metrics",
          "style": "Style descriptors", "decoding": "CTC decoding", "alphabet": "Alphabet growth", "data": "Data", "model": "Model",
          "riskcontrol": "Risk control", "frag": "FRAG-RC", "pseudo": "Line-level pseudo-labels (baseline)", "llm": "Language model",
          "curves": "Training curves", "render": "Style-matched rendering", "protocol": "Evaluation protocol", "explain": "Explanations",
          "posthoc": "Post-hoc options", "interpret": "Interpretability", "inferencer": "Inference", "report": "Run report",
          "trainer": "Training", "schedule": "Training budget", "main": "Single-run entry point", "adapt": "Adaptation", "suite": "Suite",
          "paper": "Paper outputs", "experiments": "Experiment plan"}

FORBIDDEN = (r"github", r"\brepositor", r"inklingua/", r"\binklingua\.(?!py\b)", r"\binklingua\.py\b", r"docs/", r"\bpytest\b", r"selftest",
             r"self-test", r"\btests/", r"git clone", r"Config\.smoke")

PACKAGE_WORKER_ENTRY = (
    "# How a run process is started.  The repository runs the package module; the single-cell notebook writes its own source to a file and\n"
    "# sets WORKER_ENTRY = [that file].  -X faulthandler: a native crash (SIGABRT / SIGSEGV) prints the Python stack into the run's log.\n"
    'WORKER_ENTRY: List[str] = ["-m", "inklingua.experiments"]')
NOTEBOOK_WORKER_ENTRY = (
    "# How a run process is started: notebook_main() writes the source of this cell to a file and sets WORKER_ENTRY = [that file].\n"
    "# -X faulthandler: a native crash (SIGABRT / SIGSEGV) prints the Python stack into the run's log.\n"
    "WORKER_ENTRY: List[str] = []")


def neutral_text(src: str) -> str:
    """Rewrite the package's references to its own files and documents into references to sections of this notebook."""
    def section(m: re.Match) -> str:                    # only known components; anything else is left for check_independent to report
        return f'section "{TITLES[m.group(1)]}"' if m.group(1) in TITLES else m.group(0)
    if PACKAGE_WORKER_ENTRY not in src:
        raise SystemExit("the worker-entry block of the experiment plan was not found (update PACKAGE_WORKER_ENTRY)")
    src = src.replace(PACKAGE_WORKER_ENTRY, NOTEBOOK_WORKER_ENTRY)
    src = src.replace(" (all tested in inklingua.selftest)", "")
    src = re.sub(r"\s*\(docs/[A-Za-z_]+\.md[^)]*\)", "", src)
    src = re.sub(r"\binklingua\.(\w+)\.(\w+)", r"\2", src)
    src = re.sub(r"\binklingua\.(\w+)\b", section, src)
    src = re.sub(r"\binklingua/(\w+)\.py\b", section, src)
    return src


def check_independent(src: str) -> None:
    """Refuse to write a notebook that still refers to the repository."""
    bad = [(p, m.group(0), src.count("\n", 0, m.start()) + 1) for p in FORBIDDEN for m in re.finditer(p, src, flags=re.I)]
    if bad:
        raise SystemExit("the notebook still refers to the repository:\n" + "\n".join(f"  line {ln}: {t!r}   ({p})" for p, t, ln in bad[:40]))


def toc_line(name: str, doc: str) -> str:
    first = " ".join(doc.strip().split("\n\n")[0].split())
    return f"#    {name:<42} {textwrap.shorten(first, 90, placeholder=' ...')}"


def build_source() -> str:
    imports: Dict[str, Set[Tuple[str, ...]]] = {}
    sections, toc = [], []
    order = module_order()
    for k, name in enumerate(order, 1):
        doc, body = flatten_module(name, imports)
        title = TITLES.get(name, name)
        toc.append(toc_line(f"{k:2d}. {title}", doc))
        banner = [BAR, f"#  {k}. {title.upper()}", BAR]
        if doc:
            banner += ["#  " + line if line.strip() else "#" for line in doc.strip().splitlines()]
            banner.append(BAR)
        sections.append("\n".join(banner) + "\n" + body)
    version = re.search(r'__version__ = "([^"]+)"', open(os.path.join(PKG, "__init__.py")).read()).group(1)
    header = HEADER.replace("{toc}", "\n".join(toc))
    src = (header + render_imports(imports) + SETTINGS + f'\n__version__ = "{version}"\n\n\n' + "\n\n\n".join(sections) + "\n" + FOOTER)
    src = neutral_text(src)
    check_independent(src)
    return src


def build() -> Tuple[str, str]:
    src = build_source()
    nb = {"cells": [{"cell_type": "code", "metadata": {"trusted": True}, "execution_count": None, "outputs": [], "source": src.splitlines(keepends=True)}],
          "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"},
                       "kaggle": {"accelerator": "nvidiaTeslaT4", "dataSources": [], "isInternetEnabled": True, "language": "python",
                                  "sourceType": "notebook", "isGpuEnabled": True}}}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    return OUT, src


if __name__ == "__main__":
    p, s = build()
    print(f"wrote {p} ({os.path.getsize(p) / 1024:.0f} KB, 1 cell, {s.count(chr(10)):,} lines)")
