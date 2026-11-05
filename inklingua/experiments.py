"""
The complete experiment plan of the article, as ONE resumable, budget-aware program (`run_plan`).

What it runs (docs/EXPERIMENT_PLAN.md has the rationale and the cost):

  stage per corpus (default order: IAM first, then the other corpora from the smallest to the largest), for every seed:
      core     labeled_only, ctc_only, legacy_pl, full, full_supervised           (the main table)
      ablation the 14 TRAINING ablations of the proposed method                   (the ablation table)
      budget   labeled_only and full at 5 % and 10 % labels                       (the label-budget curves; 30 % and 100 % come from core)
  joint stage     labeled_only and full trained on all corpora at once (per-corpus caps, cross-split decontamination)
  adaptation      zero-shot adaptation of the IAM model to French and Catalan lines, CER vs number of labeled target lines (0/25/100/200)
  paper outputs   tables (Markdown + LaTeX + CSV), figures, statistics, reproducibility and status reports, rebuilt after every run

Inference-only "variants" (no_wsd, no_posthoc, no LLM rescoring) are NOT retrained: every `full` run already measures the reading before and
after each validated option, so the paper tables read them from those runs.  Ablations of the legacy baseline (no_consistency, no_llm,
no_shadow, no_risk_smoothing) and redundant variants (no_pseudo = labeled_only; no_calibration = no FRAG-RC rule) are not part of the plan.

How it runs:
  * one run = one subprocess pinned to one GPU (`CUDA_VISIBLE_DEVICES`), so GPU memory is released between runs; with two GPUs (Kaggle T4 x2) two
    runs train side by side.  The LLM shares the GPU of the run that needs it.
  * every run resumes from its own last checkpoint; the ledger (`_plan/state.json`) records done / interrupted / failed runs; re-running the plan
    continues where it stopped (on Kaggle: attach the previous version's output and set `resume_from`).
  * a session deadline: no run starts unless training time remains after the evaluation reserve; a running training stops at the step where
    the deadline falls, with a step checkpoint (resumed exactly, mid-epoch included); evaluation is never started without its reserve.
  * after a run: checkpoints are pruned (best.pt kept only for the core variants at the main label budget), and when a corpus is no longer
    needed by any pending run its downloads and line caches are deleted (disk).
"""
from __future__ import annotations

import copy
import dataclasses
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .config import Config
from .datasets_registry import DATASETS
from .schedule import budget, complexity_factor
from .utils import json_safe

CORE_VARIANTS = ("labeled_only", "ctc_only", "legacy_pl", "full", "full_supervised")
TRAINING_ABLATIONS = ("no_frag", "no_strata", "no_sharpen", "no_mixline", "no_style_task", "no_ses", "no_vq", "no_film", "no_confusion",
                      "no_frame_cons", "no_mask", "no_aux_ctc", "no_v9_reg", "no_hscale")
DEFAULT_CORPORA = ("iam", "esposalles", "popp", "khatt", "rimes", "himanis", "norhand1", "belfort", "casia", "home_alcar", "norhand3")
JOINT_CORPORA = ("iam", "rimes", "belfort", "esposalles", "himanis", "home_alcar", "newseye_de", "norhand1", "popp", "casia", "khatt")
DEFAULT_CAPS = {"norhand3": {"train": 60000, "validation": 3000}, "home_alcar": {"validation": 3000}, "casia": {"validation": 3000}}
EVAL_RESERVE_MIN = {"full": 100, "standard": 40, "core": 15}
PRIOR_SEC_PER_STEP = 0.15          # T4 prior for the default model (replaced by measured rates as soon as runs finish)


OUTPUT_LIMIT_GB = 18.5                                                # Kaggle saves at most 20 GB of /kaggle/working


@dataclass
class PlanConfig:
    work_dir: str = "/kaggle/working/inklingua_runs"
    cache_dir: str = "/tmp/inklingua_cache"                       # downloads + line caches (not part of the saved output)
    corpora: Tuple[str, ...] = DEFAULT_CORPORA
    seeds: Tuple[int, ...] = (1337, 1338, 1339)
    labeled_fraction: float = 0.30
    core_variants: Tuple[str, ...] = CORE_VARIANTS
    ablations: Tuple[str, ...] = TRAINING_ABLATIONS
    ablation_corpora: Tuple[str, ...] = ()                        # () = every corpus of `corpora`
    label_sweep: Tuple[float, ...] = (0.05, 0.10)
    sweep_variants: Tuple[str, ...] = ("labeled_only", "full")
    joint: bool = True
    joint_corpora: Tuple[str, ...] = JOINT_CORPORA
    joint_variants: Tuple[str, ...] = ("labeled_only", "full")
    joint_caps: Dict[str, int] = field(default_factory=lambda: {"train": 12000, "validation": 1000})
    adaptation: bool = True
    adaptation_source: str = "iam"
    adaptation_targets: Tuple[str, ...] = ("rimes", "esposalles")
    adaptation_lines: int = 2000
    adaptation_eval_lines: int = 1000
    adaptation_anchor_counts: Tuple[int, ...] = (0, 25, 100, 200)
    caps: Dict[str, Dict[str, int]] = field(default_factory=lambda: copy.deepcopy(DEFAULT_CAPS))
    order: str = "corpus"                                         # "corpus": finish a corpus, then the next | "tier": core everywhere first
    llm_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    translation_target: str = "en"                                # English for every corpus (French for the English corpora)
    session_hours: float = 11.3                                   # Kaggle sessions stop at 12 h: keep a margin for the final reports
    max_parallel: int = 0                                         # 0 = one run per visible GPU
    max_attempts: int = 3
    max_consecutive_failures: int = 4                              # this many failed attempts in a row and no run finished in this session ->
                                                                  # the session stops (a systematic error, e.g. a broken environment)
    dry_run: bool = False                                         # True: check the GPUs, prepare the first corpus, print the plan, stop
    keep_checkpoints: str = "core"                                # "core" | "all" | "none" (best.pt after evaluation)
    cleanup_caches: bool = True
    persist_cache_gb: float = 8.0                                 # at the end of a session, the line caches (preprocessed lines + manifests) of
                                                                  # the corpora that still have pending runs are saved in the output
                                                                  # (<work_dir>/_cache/lines) up to this size, and restored by the next session:
                                                                  # a corpus that spans several sessions is downloaded and decoded once
    resume_from: str = ""                                         # previous output folder (Kaggle: /kaggle/input/<previous version>)

    @property
    def plan_dir(self) -> str:
        return os.path.join(self.work_dir, "_plan")


# ==============================================================================
# the plan
# ==============================================================================
def lf_tag(lf: float) -> str:
    return f"lf{int(round(100 * lf)):03d}"


def stage_list(plan: PlanConfig) -> List[Tuple[str, Tuple[str, ...]]]:
    st = [(c, (c,)) for c in plan.corpora if DATASETS[c].test_split]
    if plan.joint:
        st.append(("joint", tuple(plan.joint_corpora)))
    return st


def expected_lines(key: str, split: int, caps: Optional[Dict[str, int]]) -> int:
    sz = DATASETS[key].expected_sizes or ()
    if split >= len(sz):
        return 0
    cap = (caps or {}).get(("train", "validation", "test")[split])
    return int(min(sz[split], cap)) if cap else int(sz[split])


def stage_caps(plan: PlanConfig, stage: str, key: str) -> Optional[Dict[str, int]]:
    return dict(plan.joint_caps) if stage == "joint" else plan.caps.get(key)


def stage_schedule(plan: PlanConfig, stage: str, sources: Sequence[str]) -> Dict[str, Any]:
    """The training budget the run will compute from its data (inklingua/schedule.py), estimated from the published split sizes."""
    lines = [expected_lines(k, 0, stage_caps(plan, stage, k)) for k in sources]
    specs = [DATASETS[k] for k in sources]
    b = budget(sum(lines), complexity_factor(specs, lines), Config().data.batch_size)
    b.update({"train_lines": sum(lines), "val_lines": sum(expected_lines(k, 1, stage_caps(plan, stage, k)) for k in sources),
              "test_lines": sum(expected_lines(k, 2, stage_caps(plan, stage, k)) for k in sources),
              "complexity_factor": complexity_factor(specs, lines)})
    return b


def build_runs(plan: PlanConfig) -> List[Dict[str, Any]]:
    """Every run of the plan, with a priority key (lower = earlier) and its dependencies."""
    runs: List[Dict[str, Any]] = []
    main_lf = plan.labeled_fraction
    abl_on = set(plan.ablation_corpora) if plan.ablation_corpora else set(plan.corpora)
    stages = stage_list(plan)
    for si, (stage, sources) in enumerate(stages):
        sched = stage_schedule(plan, stage, sources)
        for ri, seed in enumerate(plan.seeds):
            items: List[Tuple[int, str, float]] = []
            variants = plan.joint_variants if stage == "joint" else plan.core_variants
            items += [(0, v, 1.0 if v == "full_supervised" else main_lf) for v in variants]
            if stage != "joint" and stage in abl_on:
                items += [(1, v, main_lf) for v in plan.ablations]
            if stage != "joint":
                items += [(2, v, lf) for lf in plan.label_sweep for v in plan.sweep_variants]
            for vi, (tier, v, lf) in enumerate(items):
                prof = "full" if (v == "full" and ri == 0 and abs(lf - main_lf) < 1e-9) else ("standard" if tier == 0 else "core")
                pri = (si, ri, tier, vi) if plan.order == "corpus" else (tier, ri, si, vi)
                runs.append({"id": f"{stage}/{v}_{lf_tag(lf)}_s{seed}", "kind": "run", "stage": stage, "sources": list(sources), "variant": v,
                             "seed": int(seed), "lf": float(lf), "tier": tier, "profile": prof, "priority": list(pri), "deps": [],
                             "planned_steps": int(sched["planned_steps"]), "epochs": int(sched["epochs"]), "test_lines": int(sched["test_lines"])})
        if plan.adaptation and stage == plan.adaptation_source:
            src_id = f"{stage}/full_{lf_tag(main_lf)}_s{plan.seeds[0]}"
            for ti, tgt in enumerate(plan.adaptation_targets):
                pri = (si, 0, 3, ti) if plan.order == "corpus" else (3, 0, si, ti)
                runs.append({"id": f"adaptation/{stage}_to_{tgt}", "kind": "adapt", "stage": "adaptation", "sources": [tgt], "variant": "adapt_curve",
                             "seed": int(plan.seeds[0]), "lf": main_lf, "tier": 3, "profile": "adapt", "priority": list(pri), "deps": [src_id],
                             "source_run": src_id, "target": tgt, "planned_steps": 0, "epochs": 0, "test_lines": plan.adaptation_eval_lines})
    runs.sort(key=lambda r: r["priority"])
    ids = [r["id"] for r in runs]
    assert len(ids) == len(set(ids)), "duplicate run ids in the plan"
    return runs


def apply_profile(cfg: Config, profile: str) -> None:
    """Evaluation depth.  full: every analysis (translation, interpretability, robustness, report); standard: metrics + every validated
    post-hoc option incl. writer self-distillation and LLM rescoring (the system CER); core: metrics + the cheap post-hoc options."""
    ic = cfg.infer
    if profile == "full":
        return
    ic.n_vis, ic.n_translate, ic.interpret, ic.robustness_lines, ic.pl_diag_samples, ic.report = 0, 0, False, 0, 0, False
    if profile == "core":
        ic.llm_rescoring, ic.wsd = False, False


def run_config(plan: PlanConfig, r: Dict[str, Any], n_parallel: int = 1) -> Config:
    base = Config()
    base.run.work_dir, base.run.seed, base.run.self_test, base.run.mode = plan.work_dir, r["seed"], False, "single"
    base.data.source, base.data.reader = "hf", "parquet"
    base.data.sources = tuple(r["sources"])
    base.data.labeled_fraction = r["lf"]
    base.data.hf_cache_dir = os.path.join(plan.cache_dir, "data")
    base.data.line_cache_dir = os.path.join(plan.cache_dir, "lines")
    base.data.max_text_len = 200                        # historical corpora have long lines; CTC feasibility is still checked per line
    base.data.num_workers = 1 if n_parallel > 1 else 2
    base.data.custom_sources = tuple(dataclasses.replace(DATASETS[k], max_lines=stage_caps(plan, r["stage"], k)) for k in r["sources"])
    if r["stage"] == "joint":
        base.data.decontaminate, base.data.domain_balance = True, 0.5
    base.train.auto_schedule = True
    base.llm.backend, base.llm.model_name, base.llm.device = "causal", plan.llm_name, "auto"
    base.llm.target_language = plan.translation_target
    apply_profile(base, r["profile"])
    cfg = Config.make_variant(r["variant"], base)
    cfg.run.name = r["id"]
    if r["variant"] == "full_supervised":
        cfg.data.labeled_fraction = 1.0
    return cfg


# ==============================================================================
# the ledger
# ==============================================================================
class Ledger:
    def __init__(self, plan: PlanConfig):
        self.path = os.path.join(plan.plan_dir, "state.json")
        self.lock = threading.Lock()
        self.state: Dict[str, Dict[str, Any]] = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.state = json.load(f)

    def get(self, rid: str) -> Dict[str, Any]:
        return self.state.setdefault(rid, {"status": "pending", "attempts": 0, "seconds": 0.0, "sessions": 0})

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(json_safe(self.state), f, indent=1)
        os.replace(tmp, self.path)


def run_done_marker(plan: PlanConfig, r: Dict[str, Any]) -> str:
    if r["kind"] == "adapt":
        return os.path.join(plan.work_dir, r["id"], "adapt_curve.json")
    return os.path.join(plan.work_dir, r["id"], "inference", "test", "00_summary", "metrics.json")


# ==============================================================================
# workers
# ==============================================================================
# How a run process is started.  The repository runs the package module; the single-cell notebook writes its own source to a file and
# sets WORKER_ENTRY = [that file].  -X faulthandler: a native crash (SIGABRT / SIGSEGV) prints the Python stack into the run's log.
WORKER_ENTRY: List[str] = ["-m", "inklingua.experiments"]


def worker_command(*args: str) -> List[str]:
    return [sys.executable, "-X", "faulthandler", *WORKER_ENTRY, *args]


def worker_env(plan: "PlanConfig", gpu: Optional[int], code_root: str) -> Dict[str, str]:
    """Environment of a run process: ONE visible GPU (or none), no progress bars, Hub cache of the session, the HF token of the notebook."""
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "" if gpu is None else str(gpu), "PYTHONPATH": code_root + os.pathsep + env.get("PYTHONPATH", ""),
                "TQDM_DISABLE": "1", "TOKENIZERS_PARALLELISM": "false", "HF_HOME": os.path.join(plan.cache_dir, "hf_home"), "PYTHONUNBUFFERED": "1",
                "PYTHONFAULTHANDLER": "1", "INKLINGUA_WORKER": "1"})
    return env


def tail(path: str, n: int = 25) -> str:
    """The last n lines of a log file (shown in the notebook when a run fails)."""
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return "(no log)"


def launch(plan: "PlanConfig", code_root: str, kind: str, payload: Dict[str, Any], folder: str, gpu: Optional[int],
           timeout: float) -> Tuple[Dict[str, Any], str, bool]:
    """Run one worker process (kind = run | adapt | prepare | gpucheck) and return (result, log path, stopped by timeout)."""
    os.makedirs(folder, exist_ok=True)
    spec_path, res_path, log_path = (os.path.join(folder, f"_{kind}_spec.json"), os.path.join(folder, f"_{kind}_result.json"),
                                     os.path.join(folder, "stdout.log" if kind in ("run", "adapt") else f"{kind}.log"))
    with open(spec_path, "w") as f:
        json.dump(json_safe(payload), f, indent=1)
    if os.path.exists(res_path):
        os.remove(res_path)
    stopped = False
    with open(log_path, "a") as lf:
        lf.write(f"\n===== {kind} {time.strftime('%Y-%m-%d %H:%M:%S')} on {'gpu ' + str(gpu) if gpu is not None else 'cpu'} =====\n")
        lf.flush()
        p = subprocess.Popen(worker_command(kind, spec_path, res_path), stdout=lf, stderr=subprocess.STDOUT, env=worker_env(plan, gpu, code_root),
                             cwd=plan.work_dir)
        try:
            p.wait(timeout=max(30.0, timeout))
        except subprocess.TimeoutExpired:
            stopped = True                                    # SIGTERM: a training writes a step checkpoint and exits
            p.terminate()
            try:
                p.wait(timeout=180)
            except subprocess.TimeoutExpired:
                p.kill()
    res = {"status": "failed", "error": f"no result file (exit code {p.returncode}{', SIGABRT: native abort' if p.returncode == -6 else ''}"
                                        f"{', SIGKILL: out of memory?' if p.returncode == -9 else ''}); see {log_path}"}
    if os.path.exists(res_path):
        with open(res_path) as f:
            res = json.load(f)
    elif stopped:
        res = {"status": "interrupted"}
    return res, log_path, stopped


def detect_gpus() -> List[int]:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout
        return [i for i, line in enumerate(l for l in out.splitlines() if l.startswith("GPU "))]
    except Exception:
        return []


class PlanRunner:
    def __init__(self, plan: PlanConfig, code_root: str, log=print):
        self.plan, self.code_root, self.log = plan, code_root, log
        self.runs = build_runs(plan)
        self.by_id = {r["id"]: r for r in self.runs}
        self.ledger = Ledger(plan)
        self.t0 = time.time()
        self.deadline = self.t0 + plan.session_hours * 3600
        self.paper_lock = threading.Lock()
        self.running: Dict[str, int] = {}
        self.stage_locks: Dict[str, threading.Lock] = {}
        self.prepared: Dict[str, Optional[str]] = {}         # stage -> None (ok) or the error of its preparation
        self.session_done, self.consecutive_failures, self.halt = 0, 0, ""
        for r in self.runs:                              # reconcile the ledger with the files (a copied output may be ahead of the ledger)
            st = self.ledger.get(r["id"])
            if os.path.exists(run_done_marker(plan, r)):
                st["status"] = "done"
            elif st["status"] in ("running", "done"):
                st["status"] = "interrupted"
        self.ledger.save()

    # ---------------------------------------------------------------- estimates
    def sec_per_step(self, stage: str) -> float:
        rates = {}
        for r in self.runs:
            st = self.ledger.state.get(r["id"], {})
            if st.get("status") == "done" and r["kind"] == "run" and st.get("train_steps"):
                rates.setdefault(r["stage"], []).append(st["seconds"] / max(st["train_steps"], 1))
        if stage in rates:
            return float(sorted(rates[stage])[len(rates[stage]) // 2])
        allr = sorted(x for v in rates.values() for x in v)
        return float(allr[len(allr) // 2]) if allr else PRIOR_SEC_PER_STEP

    def reserve_seconds(self, r: Dict[str, Any]) -> float:
        if r["kind"] == "adapt":
            return 60 * 60.0
        # evaluation time grows with the test set, sub-linearly (WSD, translation and the interpretability samples have fixed sizes)
        return 60.0 * EVAL_RESERVE_MIN[r["profile"]] * max(1.0, math.sqrt(r["test_lines"] / 3000.0))

    def estimate_hours(self, r: Dict[str, Any]) -> float:
        if r["kind"] == "adapt":
            return 2.0
        return (r["planned_steps"] * self.sec_per_step(r["stage"]) * 0.75 + self.reserve_seconds(r)) / 3600.0    # 0.75: early stopping (assumption)

    # ---------------------------------------------------------------- scheduling
    def next_run(self) -> Optional[Any]:
        blocked = False
        for r in self.runs:
            st = self.ledger.get(r["id"])
            if st["status"] in ("done", "skipped") or r["id"] in self.running or st["attempts"] >= self.plan.max_attempts:
                continue
            deps = [self.ledger.get(d)["status"] for d in r["deps"]]
            if any(d in ("skipped",) or (self.ledger.get(dd)["attempts"] >= self.plan.max_attempts and self.ledger.get(dd)["status"] != "done")
                   for d, dd in zip(deps, r["deps"])):
                st["status"] = "skipped"
                st["error"] = "a dependency failed"
                continue
            if all(d == "done" for d in deps):
                return r
            blocked = True
        return "WAIT" if (blocked and self.running) else None

    def worker(self, gpu: int) -> None:
        while True:
            if self.halt:
                return
            with self.ledger.lock:
                r = self.next_run()
                if r is None:
                    return
                if r != "WAIT":
                    left = self.deadline - time.time()
                    if left < self.reserve_seconds(r) + 30 * 60:
                        self.log(f"[gpu {gpu}] {left / 3600:.2f} h left in this session: not starting {r['id']} (needs its evaluation reserve "
                                 f"{self.reserve_seconds(r) / 60:.0f} min + 30 min of training)")
                        return
                    self.running[r["id"]] = gpu
                    st = self.ledger.get(r["id"])
                    st["status"], st["gpu"], st["started"] = "running", gpu, time.strftime("%Y-%m-%d %H:%M:%S")
                    st["sessions"] = st.get("sessions", 0) + 1
                    self.ledger.save()
            if r == "WAIT":
                if time.time() > self.deadline:
                    return
                time.sleep(60)
                continue
            if r["kind"] == "run":
                err = self.ensure_prepared(r)
                if err is not None:
                    self.record_failure(r, gpu, {"status": "failed", "error": f"data preparation failed: {err}"}, 0.0, "")
                    continue
            self.execute(r, gpu)

    # ---------------------------------------------------------------- data preparation and GPU check (separate processes)
    def ensure_prepared(self, r: Dict[str, Any]) -> Optional[str]:
        """Download and decode the corpora of r's stage ONCE, in a CPU-only process, before any of its runs starts: no two runs race for the
        same cache, and no training process ever decodes (or forks) next to a CUDA context."""
        stage = r["stage"]
        with self.ledger.lock:
            lock = self.stage_locks.setdefault(stage, threading.Lock())
        with lock:
            if stage not in self.prepared:
                self.log(f"[prepare] {stage}: downloading / decoding (once; cached for every run of this stage)")
                t = time.time()
                cfg = run_config(self.plan, r, self.n_workers)
                res, log_path, _ = launch(self.plan, self.code_root, "prepare", {"config": cfg.to_dict()}, os.path.join(self.plan.plan_dir, "prepare", stage),
                                          None, 6 * 3600)
                if res.get("status") == "done":
                    self.prepared[stage] = None
                    self.log(f"[prepare] {stage}: ready in {time.time() - t:.0f} s | {res.get('summary', '')}")
                else:
                    self.prepared[stage] = (res.get("error") or "")[-600:]
                    self.log(f"[prepare] {stage}: FAILED\n{_indent(tail(log_path))}")
            return self.prepared[stage]

    def gpu_check(self, gpus: List[int]) -> List[int]:
        """Train a few steps of the real model (fp16 autocast, GradScaler, CTC, cuDNN) on random tensors on every GPU, each in its own process:
        a broken GPU / CUDA / cuDNN setup is found in seconds and reported with its log, instead of failing every run."""
        out: Dict[int, Tuple[Dict[str, Any], str]] = {}

        def one(g: int) -> None:
            res, log_path, _ = launch(self.plan, self.code_root, "gpucheck", {}, os.path.join(self.plan.plan_dir, "gpucheck", str(g)), g, 900)
            out[g] = (res, log_path)
        ths = [threading.Thread(target=one, args=(g,)) for g in gpus]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        ok = []
        for g in gpus:
            res, log_path = out[g]
            if res.get("status") == "done":
                ok.append(g)
                self.log(f"[gpu {g}] check ok: {res.get('summary', '')}")
            else:
                self.log(f"[gpu {g}] check FAILED -- this GPU is not used: {res.get('error', '')[-300:]}\n{_indent(tail(log_path))}")
        if not ok:
            raise RuntimeError("no GPU passed the check (see the logs above, folder _plan/gpucheck): the environment cannot train")
        return ok

    def record_failure(self, r: Dict[str, Any], gpu: int, res: Dict[str, Any], wall: float, log_path: str) -> None:
        with self.ledger.lock:
            self.running.pop(r["id"], None)
            st = self.ledger.get(r["id"])
            st["status"], st["finished"] = "failed", time.strftime("%Y-%m-%d %H:%M:%S")
            st["attempts"] = st.get("attempts", 0) + 1
            st["error"] = res.get("error", "")[-4000:]
            self.ledger.save()
        self.after_result(r, gpu, res, wall, log_path)

    def after_result(self, r: Dict[str, Any], gpu: int, res: Dict[str, Any], wall: float, log_path: str) -> None:
        """Log the outcome (with the end of the run's log when it failed) and stop the session on a systematic failure."""
        msg = {"done": f"DONE  CER {100 * res.get('CER', float('nan')):.2f} %" if "CER" in res else "DONE",
               "interrupted": "INTERRUPTED by the session deadline (resumes next session)",
               "failed": f"FAILED (attempt {self.ledger.get(r['id'])['attempts']}/{self.plan.max_attempts}): {res.get('error', '')[-300:]}"}[res["status"]]
        self.log(f"[gpu {gpu}] {msg}  {r['id']}  [{wall / 3600:.2f} h]  | {self.progress()}")
        if res["status"] == "failed" and log_path:
            self.log(f"[gpu {gpu}] end of {log_path}:\n{_indent(tail(log_path))}")
        with self.ledger.lock:
            if res["status"] == "done":
                self.session_done, self.consecutive_failures = self.session_done + 1, 0
            elif res["status"] == "failed":
                self.consecutive_failures += 1
                if self.session_done == 0 and self.consecutive_failures >= self.plan.max_consecutive_failures and not self.halt:
                    self.halt = (f"{self.consecutive_failures} attempts failed in a row and no run has finished in this session: a systematic error. "
                                 f"The session stops instead of spending the plan; fix the cause shown above and run again (finished work is kept).")
                    self.log("[STOP] " + self.halt)

    def execute(self, r: Dict[str, Any], gpu: int) -> None:
        plan = self.plan
        run_dir = os.path.join(plan.work_dir, r["id"])
        os.makedirs(run_dir, exist_ok=True)
        stop_at = self.deadline - self.reserve_seconds(r)
        if r["kind"] == "run":
            cfg = run_config(plan, r, self.n_workers)
            cfg.train.deadline_unix = stop_at
            payload = {"config": cfg.to_dict()}
        else:
            payload = {"plan": dataclasses.asdict(plan), "run": r, "deadline_unix": stop_at}
        self.log(f"[gpu {gpu}] START {r['id']}  ({r['profile']} evaluation, ~{self.estimate_hours(r):.1f} h estimated)")
        t = time.time()
        res, log_path, _ = launch(plan, self.code_root, r["kind"], payload, run_dir, gpu, self.deadline + 15 * 60 - time.time())
        wall = time.time() - t
        if res["status"] == "failed":
            self.ledger.get(r["id"])["seconds"] = self.ledger.get(r["id"]).get("seconds", 0.0) + wall
            self.record_failure(r, gpu, res, wall, log_path)
            self.refresh_paper()
            return
        with self.ledger.lock:
            self.running.pop(r["id"], None)
            st = self.ledger.get(r["id"])
            st["seconds"] = st.get("seconds", 0.0) + wall
            st["status"] = res["status"]
            st["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
            for k in ("train_steps", "epochs_trained", "best_epoch", "CER"):
                if k in res:
                    st[k] = res[k]
            self.ledger.save()
        self.after_result(r, gpu, res, wall, log_path)
        if res["status"] == "done":
            self.prune(r)
            self.cleanup_caches()
        self.refresh_paper()

    # ---------------------------------------------------------------- hygiene
    def prune(self, r: Dict[str, Any]) -> None:
        if r["kind"] != "run":
            return
        ck = os.path.join(self.plan.work_dir, r["id"], "checkpoints")
        for p in glob.glob(os.path.join(ck, "last.pt")) + glob.glob(os.path.join(ck, "epoch_*.pt")):
            os.remove(p)
        keep = self.plan.keep_checkpoints == "all" or (self.plan.keep_checkpoints == "core" and r["tier"] == 0)
        if not keep and os.path.exists(os.path.join(ck, "best.pt")):
            os.remove(os.path.join(ck, "best.pt"))

    def cleanup_caches(self) -> None:
        if not self.plan.cleanup_caches:
            return
        with self.ledger.lock:
            needed = {k for r in self.runs if self.ledger.get(r["id"])["status"] not in ("done", "skipped") for k in r["sources"]}
        for key in {k for r in self.runs for k in r["sources"]} - needed:
            for d in (os.path.join(self.plan.cache_dir, "data", "hub", key), os.path.join(self.plan.cache_dir, "data", "hf", key)):
                shutil.rmtree(d, ignore_errors=True)
            for f in glob.glob(os.path.join(self.plan.cache_dir, "lines", f"{key}_*")):
                try:
                    os.remove(f)
                except OSError:
                    pass

    def persist_caches(self) -> None:
        """Copy the line-cache entries of the corpora that still have pending runs into the saved output (in plan order, within
        plan.persist_cache_gb); entries of finished corpora are removed from it."""
        src, dst = os.path.join(self.plan.cache_dir, "lines"), os.path.join(self.plan.work_dir, "_cache", "lines")
        if self.plan.persist_cache_gb <= 0 or not os.path.isdir(src):
            shutil.rmtree(dst, ignore_errors=True)
            return
        with self.ledger.lock:
            pending = [r for r in self.runs if self.ledger.get(r["id"])["status"] not in ("done", "skipped")]
        order = list(dict.fromkeys(k for r in pending for k in r["sources"]))
        used = sum(os.path.getsize(os.path.join(r_, f)) for r_, _d, fs in os.walk(self.plan.work_dir) if "_cache" not in r_.split(os.sep)
                   for f in fs if os.path.isfile(os.path.join(r_, f)))
        budget, chosen = min(self.plan.persist_cache_gb * 1e9, OUTPUT_LIMIT_GB * 1e9 - used), []     # the saved output must stay < 20 GB
        for key in order:
            files = [f for f in glob.glob(os.path.join(src, f"{key}_*")) if not f.endswith((".lock", ".tmp"))]
            size = sum(os.path.getsize(f) for f in files)
            if size > budget:
                self.log(f"line cache of '{key}' ({size / 1e9:.1f} GB) does not fit the {self.plan.persist_cache_gb:.0f} GB left: it will be rebuilt")
                break
            budget -= size
            chosen += files
        keep = {os.path.basename(f) for f in chosen}
        os.makedirs(dst, exist_ok=True)
        for f in os.listdir(dst):
            if f not in keep:
                os.remove(os.path.join(dst, f))
        for f in chosen:
            b = os.path.join(dst, os.path.basename(f))
            if not os.path.exists(b) or os.path.getsize(b) != os.path.getsize(f):
                shutil.copy2(f, b)
        if chosen:
            self.log(f"line caches saved for the next session: {len(chosen)} files, {sum(os.path.getsize(f) for f in chosen) / 1e9:.1f} GB")

    # ---------------------------------------------------------------- reporting
    def progress(self) -> str:
        c: Dict[str, int] = {}
        for r in self.runs:
            s = self.ledger.get(r["id"])["status"]
            c[s] = c.get(s, 0) + 1
        rem = sum(self.estimate_hours(r) for r in self.runs if self.ledger.get(r["id"])["status"] not in ("done", "skipped"))
        return (f"{c.get('done', 0)}/{len(self.runs)} done, {c.get('running', 0)} running, {c.get('interrupted', 0)} interrupted, "
                f"{c.get('failed', 0)} failed | ~{rem:.0f} GPU-h left")

    def refresh_paper(self) -> None:
        from .paper import build_paper
        with self.paper_lock:
            try:
                build_paper(self.plan, self.runs, self.ledger.state)
            except Exception:
                self.log("paper outputs could not be rebuilt:\n" + traceback.format_exc()[-1500:])

    def describe(self) -> str:
        rows = ["| stage | corpora | train lines | val | test | complexity | epochs (max) | steps/epoch | runs | est. GPU-h |", "|---|---|---|---|---|---|---|---|---|---|"]
        for stage, sources in stage_list(self.plan):
            b = stage_schedule(self.plan, stage, sources)
            rs = [r for r in self.runs if r["stage"] == stage]
            rows.append(f"| {stage} | {', '.join(sources) if len(sources) < 4 else f'{len(sources)} corpora'} | {b['train_lines']:,} | {b['val_lines']:,} | "
                        f"{b['test_lines']:,} | {b['complexity_factor']:.2f} | {b['epochs']} | {b['steps_per_epoch']:,} | {len(rs)} | "
                        f"{sum(self.estimate_hours(r) for r in rs):.0f} |")
        ad = [r for r in self.runs if r["kind"] == "adapt"]
        if ad:
            rows.append(f"| adaptation | {', '.join(r['target'] for r in ad)} | {self.plan.adaptation_lines:,} | - | {self.plan.adaptation_eval_lines:,} | - | - | - | "
                        f"{len(ad)} | {sum(self.estimate_hours(r) for r in ad):.0f} |")
        tot = sum(self.estimate_hours(r) for r in self.runs)
        rem = sum(self.estimate_hours(r) for r in self.runs if self.ledger.get(r["id"])["status"] not in ("done", "skipped"))
        rate = self.sec_per_step("iam")
        return ("\n".join(rows) + f"\n\n{len(self.runs)} runs; estimated {tot:.0f} GPU-hours in total, {rem:.0f} left "
                f"({'measured' if rate != PRIOR_SEC_PER_STEP else 'PRIOR'} {rate:.3f} s/step; early stopping assumed at 75 % of the budget).")

    def run(self) -> Dict[str, Any]:
        gpus = detect_gpus()
        if not gpus:
            raise RuntimeError("no GPU visible (nvidia-smi -L): the plan trains real models on real data and needs a GPU (Kaggle: GPU T4 x2)")
        gpus = self.gpu_check(gpus)
        self.n_workers = min(len(gpus), self.plan.max_parallel or len(gpus))
        self.log(f"GPUs: {len(gpus)} -> {self.n_workers} runs in parallel | session budget {self.plan.session_hours:.1f} h | work dir {self.plan.work_dir}")
        self.log(self.describe())
        if self.plan.dry_run:
            first = next((r for r in self.runs if r["kind"] == "run" and self.ledger.get(r["id"])["status"] not in ("done", "skipped")), None)
            err = self.ensure_prepared(first) if first is not None else None
            self.log(f"dry run finished: GPUs {gpus} ok, data of '{first['stage'] if first else '-'}' {'ready' if err is None else 'FAILED'}; "
                     "set dry_run=False to start the plan")
            return {"progress": self.progress(), "paper": os.path.join(self.plan.work_dir, "paper"), "dry_run": True}
        threads = [threading.Thread(target=self.worker, args=(g,), daemon=True) for g in gpus[:self.n_workers]]
        for k, th in enumerate(threads):
            th.start()
            time.sleep(20 if k == 0 else 0)          # stagger: the first run downloads the corpus, the second one then reads the shared caches
        for th in threads:
            th.join()
        if self.halt:
            self.log("[STOP] " + self.halt)
        self.refresh_paper()
        self.persist_caches()
        self.log("session finished | " + self.progress())
        return {"progress": self.progress(), "paper": os.path.join(self.plan.work_dir, "paper")}


# ==============================================================================
# restore a previous session's output (Kaggle: attach the previous version as input)
# ==============================================================================
def restore_previous(src: str, dst: str, log=print) -> int:
    if not src:
        return 0
    if os.path.isdir(os.path.join(src, os.path.basename(os.path.normpath(dst)))):
        src = os.path.join(src, os.path.basename(os.path.normpath(dst)))
    if not os.path.isdir(src):
        log(f"resume_from={src!r} is not a folder: nothing restored")
        return 0
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel.split(os.sep)[0] == "_cache":          # saved line caches go straight back to the cache folder (see run_plan)
            continue
        for fn in files:
            a, b = os.path.join(root, fn), os.path.join(dst, rel, fn)
            if not os.path.exists(b) or (fn == "state.json" and os.path.getmtime(a) > os.path.getmtime(b)):
                os.makedirs(os.path.dirname(b), exist_ok=True)
                shutil.copy2(a, b)
                n += 1
    log(f"restored {n} files from {src}")
    return n


def restore_caches(plan: PlanConfig, log=print) -> int:
    """Line caches saved by the previous session (in its output, or already in this work dir) -> the cache folder of this session."""
    dst = os.path.join(plan.cache_dir, "lines")
    found = []
    for base in ([plan.resume_from, os.path.join(plan.resume_from, os.path.basename(os.path.normpath(plan.work_dir)))] if plan.resume_from else []) \
            + [plan.work_dir]:
        d = os.path.join(base, "_cache", "lines")
        if os.path.isdir(d):
            found += [os.path.join(d, f) for f in os.listdir(d) if not f.endswith((".lock", ".tmp"))]
    n = 0
    os.makedirs(dst, exist_ok=True)
    for f in found:
        b = os.path.join(dst, os.path.basename(f))
        if not os.path.exists(b):
            shutil.copy2(f, b + ".tmp")
            os.replace(b + ".tmp", b)
            n += 1
    if n:
        log(f"restored {n} line-cache files from the previous session (no download / decoding for those corpora)")
    return n


def run_plan(plan: PlanConfig, code_root: Optional[str] = None, log=print) -> Dict[str, Any]:
    os.makedirs(plan.work_dir, exist_ok=True)
    os.makedirs(plan.cache_dir, exist_ok=True)
    restore_previous(plan.resume_from, plan.work_dir, log)
    restore_caches(plan, log)
    code_root = code_root or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(plan.plan_dir, exist_ok=True)
    runner = PlanRunner(plan, code_root, log)
    with open(os.path.join(plan.plan_dir, "plan.json"), "w") as f:
        json.dump(json_safe({"plan": dataclasses.asdict(plan), "runs": runner.runs}), f, indent=1)
    return runner.run()


# ==============================================================================
# subprocess entry points
# ==============================================================================
def _result_from_run(cfg: Config, out: Any) -> Dict[str, Any]:
    if isinstance(out, dict) and out.get("interrupted"):
        return {"status": "interrupted"}
    mp = os.path.join(cfg.infer_dir, "00_summary", "metrics.json")
    if not os.path.exists(mp):
        return {"status": "failed", "error": "evaluation produced no metrics.json"}
    res: Dict[str, Any] = {"status": "done", "CER": float(out.get("CER")) if isinstance(out, dict) and out.get("CER") is not None else None}
    try:
        with open(os.path.join(cfg.run_dir, "schedule.json")) as f:
            sch = json.load(f)
        with open(os.path.join(cfg.curve_dir, "history.json")) as f:
            H = json.load(f)
        ep = len(H.get("epoch", []))
        res.update({"epochs_trained": ep, "train_steps": ep * sch["steps_per_epoch"], "planned_epochs": sch["epochs"]})
        with open(mp) as f:
            res["best_epoch"] = (json.load(f).get("checkpoint") or {}).get("epoch")
    except Exception:
        pass
    return res


def _cmd_run(spec_path: str, res_path: str) -> None:
    from .main import main
    with open(spec_path) as f:
        cfg = Config.from_dict(json.load(f)["config"])
    try:
        res = _result_from_run(cfg, main(cfg))
    except Exception:
        res = {"status": "failed", "error": traceback.format_exc()}
    with open(res_path, "w") as f:
        json.dump(json_safe(res), f)


def export_adaptation_lines(plan: PlanConfig, target: str, folder: str, source_units: Sequence[str]) -> Dict[str, Any]:
    """Real target-corpus lines for the zero-shot adapter: `adaptation_lines` train lines (whole blocks) as PNG, the transcriptions of the first
    max(anchor_counts) of them as anchors.tsv, `adaptation_eval_lines` TEST lines as PNG + eval.tsv (measured only), and the target
    orthography's character inventory minus the source alphabet (prior knowledge of the written language, no line-level label)."""
    from .data import HandwritingDataModule
    from .text import grapheme_clusters
    os.makedirs(folder, exist_ok=True)
    c = Config()
    c.run.work_dir, c.data.sources, c.data.hf_cache_dir = plan.work_dir, (target,), os.path.join(plan.cache_dir, "data")
    dm = HandwritingDataModule(c)
    spec = DATASETS[target]
    srcs = dm._load_spec(spec)
    tr = HandwritingDataModule._select(len(srcs["train"]), plan.adaptation_lines, c.data.subset_seed, 20)
    te = HandwritingDataModule._select(len(srcs["test"]), plan.adaptation_eval_lines, c.data.subset_seed + 2, 20)
    n_anchor = max(plan.adaptation_anchor_counts) if plan.adaptation_anchor_counts else 0
    anchors, evals = [], []
    for k, j in enumerate(tr):
        name = f"train_{int(j):06d}.png"
        srcs["train"].image(int(j)).convert("L").save(os.path.join(folder, name))
        if k < n_anchor:
            anchors.append(f"{name}\t{srcs['train'].text(int(j))}")
    for j in te:
        name = f"test_{int(j):06d}.png"
        srcs["test"].image(int(j)).convert("L").save(os.path.join(folder, name))
        evals.append(f"{name}\t{srcs['test'].text(int(j))}")
    with open(os.path.join(folder, "anchors.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(anchors) + "\n")
    with open(os.path.join(folder, "eval.tsv"), "w", encoding="utf-8") as f:
        f.write("\n".join(evals) + "\n")
    inv = set()
    for j in range(len(srcs["train"])):
        inv.update(grapheme_clusters(srcs["train"].text(j)))
    charset = "".join(sorted(u for u in inv if u not in set(source_units) and not u.isspace()))
    return {"lines": len(tr), "eval_lines": len(te), "anchors": len(anchors), "charset": charset, "language": spec.language, "rtl": spec.rtl}


def _cmd_adapt(spec_path: str, res_path: str) -> None:
    from .adapt import run_adapt_curve
    from .utils import torch_load
    try:
        with open(spec_path) as f:
            payload = json.load(f)
        plan = PlanConfig(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in payload["plan"].items()})
        r = payload["run"]
        src_dir = os.path.join(plan.work_dir, r["source_run"])
        with open(os.path.join(src_dir, "config.json")) as f:
            cfg = Config.from_dict(json.load(f))
        st = torch_load(os.path.join(src_dir, "checkpoints", "best.pt"))
        out_dir = os.path.join(plan.work_dir, r["id"])
        folder = os.path.join(plan.cache_dir, "adaptation", r["target"])
        info = export_adaptation_lines(plan, r["target"], folder, st.get("units") or list(st["chars"]))
        cfg.run.mode = "adapt_curve"
        a = cfg.adapt
        a.checkpoint, a.lines_dir = "best", folder
        a.anchors_file, a.eval_file, a.out_dir = os.path.join(folder, "anchors.tsv"), os.path.join(folder, "eval.tsv"), out_dir
        a.charset, a.rtl, a.anchor_counts = info["charset"], info["rtl"], tuple(plan.adaptation_anchor_counts)
        a.target_languages = (plan.translation_target if info["language"] != plan.translation_target else "fr",)
        cfg.llm.backend, cfg.llm.model_name, cfg.llm.source_language = "causal", plan.llm_name, info["language"]
        with open(os.path.join(out_dir, "adaptation_setup.json"), "w") as f:
            json.dump(json_safe(info), f, indent=1, ensure_ascii=False)
        cfg.train.deadline_unix = float(payload.get("deadline_unix") or 0.0)
        out = run_adapt_curve(cfg)
        if isinstance(out, dict) and out.get("interrupted"):
            res = {"status": "interrupted"}
        else:
            res = {"status": "done"} if os.path.exists(os.path.join(out_dir, "adapt_curve.json")) else {"status": "failed", "error": "no adapt_curve.json"}
    except Exception:
        res = {"status": "failed", "error": traceback.format_exc()}
    with open(res_path, "w") as f:
        json.dump(json_safe(res), f)


def _cmd_prepare(spec_path: str, res_path: str) -> None:
    """Download and decode the corpora of a run configuration (no GPU): fills the manifests and line caches its runs will read."""
    from .data import HandwritingDataModule
    from .utils import setup_logging
    try:
        with open(spec_path) as f:
            cfg = Config.from_dict(json.load(f)["config"])
        cfg.run.name = os.path.join("_plan", "prepare", "_".join(cfg.data.sources) if len(cfg.data.sources) < 4 else "joint")
        cfg.make_dirs()
        setup_logging(cfg)
        dm = HandwritingDataModule(cfg).prepare()
        res = {"status": "done", "summary": f"train {len(dm.train_ds):,} | val {len(dm.val_ds):,} | test {len(dm.test_ds):,} lines | "
                                            f"alphabet {dm.codec.num_classes - 1} units"}
    except Exception:
        res = {"status": "failed", "error": traceback.format_exc()}
    with open(res_path, "w") as f:
        json.dump(json_safe(res), f)


def _cmd_gpucheck(spec_path: str, res_path: str) -> None:
    """A few optimisation steps of the real model on random tensors on the visible GPU: fp16 autocast, GradScaler, cuDNN convolutions, the
    recurrent / attention layers and the CTC loss.  This checks the environment (drivers, CUDA, cuDNN, memory), not the method."""
    import torch
    from .model import InkLinguaNet, ctc_per_sample
    from .style import NUM_DESC
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available in this process")
        dev = torch.device("cuda")
        cfg = Config()
        model = InkLinguaNet(cfg.model, 80, NUM_DESC, 8, cfg.data.img_height).to(dev).train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scaler = torch.amp.GradScaler("cuda")
        g = torch.Generator(device="cpu").manual_seed(0)
        t0 = time.time()
        for _ in range(3):
            x = torch.rand(8, 1, cfg.data.img_height, 1024, generator=g).to(dev)
            wd = torch.full((8,), 1024, dtype=torch.long, device=dev)
            tgt = torch.randint(1, 80, (8, 40), generator=g).to(dev)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = model(x, wd)
            loss = ctc_per_sample(out["log_probs"].float(), out["out_lens"], tgt, torch.full((8,), 40, dtype=torch.long, device=dev)).mean()
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        p = torch.cuda.get_device_properties(dev)
        res = {"status": "done", "summary": f"{p.name}, {p.total_memory / 2 ** 30:.0f} GB | torch {torch.__version__} | CUDA {torch.version.cuda} | "
                                            f"cuDNN {torch.backends.cudnn.version()} | 3 fp16 steps in {time.time() - t0:.1f} s, loss {float(loss):.2f}"}
    except Exception:
        res = {"status": "failed", "error": traceback.format_exc()}
    with open(res_path, "w") as f:
        json.dump(json_safe(res), f)


def _indent(text: str, pad: str = "    | ") -> str:
    return "\n".join(pad + line for line in text.rstrip().splitlines())


WORKER_COMMANDS = {"run": _cmd_run, "adapt": _cmd_adapt, "prepare": _cmd_prepare, "gpucheck": _cmd_gpucheck}


def worker_main(argv: Sequence[str]) -> None:
    """Entry point of a worker process:  <entry> {run|adapt|prepare|gpucheck} <spec.json> <result.json>."""
    WORKER_COMMANDS[argv[0]](argv[1], argv[2])


if __name__ == "__main__":
    worker_main(sys.argv[1:])
