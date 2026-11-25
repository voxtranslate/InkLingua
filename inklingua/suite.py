"""
The experiment suite: train + evaluate every variant for every seed and write the seed-paired comparison.  This is the experiment that
decides what a paper can claim: compare `full` with `labeled_only` (same labeled lines, no unlabeled data), with `ctc_only` (plain CTC
Transformer) and with `legacy_pl` (the v9 line-level pseudo-labels), and read every v10/v11 component off its own ablation.

suite_summary.json / .md hold, per variant: greedy CER, WER, CER with the validated post-hoc options, the system CER (incl. LLM rescoring if
adopted), mean +- std over seeds; against the baseline: the form-clustered paired bootstrap of CER(variant) - CER(baseline) per seed, the mean
over seeds and how many seeds are (significantly) better; and a component-contribution table "full minus each ablation".
"""
from __future__ import annotations

import copy
import csv
import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from .config import SUITE_PRESETS, Config
from .text import paired_bootstrap_delta
from .utils import json_safe, LOG


def _read_edits(path: str) -> Dict[int, Tuple[int, int, int]]:
    """line_index -> (edits, reference length, edits after the validated post-hoc options) from a predictions.csv."""
    out: Dict[int, Tuple[int, int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ed, rl = int(r["edits"]), int(r["ref_len"])
            edp = int(round(float(r["CER_posthoc"]) * max(rl, 1))) if r.get("CER_posthoc") not in (None, "") else ed
            out[int(r["line_index"])] = (ed, rl, edp)
    return out


def _read_metrics(pred_path: str) -> Dict[str, Any]:
    m = os.path.join(os.path.dirname(os.path.dirname(pred_path)), "00_summary", "metrics.json")
    if not os.path.exists(m):
        m = os.path.join(os.path.dirname(pred_path), "metrics.json")
    try:
        with open(m) as f:
            return json.load(f)
    except Exception:
        return {}


def summarise_suite(runs: Dict[str, Dict[int, str]], baseline: str = "labeled_only", n_boot: int = 1000, block: int = 20) -> Dict[str, Any]:
    """runs[variant][seed] = path of that run's predictions.csv (test split)."""
    edits = {v: {s: _read_edits(p) for s, p in d.items()} for v, d in runs.items()}
    mets = {v: {s: _read_metrics(p) for s, p in d.items()} for v, d in runs.items()}
    out: Dict[str, Any] = {"baseline": baseline, "variants": {}, "paired_vs_baseline": {}, "contribution_vs_full": {}}

    def agg(vals):
        a = np.array([v for v in vals if v is not None and np.isfinite(v)], np.float64)
        return {"mean": float(a.mean()) if len(a) else float("nan"), "std": float(a.std(ddof=1)) if len(a) > 1 else float("nan"), "n": int(len(a))}

    for v, d in edits.items():
        cers = {s: sum(e for e, _, _ in m.values()) / max(sum(l for _, l, _ in m.values()), 1) for s, m in d.items()}
        cerp = {s: sum(e for _, _, e in m.values()) / max(sum(l for _, l, _ in m.values()), 1) for s, m in d.items()}
        wers = {s: mets[v][s].get("WER") for s in d}
        sysc = {s: mets[v][s].get("CER_system_final", mets[v][s].get("CER_final_with_validated_posthoc")) for s in d}
        out["variants"][v] = {"CER_per_seed": cers, "CER": agg(cers.values()), "CER_posthoc_per_seed": cerp, "CER_posthoc": agg(cerp.values()),
                              "WER_per_seed": wers, "WER": agg(wers.values()), "CER_system_per_seed": sysc, "CER_system": agg(sysc.values()),
                              "CER_mean": agg(cers.values())["mean"], "CER_std": agg(cers.values())["std"], "n_seeds": len(cers)}

    def paired(v, ref):
        per, deltas = {}, []
        for s, m in edits[v].items():
            b = edits[ref].get(s)
            if b is None:
                continue
            ids = sorted(set(m) & set(b))
            ea, eb = np.array([m[i][0] for i in ids], np.float64), np.array([b[i][0] for i in ids], np.float64)
            rl = np.array([m[i][1] for i in ids], np.float64)
            per[s] = paired_bootstrap_delta(ea, eb, rl, n_boot, s, groups=np.asarray(ids) // max(1, int(block)))
            deltas.append(per[s]["delta"])
        if not deltas:
            return None
        return {"per_seed": per, "mean_delta": float(np.mean(deltas)), "seeds_better": int(sum(d_ < 0 for d_ in deltas)), "n_seeds": len(deltas),
                "seeds_significantly_better": int(sum(p["hi"] < 0 for p in per.values())), "seeds_significantly_worse": int(sum(p["lo"] > 0 for p in per.values()))}

    if baseline in edits:
        for v in edits:
            if v != baseline:
                r = paired(v, baseline)
                if r:
                    out["paired_vs_baseline"][v] = r
    if "full" in edits:
        for v in edits:
            if v != "full" and (v.startswith("no_") or v in ("legacy_pl", "ctc_only")):
                r = paired(v, "full")
                if r:
                    out["contribution_vs_full"][v] = {**r, "reading": "positive mean_delta = removing the component HURTS (the component helps)"}
    return out


def suite_markdown(summ: Dict[str, Any]) -> str:
    rows = ["| variant | test CER (mean +- std) | WER | CER + validated post-hoc | system CER | per seed (CER) | vs baseline: mean delta | seeds better / significant |",
            "|---|---|---|---|---|---|---|---|"]
    f = lambda a: ("n/a" if not np.isfinite(a["mean"]) else f"{a['mean'] * 100:.2f}" + (f" +- {a['std'] * 100:.2f}" if np.isfinite(a["std"]) else "") + " %")  # noqa: E731
    for v, d in summ["variants"].items():
        pv = summ["paired_vs_baseline"].get(v)
        ps = ", ".join(f"{s}: {c * 100:.2f}" for s, c in d["CER_per_seed"].items())
        rows.append(f"| {v} | {f(d['CER'])} | {f(d['WER'])} | {f(d['CER_posthoc'])} | {f(d['CER_system'])} | {ps} | "
                    + (f"{pv['mean_delta'] * 100:+.2f} pt | {pv['seeds_better']}/{pv['n_seeds']} better, {pv['seeds_significantly_better']} significant |"
                       if pv else "(baseline) | |"))
    md = "\n".join(rows) + f"\n\nbaseline = `{summ['baseline']}`; negative delta = better than the baseline; 'significant' = the 95% form-clustered paired-bootstrap interval excludes 0.\n"
    if summ.get("contribution_vs_full"):
        md += "\n### Component contributions (ablation minus full; positive = the removed component helps)\n\n| removed | mean delta CER | seeds where removal hurts / significant |\n|---|---|---|\n"
        for v, r in summ["contribution_vs_full"].items():
            md += f"| {v} | {r['mean_delta'] * 100:+.2f} pt | {r['n_seeds'] - r['seeds_better']}/{r['n_seeds']} , {r['seeds_significantly_worse']} significant |\n"
    return md


def suite_variants(cfg: Config) -> Tuple[str, ...]:
    if cfg.run.suite_preset == "custom":
        return tuple(cfg.run.suite_variants)
    if cfg.run.suite_preset not in SUITE_PRESETS:
        raise ValueError(f"unknown suite preset '{cfg.run.suite_preset}' (choose from {sorted(SUITE_PRESETS)} or 'custom')")
    return SUITE_PRESETS[cfg.run.suite_preset]


def _member_pred_path(c: Config) -> str:
    return os.path.join(c.infer_dir, "01_recognition", "predictions.csv")


def run_suite(cfg: Optional[Config] = None, train: bool = True) -> Dict[str, Any]:
    """Every run has its own directory and resumes if interrupted, so the suite can be re-launched until it is complete."""
    from .main import main
    cfg = cfg or Config()
    base = copy.deepcopy(cfg)
    base.run.mode = "single"
    runs: Dict[str, Dict[int, str]] = {}
    for v in suite_variants(cfg):
        for s in cfg.run.suite_seeds:
            c = Config.make_variant(v, base).with_seed(s)
            c.run.mode = "single"
            p = _member_pred_path(c)
            if train:
                LOG.info("=== suite: variant '%s', seed %d  (%s)", v, s, c.run_name)
                m = main(c)
                if m is None or not os.path.exists(p):
                    continue
            elif not os.path.exists(p):
                LOG.warning("suite_summary: no predictions for variant '%s' seed %d (%s)", v, s, p)
                continue
            runs.setdefault(v, {})[s] = p
    out_dir = os.path.join(cfg.run.work_dir, f"{cfg.run_name}_suite")
    os.makedirs(out_dir, exist_ok=True)
    baseline = "labeled_only" if "labeled_only" in runs else next(iter(runs), "labeled_only")
    summ = summarise_suite(runs, baseline, cfg.infer.bootstrap, cfg.infer.bootstrap_block)
    summ["variants_requested"] = list(suite_variants(cfg))
    summ["seeds"] = list(cfg.run.suite_seeds)
    with open(os.path.join(out_dir, "suite_summary.json"), "w") as f:
        json.dump(json_safe(summ), f, indent=2, allow_nan=False)
    md = suite_markdown(summ)
    with open(os.path.join(out_dir, "suite_summary.md"), "w") as f:
        f.write(md)
    LOG.info("suite summary (%s):\n%s", out_dir, md)
    return summ


def run_seeds(cfg: Optional[Config] = None, seeds=(1337, 1338, 1339)) -> Dict[str, Any]:
    """Run the full experiment for several seeds and report mean +- std of the test CER / WER."""
    from .main import main
    cfg = cfg or Config()
    rows = []
    for s in seeds:
        m = main(cfg.with_seed(s))
        if m:
            rows.append({"seed": s, "CER": m["CER"], "WER": m["WER"]})
    summ: Dict[str, Any] = {"runs": rows}
    for k in ("CER", "WER"):
        v = np.array([r[k] for r in rows], np.float64)
        if len(v):
            summ[k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else float("nan")}
    path = os.path.join(cfg.run.work_dir, f"{cfg.run_name}_seeds.json")
    os.makedirs(cfg.run.work_dir, exist_ok=True)
    with open(path, "w") as f:
        json.dump(json_safe(summ), f, indent=2, allow_nan=False)
    LOG.info("multi-seed summary: %s  (written to %s)", summ.get("CER"), path)
    return summ
