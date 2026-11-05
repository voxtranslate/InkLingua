"""
Paper outputs of the experiment plan, rebuilt from the run files after every finished run (work_dir/paper/):

    RESULTS.md                 every table below in one document (with the statistics used)
    tables/*.md|.tex|.csv      T1 main results, T2 significance of the main claim, T3 ablations, T4 post-hoc decomposition, T5 label budget,
                               T6 joint vs per-corpus training, T7 zero-shot adaptation, T8 translation, T9 uncertainty signals, T10 compute
    figures/*.png              label-budget curves, ablation heatmap, post-hoc waterfall
    results_long.csv           one row per finished run (every number the tables are built from)
    REPRODUCIBILITY.md         datasets and pinned commits, caps, decontamination, schedule rule and resulting budgets, environment, code hash
    STATUS.md                  what is done / pending / interrupted / failed, measured GPU-hours, failures with their error

Statistics.  A comparison of two variants on one corpus pools the seeds: in each bootstrap replicate, the 20-line blocks of the test set are
resampled independently for every seed (same blocks for both variants: paired), the CER difference is computed per seed and averaged over
seeds.  The 95 % percentile interval and a two-sided p-value (proportion of replicates on the other side of zero, x2) are reported; families of
tests (the corpora of the main claim; the ablations of one corpus) are Holm-corrected.  Across corpora, the per-(corpus, seed) differences
are summarised with an exact two-sided sign test.  Cells with fewer seeds than planned are marked.
"""
from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .datasets_registry import DATASETS
from .utils import json_safe

NO_WORD_SPACE_SCRIPTS = {"Han"}
VARIANT_LABEL = {"labeled_only": "labeled only", "ctc_only": "plain CTC", "legacy_pl": "line-level PL (v9)", "full": "InkLingua (full)",
                 "full_supervised": "100 % labels"}
ABLATION_LABEL = {"no_frag": "- FRAG-RC", "no_strata": "- style strata (one global threshold)", "no_sharpen": "- sharpening (plain distillation)",
                  "no_mixline": "- MixLine", "no_style_task": "- style task + FiLM", "no_ses": "- style-equivariant targets", "no_vq": "- VQ bottleneck",
                  "no_film": "- FiLM", "no_confusion": "- leakage penalty + decorrelation", "no_frame_cons": "- frame consistency",
                  "no_mask": "- masked-code modelling", "no_aux_ctc": "- inter-CTC", "no_v9_reg": "- layer drop + CTC label smoothing",
                  "no_hscale": "- horizontal-stretch augmentation"}


# ---------------------------------------------------------------- reading a run
def _run_dir(work_dir: str, rid: str) -> str:
    return os.path.join(work_dir, rid)


def _load_json(p: str) -> Optional[Any]:
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_edits(path: str) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["line_index"])] = (int(r["edits"]), int(r["ref_len"]))
    return out


def read_corpus_column(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[int(r["line_index"])] = r.get("corpus", "")
    return out


def collect(plan, runs: Sequence[Dict[str, Any]], state: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for r in runs:
        if r["kind"] != "run" or state.get(r["id"], {}).get("status") != "done":
            continue
        d = _run_dir(plan.work_dir, r["id"])
        M = _load_json(os.path.join(d, "inference", "test", "00_summary", "metrics.json")) or {}
        if "CER" not in M:
            continue
        H = _load_json(os.path.join(d, "curves", "history.json")) or {}
        S = _load_json(os.path.join(d, "schedule.json")) or {}
        ph = M.get("posthoc") or {}
        rs = M.get("llm_rescoring") or {}
        rows.append({**{k: r[k] for k in ("id", "stage", "variant", "seed", "lf", "tier", "profile")},
                     "CER": M["CER"], "CER_lo": (M.get("CER_ci95") or [None, None])[0], "CER_hi": (M.get("CER_ci95") or [None, None])[1],
                     "WER": M.get("WER"), "exact": M.get("exact_line_accuracy"), "CER_posthoc": M.get("CER_final_with_validated_posthoc"),
                     "CER_system": M.get("CER_system_final"), "CER_beam": rs.get("CER_beam_top1"), "CER_rescored": rs.get("CER_rescored"),
                     "CER_oracle_nbest": rs.get("CER_oracle_nbest"), "llm_adopted": rs.get("adopted_on_validation"),
                     "n_test": M.get("n_lines"), "epochs_planned": S.get("epochs"), "epochs_trained": len(H.get("epoch", [])),
                     "best_epoch": (M.get("checkpoint") or {}).get("epoch"), "gpu_hours": state[r["id"]].get("seconds", 0.0) / 3600.0,
                     "steps_per_epoch": S.get("steps_per_epoch"), "complexity": S.get("complexity_factor"),
                     "posthoc": {k: {"test_CER": (ph.get(k) or {}).get("test_CER"), "adopted": (ph.get(k) or {}).get("adopted")}
                                 for k in ("view_ensemble", "tta", "wsd", "space_bonus") if isinstance(ph.get(k), dict)},
                     "posthoc_base": ph.get("test_CER_base"), "per_corpus": M.get("per_corpus"), "translation": M.get("translation"),
                     "interpret": (M.get("interpretability") or {}).get("uncertainty_decomposition"),
                     "pred_path": os.path.join(d, "inference", "test", "01_recognition", "predictions.csv"), "provenance": M.get("provenance")})
    return rows


# ---------------------------------------------------------------- statistics
def pooled_paired(pairs: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], n_boot: int = 2000, seed: int = 0) -> Dict[str, float]:
    """pairs: per seed (edits_a, edits_b, ref_len, groups) on the SAME lines.  Mean over seeds of CER(a) - CER(b), percentile CI, two-sided p."""
    if not pairs:
        nan = float("nan")
        return {"delta": nan, "lo": nan, "hi": nan, "p": nan, "n_seeds": 0}
    rng = np.random.RandomState(seed)
    deltas, boots = [], np.zeros(n_boot)
    for ea, eb, rl, g in pairs:
        _, gi = np.unique(g, return_inverse=True)
        G = gi.max() + 1
        Ea, Eb, L = (np.bincount(gi, weights=np.asarray(x, np.float64), minlength=G) for x in (ea, eb, rl))
        deltas.append((Ea.sum() - Eb.sum()) / max(L.sum(), 1.0))
        idx = rng.randint(0, G, size=(n_boot, G))
        boots += (Ea[idx].sum(1) - Eb[idx].sum(1)) / np.maximum(L[idx].sum(1), 1.0)
    boots /= len(pairs)
    p = min(1.0, 2.0 * min(((boots <= 0).sum() + 1) / (n_boot + 1), ((boots >= 0).sum() + 1) / (n_boot + 1)))
    return {"delta": float(np.mean(deltas)), "lo": float(np.percentile(boots, 2.5)), "hi": float(np.percentile(boots, 97.5)), "p": float(p),
            "n_seeds": len(pairs), "per_seed": [float(x) for x in deltas]}


def holm(pvals: Sequence[float]) -> List[float]:
    p = np.asarray([1.0 if not np.isfinite(x) else x for x in pvals], np.float64)
    m = len(p)
    o = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, (m - k) * p[i])
        adj[i] = min(1.0, run)
    return adj.tolist()


def sign_test(deltas: Sequence[float]) -> Dict[str, float]:
    d = [x for x in deltas if np.isfinite(x) and abs(x) > 1e-12]
    n, k = len(d), sum(1 for x in d if x < 0)
    if n == 0:
        return {"n": 0, "better": 0, "p": float("nan")}
    tail = sum(math.comb(n, j) for j in range(0, min(k, n - k) + 1)) / 2 ** n
    return {"n": n, "better": k, "p": float(min(1.0, 2 * tail))}


def _pairs(rows_a: List[Dict[str, Any]], rows_b: List[Dict[str, Any]], line_filter=None, offsets=None):
    """Seed-matched (edits_a, edits_b, ref_len, groups) on the common test lines."""
    by_b = {r["seed"]: r for r in rows_b}
    out = []
    for ra in rows_a:
        rb = by_b.get(ra["seed"])
        if rb is None or not os.path.exists(ra["pred_path"]) or not os.path.exists(rb["pred_path"]):
            continue
        A, B = read_edits(ra["pred_path"]), read_edits(rb["pred_path"])
        if offsets is not None:                 # map the joint run's concatenated indices onto the per-corpus indices
            A = {i - offsets[0]: v for i, v in A.items() if line_filter(i)}
        ids = sorted(set(A) & set(B))
        if not ids:
            continue
        out.append((np.array([A[i][0] for i in ids]), np.array([B[i][0] for i in ids]), np.array([B[i][1] for i in ids], np.float64),
                    np.asarray(ids) // 20))
    return out


# ---------------------------------------------------------------- formatting
def _pct(x, d=2):
    return "n/a" if x is None or not np.isfinite(x) else f"{100 * x:.{d}f}"


def _ms(vals: Sequence[float], planned: int, d: int = 2) -> str:
    v = [x for x in vals if x is not None and np.isfinite(x)]
    if not v:
        return "-"
    s = f"{100 * np.mean(v):.{d}f}" + (f" ± {100 * np.std(v, ddof=1):.{d}f}" if len(v) > 1 else "")
    return s + ("" if len(v) >= planned else f" ({len(v)}/{planned})")


def _delta_str(t: Dict[str, float], p_adj: Optional[float] = None) -> str:
    if not np.isfinite(t.get("delta", float("nan"))):
        return "-"
    p = t["p"] if p_adj is None else p_adj
    star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
    return f"{100 * t['delta']:+.2f} [{100 * t['lo']:+.2f}, {100 * t['hi']:+.2f}]{star}"


def _write_table(out_dir: str, name: str, title: str, header: List[str], rows: List[List[str]], note: str = "") -> str:
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)
    md = f"### {title}\n\n| " + " | ".join(header) + " |\n|" + "|".join("---" for _ in header) + "|\n" + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"
    if note:
        md += f"\n{note}\n"
    with open(os.path.join(out_dir, "tables", name + ".md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "tables", name + ".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    esc = lambda s: str(s).replace("\\", "\\textbackslash{}").replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#").replace("±", "$\\pm$")  # noqa: E731
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n\\caption{" + esc(title) + "}\n\\label{tab:" + name + "}\n\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\n\\toprule\n"
           + " & ".join(esc(h) for h in header) + " \\\\\n\\midrule\n" + "\n".join(" & ".join(esc(c) for c in r) + " \\\\" for r in rows)
           + "\n\\bottomrule\n\\end{tabular}\n" + (("\\par\\smallskip\\footnotesize " + esc(note) + "\n") if note else "") + "\\end{table}\n")
    with open(os.path.join(out_dir, "tables", name + ".tex"), "w", encoding="utf-8") as f:
        f.write(tex)
    return md


# ---------------------------------------------------------------- the paper
def build_paper(plan, runs: Sequence[Dict[str, Any]], state: Dict[str, Dict[str, Any]]) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = os.path.join(plan.work_dir, "paper")
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    rows = collect(plan, runs, state)
    S = len(plan.seeds)
    main_lf = plan.labeled_fraction
    corpora = [c for c in plan.corpora if DATASETS[c].test_split]
    sel = lambda st, v, lf: [r for r in rows if r["stage"] == st and r["variant"] == v and abs(r["lf"] - lf) < 1e-9]  # noqa: E731
    lf_of = lambda v: 1.0 if v == "full_supervised" else main_lf  # noqa: E731
    docs: List[str] = [f"# InkLingua - results of the experiment plan\n\nBuilt {time.strftime('%Y-%m-%d %H:%M')} from {len(rows)} finished runs. "
                       "CER / WER in %, mean ± standard deviation over seeds; '(k/n)' = only k of the n planned seeds are finished. "
                       "Differences: pooled seed-paired, form-clustered bootstrap (20-line blocks), 95 % interval; * p < 0.05, ** p < 0.01 after Holm "
                       "correction within the table's family.\n"]

    # ---- long table
    keys = ["id", "stage", "variant", "seed", "lf", "profile", "CER", "CER_lo", "CER_hi", "WER", "exact", "CER_posthoc", "CER_system", "CER_beam",
            "CER_rescored", "CER_oracle_nbest", "llm_adopted", "n_test", "epochs_planned", "epochs_trained", "best_epoch", "steps_per_epoch", "complexity", "gpu_hours"]
    with open(os.path.join(out, "results_long.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k) for k in keys])

    # ---- T1 main results
    t1 = []
    for c in corpora:
        t1.append([c] + [_ms([r["CER"] for r in sel(c, v, lf_of(v))], S) for v in plan.core_variants]
                  + [_ms([r["CER_system"] for r in sel(c, "full", main_lf)], S)])
    docs.append(_write_table(out, "T1_main_cer", f"Test CER (%) at {int(100 * main_lf)} % labels", ["corpus"] + [VARIANT_LABEL.get(v, v) for v in plan.core_variants]
                             + ["full + validated options"], t1, "Greedy decoding unless stated; the last column adds every post-hoc option adopted on val_gate "
                             "(view ensemble / AdaBN / writer self-distillation / word-gap bonus) and the LLM N-best rescoring if adopted (an external language model)."))
    t1w = [[c] + [_ms([r["WER"] for r in sel(c, v, lf_of(v))], S) for v in plan.core_variants] for c in corpora if DATASETS[c].script not in NO_WORD_SPACE_SCRIPTS]
    docs.append(_write_table(out, "T1b_main_wer", f"Test WER (%) at {int(100 * main_lf)} % labels", ["corpus"] + [VARIANT_LABEL.get(v, v) for v in plan.core_variants], t1w,
                             "Scripts written without word spaces (Han) are omitted."))

    # ---- T2 significance of the main claim
    t2, claims, sign_deltas = [], [], []
    for c in corpora:
        a = pooled_paired(_pairs(sel(c, "full", main_lf), sel(c, "labeled_only", main_lf)))
        b = pooled_paired(_pairs(sel(c, "full", main_lf), sel(c, "legacy_pl", main_lf)))
        claims.append((c, a, b))
        sign_deltas += a.get("per_seed", [])
    ha, hb = holm([x[1]["p"] for x in claims]), holm([x[2]["p"] for x in claims])
    for (c, a, b), pa, pb in zip(claims, ha, hb):
        t2.append([c, str(a["n_seeds"]), _delta_str(a, pa), f"{pa:.3g}" if np.isfinite(pa) and a["n_seeds"] else "-", _delta_str(b, pb),
                   f"{pb:.3g}" if np.isfinite(pb) and b["n_seeds"] else "-"])
    st = sign_test(sign_deltas)
    docs.append(_write_table(out, "T2_main_claim", "Does the unlabeled data help? CER difference (points), InkLingua minus baseline",
                             ["corpus", "seeds", "Δ vs labeled only", "p (Holm)", "Δ vs line-level PL", "p (Holm)"], t2,
                             f"Negative = InkLingua is better.  Across all (corpus, seed) pairs: InkLingua better in {st['better']} of {st['n']} "
                             f"(exact sign test p = {(format(st['p'], '.3g') if st['n'] and np.isfinite(st['p']) else '-')}).  labeled_only also trains on the 10 % of the label budget that InkLingua holds out for calibration."))

    # ---- T3 ablations
    abl_corpora = [c for c in corpora if (not plan.ablation_corpora or c in plan.ablation_corpora)]
    if plan.ablations and abl_corpora:
        hdr, t3, heat = ["ablation"] + abl_corpora, [], np.full((len(plan.ablations), len(abl_corpora)), np.nan)
        tests = {c: [pooled_paired(_pairs(sel(c, v, main_lf), sel(c, "full", main_lf))) for v in plan.ablations] for c in abl_corpora}
        adj = {c: holm([t["p"] for t in tests[c]]) for c in abl_corpora}
        for vi, v in enumerate(plan.ablations):
            t3.append([ABLATION_LABEL.get(v, v)] + [_delta_str(tests[c][vi], adj[c][vi]) if tests[c][vi]["n_seeds"] else "-" for c in abl_corpora])
            for ci, c in enumerate(abl_corpora):
                heat[vi, ci] = 100 * tests[c][vi]["delta"] if tests[c][vi]["n_seeds"] else np.nan
        docs.append(_write_table(out, "T3_ablations", "Ablations: CER difference (points), ablation minus InkLingua (full)", hdr, t3,
                                 "Positive = removing the component hurts (the component helps).  Holm correction over the ablations of each corpus."))
        if np.isfinite(heat).any():
            fig, ax = plt.subplots(figsize=(1.0 + 0.9 * len(abl_corpora), 0.5 + 0.38 * len(plan.ablations)))
            vmax = np.nanmax(np.abs(heat)) or 1.0
            im = ax.imshow(heat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(abl_corpora))), ax.set_xticklabels(abl_corpora, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(len(plan.ablations))), ax.set_yticklabels([ABLATION_LABEL.get(v, v) for v in plan.ablations], fontsize=8)
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    if np.isfinite(heat[i, j]):
                        ax.text(j, i, f"{heat[i, j]:+.2f}", ha="center", va="center", fontsize=6)
            fig.colorbar(im, ax=ax, label="Δ CER (points); red = component helps")
            fig.tight_layout(), fig.savefig(os.path.join(out, "figures", "ablation_heatmap.png"), dpi=170), plt.close(fig)

    # ---- T4 post-hoc decomposition (full runs)
    t4 = []
    for c in corpora:
        rs = sel(c, "full", main_lf)
        if not rs:
            continue
        def m(f):
            v = [f(r) for r in rs]
            v = [x for x in v if x is not None and np.isfinite(x)]
            return _pct(np.mean(v)) if v else "-"
        def adopt(k):
            v = [(r["posthoc"].get(k) or {}).get("adopted") for r in rs]
            return f"{sum(1 for x in v if x)}/{sum(1 for x in v if x is not None)}"
        t4.append([c, m(lambda r: r["posthoc_base"]), f"{m(lambda r: (r['posthoc'].get('view_ensemble') or {}).get('test_CER'))} ({adopt('view_ensemble')})",
                   f"{m(lambda r: (r['posthoc'].get('tta') or {}).get('test_CER'))} ({adopt('tta')})",
                   f"{m(lambda r: (r['posthoc'].get('wsd') or {}).get('test_CER'))} ({adopt('wsd')})",
                   f"{m(lambda r: (r['posthoc'].get('space_bonus') or {}).get('test_CER'))} ({adopt('space_bonus')})", m(lambda r: r["CER_posthoc"]),
                   m(lambda r: r["CER_beam"]), f"{m(lambda r: r['CER_rescored'])} ({sum(1 for r in rs if r['llm_adopted'])}/{sum(1 for r in rs if r['llm_adopted'] is not None)})",
                   m(lambda r: r["CER_oracle_nbest"])])
    if t4:
        docs.append(_write_table(out, "T4_posthoc", "Validated post-hoc options on the InkLingua model: test CER (%) of each option (adopted in k/n seeds)",
                                 ["corpus", "greedy", "view ensemble", "style AdaBN", "writer self-distillation", "word-gap bonus", "with adopted options",
                                  "beam top-1", "LLM rescoring", "oracle N-best"], t4,
                                 "An option's column is its test CER when applied (reported even when not adopted); it is applied in the system only when its "
                                 "Bonferroni-adjusted val_gate gain excludes zero.  This replaces separate no_wsd / no_posthoc / no_llm runs."))

    # ---- T5 label budget
    lfs = sorted(set(list(plan.label_sweep) + [main_lf]))
    t5, fig_ok = [], False
    if plan.label_sweep:
        ncol = 4
        nrow = int(math.ceil(len(corpora) / ncol)) or 1
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.0 * nrow), squeeze=False)
        for k, c in enumerate(corpora):
            row = [c]
            ax = axes[k // ncol][k % ncol]
            for v in plan.sweep_variants:
                ys, es = [], []
                for lf in lfs:
                    vals = [r["CER"] for r in sel(c, v, lf)]
                    row.append(_ms(vals, S))
                    ys.append(100 * np.mean(vals) if vals else np.nan), es.append(100 * np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                if np.isfinite(ys).any():
                    ax.errorbar([100 * x for x in lfs], ys, yerr=es, marker="o", capsize=3, label=VARIANT_LABEL.get(v, v))
                    fig_ok = True
            fs = [r["CER"] for r in sel(c, "full_supervised", 1.0)]
            row.append(_ms(fs, S))
            if fs:
                ax.axhline(100 * np.mean(fs), color="k", ls=":", lw=1, label="100 % labels")
            ax.set_xscale("log"), ax.set_title(c, fontsize=9), ax.set_xlabel("labeled training lines (%)"), ax.set_ylabel("test CER (%)")
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=6)
            t5.append(row)
        for k in range(len(corpora), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        fig.tight_layout()
        if fig_ok:
            fig.savefig(os.path.join(out, "figures", "label_budget.png"), dpi=170)
        plt.close(fig)
        docs.append(_write_table(out, "T5_label_budget", "Test CER (%) as a function of the label budget",
                                 ["corpus"] + [f"{VARIANT_LABEL.get(v, v)} {int(100 * lf)} %" for v in plan.sweep_variants for lf in lfs] + ["100 % labels"], t5))

    # ---- T6 joint vs per-corpus
    if plan.joint:
        jr = sel("joint", "full", main_lf)
        t6 = []
        for c in corpora:
            if c not in plan.joint_corpora:
                continue
            per = sel(c, "full", main_lf)
            jv = [((r.get("per_corpus") or {}).get(c) or {}).get("CER") for r in jr]
            pairs = []
            for rj in jr:
                rc = next((x for x in per if x["seed"] == rj["seed"]), None)
                if rc is None or not os.path.exists(rj["pred_path"]):
                    continue
                corp = read_corpus_column(rj["pred_path"])
                ids = [i for i, k in corp.items() if k == c]
                if not ids:
                    continue
                off = min(ids)
                pairs += _pairs([rj], [rc], line_filter=lambda i, ids=set(ids): i in ids, offsets=(off,))
            t = pooled_paired(pairs)
            t6.append([c, _ms([r["CER"] for r in per], S), _ms(jv, S), _delta_str(t) if t["n_seeds"] else "-"])
        if t6:
            docs.append(_write_table(out, "T6_joint", "One model for all corpora vs one model per corpus: test CER (%) per corpus (InkLingua, full)",
                                     ["corpus", "per-corpus model", "joint model", "Δ joint - per-corpus"], t6,
                                     f"The joint model is trained on capped subsets ({plan.joint_caps}) of every corpus with cross-split decontamination; "
                                     "its training lines per corpus are therefore fewer than the per-corpus model's for the large corpora."))

    # ---- T7 adaptation
    t7 = []
    for r in runs:
        if r["kind"] != "adapt" or state.get(r["id"], {}).get("status") != "done":
            continue
        cur = _load_json(os.path.join(plan.work_dir, r["id"], "adapt_curve.json")) or {}
        for k in sorted(cur, key=lambda x: int(x)):
            s0, s1 = cur[k].get("start") or {}, cur[k].get("end") or {}
            t7.append([f"{plan.adaptation_source} -> {r['target']}", str(k), _pct(s0.get("CER_vision_only")), _pct(s0.get("CER_lex_tie")),
                       _pct(s1.get("CER_vision_only")), _pct(s1.get("CER_lex_tie"))])
    if t7:
        docs.append(_write_table(out, "T7_adaptation", "Zero-shot adaptation to a new written language: CER (%) on target test lines",
                                 ["adaptation", "labeled target lines", "before: vision", "before: + language prior", "after: vision", "after: + language prior"], t7,
                                 "Seed 1 only.  'language prior' = [LEX-TIE] resolution of the tied accent variants with the LLM of the written language."))

    # ---- T8 translation, T9 uncertainty (full profile runs)
    def _fx(v: Any, spec: str) -> str:
        """A number for a table cell, '-' when it is missing or undefined (e.g. an AUROC on a set where every line is wrong)."""
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "-"
        return format(v, spec) if np.isfinite(v) else "-"

    t8, t9_raw = [], []
    for c in corpora + (["joint"] if plan.joint else []):
        rs = [r for r in sel(c, "full", main_lf) if r["profile"] == "full"]
        for r in rs:
            tr = r.get("translation") or {}
            if tr.get("n_random"):
                d = tr.get("chrf_uncertain_minus_plain_same_prompt_oracles") or {}
                t8.append([c, str(tr["n_random"]), _fx(tr.get("chrf_plain_vs_oracle"), ".1f"), _fx(tr.get("chrf_uncertain_vs_oracle_same_prompt"), ".1f"),
                           f"{_fx(d.get('delta'), '+.1f')} [{_fx(d.get('lo'), '+.1f')}, {_fx(d.get('hi'), '+.1f')}]"])
            u = (r.get("interpret") or {}).get("signals") or {}
            if u:
                t9_raw.append((c, u))
    sig_names = sorted({k for _, u in t9_raw for k in u})            # one column per signal, aligned across corpora
    t9 = [[c] + [_fx((u.get(k) or {}).get("auroc_line_has_error"), ".3f") for k in sig_names] for c, u in t9_raw]
    if t8:
        docs.append(_write_table(out, "T8_translation", "Translation of the reading (chrF against the translation of the true text; seed 1)",
                                 ["corpus", "lines", "plain prompt", "uncertainty prompt", "Δ (95 % CI)"], t8,
                                 "No human reference translation exists: the LLM's translation of the ground-truth line is the proxy."))
    if t9:
        docs.append(_write_table(out, "T9_uncertainty", "Which uncertainty signal detects a wrong line? AUROC (seed 1)", ["corpus"] + sig_names, t9))

    # ---- T10 compute
    t10 = []
    for c in corpora + (["joint"] if plan.joint else []):
        rs = [r for r in rows if r["stage"] == c]
        if not rs:
            continue
        def _mean(key: str) -> float:
            v = [float(r[key]) for r in rs if r.get(key) is not None]
            return float(np.mean(v)) if v else float("nan")
        t10.append([c, _fx(rs[0].get("epochs_planned"), ".0f"), _fx(_mean("epochs_trained"), ".0f"), _fx(_mean("best_epoch"), ".0f"),
                    _fx(rs[0].get("complexity"), ".2f"), str(len(rs)), _fx(_mean("gpu_hours"), ".2f"),
                    _fx(sum(float(r.get("gpu_hours") or 0.0) for r in rs), ".1f")])
    if t10:
        docs.append(_write_table(out, "T10_compute", "Training budget and compute (NVIDIA T4)", ["stage", "epoch budget", "epochs trained (mean)",
                                 "best epoch (mean)", "complexity", "finished runs", "GPU-h per run", "GPU-h total"], t10,
                                 "Epoch budget from inklingua/schedule.py (IAM reproduces the v10 defaults); training stops early on val_sel."))

    with open(os.path.join(out, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(docs) + "\n\n### Figures\n\n![label budget](figures/label_budget.png)\n\n![ablations](figures/ablation_heatmap.png)\n")
    _write_status(plan, runs, state, out)
    _write_repro(plan, rows, out)
    return out


def _write_status(plan, runs, state, out) -> None:
    by = defaultdict(list)
    for r in runs:
        by[state.get(r["id"], {}).get("status", "pending")].append(r["id"])
    gh = sum(s.get("seconds", 0.0) for s in state.values()) / 3600.0
    lines = [f"# Status ({time.strftime('%Y-%m-%d %H:%M')})", "", f"{len(by['done'])} of {len(runs)} runs done; GPU time used so far {gh:.1f} h.", ""]
    for k in ("running", "interrupted", "pending", "failed", "skipped"):
        if by[k]:
            lines.append(f"## {k} ({len(by[k])})\n")
            for rid in by[k][:400]:
                err = state.get(rid, {}).get("error", "")
                lines.append(f"- `{rid}`" + (f" - attempts {state[rid].get('attempts', 0)}: `{err.strip().splitlines()[-1][:200]}`" if err else ""))
            lines.append("")
    with open(os.path.join(out, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _write_repro(plan, rows, out) -> None:
    import torch
    from . import __version__
    from .schedule import ERA_FACTOR, MAX_EPOCHS, MIN_EPOCHS, REF_LINES, REF_STEPS, SCRIPT_FACTOR
    from .experiments import stage_list, stage_schedule
    prov = {}
    for r in rows:
        for k, v in ((r.get("provenance") or {}).get("sources") or {}).items():
            prov.setdefault(k, v)
        if (r.get("provenance") or {}).get("decontamination") and r["stage"] == "joint":
            prov.setdefault("_joint_decontamination", r["provenance"]["decontamination"])
    code_sha = ""
    p = os.path.join(plan.work_dir, "_plan", "code_sha256.txt")
    if os.path.exists(p):
        code_sha = open(p).read().strip()
    L = [f"# Reproducibility\n\nInkLingua v{__version__}; code sha256 `{code_sha or 'n/a'}` (the notebook cell, or the repository archive); torch {torch.__version__}; CUDA {torch.version.cuda}; "
         f"Python {platform.python_version()}; LLM `{plan.llm_name}`; seeds {list(plan.seeds)}; main label budget {int(100 * plan.labeled_fraction)} %.\n",
         "## Datasets\n", "| key | Hub dataset | pinned revision | resolved commit | language / script / era | split sizes | caps |", "|---|---|---|---|---|---|---|"]
    for k in dict.fromkeys(list(plan.corpora) + (list(plan.joint_corpora) if plan.joint else [])):
        s = DATASETS[k]
        pv = prov.get(k, {})
        L.append(f"| {k} | {s.hf_dataset} | `{(s.hf_revision or '')[:12]}` | `{(pv.get('resolved_commit') or 'not yet loaded')[:12]}` | {s.language} / {s.script} / {s.era} | "
                 f"{list(s.expected_sizes or [])} | {plan.caps.get(k) or '-'} (joint: {plan.joint_caps}) |")
    if "_joint_decontamination" in prov:
        L.append(f"\nJoint-stage decontamination (exact image digests): {prov['_joint_decontamination']}.\n")
    L += ["\n## Training budget rule\n", f"S = {REF_STEPS:,} x sqrt(N_train / {REF_LINES:,}) x k_script x k_era; epochs = clip(ceil(S / steps per epoch), {MIN_EPOCHS}, {MAX_EPOCHS}); "
          f"k_script = {SCRIPT_FACTOR}; k_era = {ERA_FACTOR}.  Patience, warm-ups and refresh intervals scale with the epoch budget (inklingua/schedule.py).\n",
          "| stage | train lines | complexity | epoch budget | steps / epoch |", "|---|---|---|---|---|"]
    for stage, sources in stage_list(plan):
        b = stage_schedule(plan, stage, sources)
        L.append(f"| {stage} | {b['train_lines']:,} | {b['complexity_factor']:.2f} | {b['epochs']} | {b['steps_per_epoch']:,} |")
    L += ["\n## Protocol\n", "- labeled lines: a seeded random 30 % of the training lines (the seed varies the labeled subset); 10 % of that budget is held out to "
          "calibrate the FRAG-RC trust rule (images join the unlabeled pool; their text only measures the rule);",
          "- validation = val_sel (checkpoint selection, early stopping) + val_gate (every post-hoc decision), disjoint blocks of 20 consecutive lines;",
          "- test: never used for any decision; every evaluation is logged in <run>/protocol/test_looks.jsonl;",
          "- caps keep whole blocks of 20 consecutive lines with a fixed subset seed (identical pools for every variant and seed); test sets are never capped;",
          "- all variants of a corpus share the same optimisation budget; labeled_only additionally trains on the calibration lines;",
          "- the LLM (external knowledge) only re-ranks the recogniser's own hypotheses and is reported separately (system CER)."]
    with open(os.path.join(out, "REPRODUCIBILITY.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(os.path.join(out, "paper_state.json"), "w") as f:
        json.dump(json_safe({"runs_finished": len(rows), "time": time.strftime("%Y-%m-%d %H:%M:%S")}), f)
