"""
The run report ("Reading Atlas"): one self-contained HTML page + numbered folders of the numbers behind every panel.  Built from the files a run
writes (predictions.csv, metrics.json, curves/history.json, protocol/, v11_extras.json); it also reads the flat layout of v9 / v10 runs.  Panels
whose data are missing are left out.  Nothing in it is simulated.
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .style import STYLE_NAMES
from .text import bootstrap_cer_ci
from .utils import LOG, json_safe

_REPORT_REFS = [{"label": "HTR-VT, all 6,482 labeled IAM-line training lines, no external data (arXiv 2409.08573; Pattern Recognition 2025)", "cer": 0.047,
                 "note": "different label budget: context only, NOT a like-for-like comparison"}]
_STYLE_LABELS = {"stroke_width": ("ink thickness", "thin -> thick"), "gap_ratio": ("word-gap ratio", "tight -> wide"), "slant_deg": ("slant (deg)", "left -> right"),
                 "ink_density": ("ink density", "light -> dark"), "body_ratio": ("body / line height", "small -> large")}


def _fnum(s: Any) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def _rd(path: str) -> Optional[Any]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _first(*paths: str) -> Optional[str]:
    return next((p for p in paths if os.path.exists(p)), None)


def _lst(a: Any, nd: int = 4) -> List[Any]:
    a = np.asarray(a, np.float64)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a]


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    ra, rb = np.argsort(np.argsort(a[m])).astype(float), np.argsort(np.argsort(b[m])).astype(float)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if (isinstance(v, float) and not np.isfinite(v)) or v is None else v for v in r])


def read_predictions_csv(path: str) -> Dict[str, Any]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    rows.sort(key=lambda r: int(r["line_index"]))
    n = len(rows)
    L: Dict[str, Any] = {"idx": np.array([int(r["line_index"]) for r in rows]), "gt": [r["ground_truth"] for r in rows], "pr": [r["prediction"] for r in rows],
                         "corpus": [r.get("corpus", "") for r in rows], "conf": np.array([_fnum(r["confidence"]) for r in rows]),
                         "dis": np.array([_fnum(r.get("view_disagreement")) for r in rows]), "ed": np.array([_fnum(r["edits"]) for r in rows]),
                         "rl": np.maximum(np.array([_fnum(r["ref_len"]) for r in rows]), 1.0)}
    L["cer"] = L["ed"] / L["rl"]
    for s in STYLE_NAMES:
        L[s] = np.array([_fnum(r.get(f"measured_{s}")) for r in rows]) if n else np.zeros(0)
    if n and "prediction_posthoc" in rows[0]:
        L["pf"] = [r["prediction_posthoc"] for r in rows]
        L["edf"] = np.array([round(_fnum(r["CER_posthoc"]) * rl) if r.get("CER_posthoc") not in (None, "") else e for r, rl, e in zip(rows, L["rl"], L["ed"])], np.float64)
    else:
        L["pf"], L["edf"] = list(L["pr"]), L["ed"].copy()
    return L


def _hardness_strata(L: Dict[str, Any], n_str: int = 3) -> np.ndarray:
    def z(v):
        m = np.isfinite(v)
        out = np.zeros(len(v))
        if m.sum() > 2 and v[m].std() > 0:
            out[m] = (v[m] - v[m].mean()) / v[m].std()
        return out
    h = -(z(L["stroke_width"]) + z(L["gap_ratio"]))
    return np.digitize(h, np.quantile(h, np.linspace(0, 1, n_str + 1)[1:-1])).astype(int)


def _bin_table(v: np.ndarray, ed: np.ndarray, rl: np.ndarray, edf: np.ndarray, groups: np.ndarray, edges: Sequence[float], seed: int = 0) -> List[Dict[str, Any]]:
    out = []
    b = np.digitize(v, edges[1:-1], right=True)
    fin = np.isfinite(v)
    for k in range(len(edges) - 1):
        m = fin & (b == k)
        if m.sum() < 5:
            out.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(m.sum()), "cer": None, "ci": None, "cer_final": None})
            continue
        lo, hi = bootstrap_cer_ci(ed[m], rl[m], 300, seed + k, groups=groups[m])
        out.append({"lo": float(edges[k]), "hi": float(edges[k + 1]), "n": int(m.sum()), "cer": float(ed[m].sum() / rl[m].sum()), "ci": [float(lo), float(hi)],
                    "cer_final": float(edf[m].sum() / rl[m].sum())})
    return out


def collect_report_data(infer_dir: str, run_dir: Optional[str] = None, curve_dir: Optional[str] = None) -> Dict[str, Any]:
    """All numbers of the report, from files only (v11 layout first, flat v9/v10 layout as a fallback)."""
    mp = _first(os.path.join(infer_dir, "00_summary", "metrics.json"), os.path.join(infer_dir, "metrics.json"))
    pp = _first(os.path.join(infer_dir, "01_recognition", "predictions.csv"), os.path.join(infer_dir, "predictions.csv"))
    xp = _first(os.path.join(infer_dir, "00_summary", "v11_extras.json"), os.path.join(infer_dir, "v10_extras.json"))
    if pp is None:
        raise FileNotFoundError(f"no predictions.csv under {infer_dir}")
    M = (_rd(mp) if mp else None) or {}
    P = read_predictions_csv(pp)
    X = (_rd(xp) if xp else None) or {}
    run_dir = run_dir or os.path.dirname(os.path.dirname(os.path.normpath(infer_dir)))
    H = _rd(os.path.join(curve_dir or os.path.join(run_dir, "curves"), "history.json")) or {}
    prot = _rd(os.path.join(run_dir, "protocol", "protocol.json"))
    n = len(P["idx"])
    blk = 20
    grp = P["idx"] // blk
    synthetic = bool(M.get("SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION"))
    ph = M.get("posthoc") or {}
    rs = M.get("llm_rescoring") or {}
    cer_all = float(P["ed"].sum() / P["rl"].sum())
    ci = M.get("CER_ci95") or list(bootstrap_cer_ci(P["ed"], P["rl"], 400, 0, groups=grp))
    final_cer = M.get("CER_system_final", M.get("CER_final_with_validated_posthoc", cer_all))
    D: Dict[str, Any] = {"meta": {"title": "InkLingua - Reading Atlas", "run": os.path.basename(os.path.normpath(run_dir)), "synthetic": synthetic,
                                  "split": M.get("split", "test"), "n": n, "generated": time.strftime("%Y-%m-%d %H:%M"),
                                  "version": X.get("version", "v11") if X else ("v11" if prot else "older files (read by the v11 report)"),
                                  "corpora": M.get("corpora") or sorted(set(c for c in P["corpus"] if c))}}
    ck = M.get("checkpoint") or {}
    D["head"] = {"cer": cer_all, "ci": ci, "wer": M.get("WER"), "wer_ci": M.get("WER_ci95"), "exact": M.get("exact_line_accuracy"), "final_cer": final_cer,
                 "final_note": M.get("CER_system_final_note"), "n": n, "ckpt_epoch": ck.get("epoch"), "weights_from": ck.get("weights_from"),
                 "val_at_save": ck.get("val_sel_cer_at_save", ck.get("val_cer_at_save")), "labeled_fraction": M.get("labeled_fraction"),
                 "calibration_fraction": M.get("calibration_fraction"), "looks": M.get("evaluation_looks_so_far"), "val_sel_lines": M.get("val_sel_lines"),
                 "val_gate_lines": M.get("val_gate_lines"), "per_corpus": M.get("per_corpus")}
    led: List[Dict[str, Any]] = [{"id": "greedy", "label": "greedy reading", "cer": cer_all, "ci": ci, "kind": "base", "adopted": True,
                                  "note": "arg-max CTC path of the selected checkpoint; nothing tuned on the test split"}]
    for key, label in (("view_ensemble", "view ensemble"), ("tta", "style-cluster AdaBN"), ("wsd", "writer self-distillation (WSD)"), ("space_bonus", "word-gap bonus")):
        o = ph.get(key)
        if not isinstance(o, dict) or o.get("test_CER") is None:
            continue
        gv = o.get("val_delta_vs_base") or o.get("val_delta_vs_before") or {}
        gt_ = o.get("test_delta_vs_base") or o.get("test_delta_vs_before") or {}
        led.append({"id": key, "label": label, "cer": o["test_CER"], "kind": "option", "adopted": bool(o.get("adopted")), "val_gain": [gv.get("lo"), gv.get("hi")],
                    "val_hi_adj": gv.get("hi_adj"), "test_gain": gt_.get("delta"), "test_gain_ci": [gt_.get("lo"), gt_.get("hi")],
                    "note": ("adopted: Bonferroni-adjusted validation gain excludes 0" if o.get("adopted") else
                             "NOT adopted: the validation gain interval includes 0 - test value shown for information only")})
    if ph.get("test_CER_final") is not None:
        led.append({"id": "posthoc_final", "label": "greedy + validated options", "cer": ph["test_CER_final"], "kind": "milestone", "adopted": True,
                    "note": "only the options adopted on val_gate"})
    if rs:
        led.append({"id": "beam", "label": "CTC beam, top-1", "cer": rs.get("CER_beam_top1"), "kind": "milestone", "adopted": True, "note": "prefix beam search on the same posteriors"})
        led.append({"id": "llm", "label": "LLM N-best rescoring", "cer": rs.get("CER_rescored"), "ci": rs.get("CER_rescored_ci95"), "kind": "option",
                    "adopted": bool(rs.get("adopted_on_validation")), "note": "an external language model may only choose among the recogniser's own hypotheses; "
                    + ("adopted on validation" if rs.get("adopted_on_validation") else "NOT adopted on validation"), "llm": rs.get("llm")})
        led.append({"id": "oracle", "label": "oracle N-best (ceiling)", "cer": rs.get("CER_oracle_nbest"), "kind": "ceiling", "adopted": False,
                    "note": "best hypothesis in the N-best list chosen WITH the reference: the most any rescorer could reach"})
    D["ledger"] = led
    D["refs"] = [] if (synthetic or (D["meta"]["corpora"] and D["meta"]["corpora"] != ["iam"])) else _REPORT_REFS

    strat = np.array(X["strata"], int) if X.get("strata") and len(X["strata"]) == n else _hardness_strata(P)
    clus = np.array(X["wsd_cluster"], int) if X.get("wsd_cluster") and len(X["wsd_cluster"]) == n else None
    pw = X.get("wsd_pred") if X.get("wsd_pred") and len(X["wsd_pred"]) == n else None
    edw = np.array(X["wsd_edits"], np.float64) if X.get("wsd_edits") and len(X["wsd_edits"]) == n else None
    risk = 1.0 - np.nan_to_num(P["conf"], nan=0.5)
    D["lines"] = {"idx": P["idx"].tolist(), "cer": _lst(P["cer"]), "conf": _lst(P["conf"]), "dis": _lst(P["dis"]), "ed": P["ed"].astype(int).tolist(),
                  "rl": P["rl"].astype(int).tolist(), "edf": P["edf"].astype(int).tolist(), "strat": strat.tolist(), "corpus": P["corpus"],
                  **{k: _lst(P[k], 5) for k in STYLE_NAMES}, "gt": P["gt"], "pr": P["pr"], "pf": P["pf"],
                  "edw": None if edw is None else edw.astype(int).tolist(), "pw": pw, "clus": None if clus is None else clus.tolist()}
    starts = np.unique(grp)
    be = np.array([P["ed"][grp == g].sum() for g in starts])
    bl = np.array([P["rl"][grp == g].sum() for g in starts])
    D["blocks"] = {"size": blk, "first": (starts * blk).tolist(), "cer": _lst(be / np.maximum(bl, 1))}
    o = np.lexsort((-P["cer"], -P["ed"]))
    cum = np.cumsum(P["ed"][o]) / max(P["ed"].sum(), 1)
    xs = np.arange(1, n + 1) / n
    pick = np.unique(np.linspace(0, n - 1, min(n, 400)).astype(int))
    shares = {str(p): float(cum[max(int(round(p / 100 * n)) - 1, 0)]) for p in (5, 10, 20, 50)}
    D["lorenz"] = {"x": _lst(xs[pick]), "y": _lst(cum[pick]), "shares": shares}
    ro, oo = np.argsort(risk, kind="stable"), np.argsort(P["cer"], kind="stable")
    ab, cc, co, sh = [], [], [], []
    for a in range(0, 91):
        k = max(int(round((1 - a / 100.0) * n)), 1)
        ab.append(a / 100.0)
        cc.append(float(P["ed"][ro[:k]].sum() / P["rl"][ro[:k]].sum()))
        co.append(float(P["ed"][oo[:k]].sum() / P["rl"][oo[:k]].sum()))
        sh.append(float(1.0 - P["ed"][ro[:k]].sum() / max(P["ed"].sum(), 1)))
    D["abstain"] = {"a": ab, "conf": cc, "oracle": co, "share": sh, "base": cer_all, "aurc_conf": float(np.mean(cc)), "aurc_oracle": float(np.mean(co)),
                    "risk_name": "the model's own doubt (1 - sequence confidence)", "frag": None, "aurc_frag": None}
    if "risk" in X and len(X["risk"]) == n:
        rf = np.argsort(np.array(X["risk"], np.float64), kind="stable")
        cf = [float(P["ed"][rf[:max(int(round((1 - a / 100.0) * n)), 1)]].sum() / P["rl"][rf[:max(int(round((1 - a / 100.0) * n)), 1)]].sum()) for a in range(0, 91)]
        D["abstain"]["frag"], D["abstain"]["aurc_frag"] = cf, float(np.mean(cf))
    bins = []
    for s in STYLE_NAMES:
        v = P[s]
        if np.isfinite(v).sum() < 50:
            continue
        edges = np.quantile(v[np.isfinite(v)], np.linspace(0, 1, 6))
        edges[0], edges[-1] = edges[0] - 1e-9, edges[-1] + 1e-9
        lab, dirn = _STYLE_LABELS.get(s, (s, ""))
        bins.append({"key": s, "label": lab, "direction": dirn, "spearman": _spearman(v, P["cer"]), "rows": _bin_table(v, P["ed"], P["rl"], P["edf"], grp, edges, 11)})
    D["bins"] = bins
    D["lenbins"] = {"key": "length", "label": "reference length (characters)", "direction": "short -> long",
                    "rows": _bin_table(P["rl"].astype(np.float64), P["ed"], P["rl"], P["edf"], grp, [0, 20, 35, 50, 65, 1e9], 21)}
    if H and H.get("epoch"):
        keep = ("epoch", "train_eval_cer", "val_cer", "val_ema_cer", "val_sel_cer", "val_ema_sel_cer", "val_gate_cer", "val_ema_gate_cer", "train_ctc_labeled", "lr",
                "pl_coverage", "rc_calib_cov_raw", "pl_sel_cer", "train_frag", "train_frag_trust", "train_frag_char_trust")
        ser = {k: _lst(H[k], 5) for k in H if k in keep or k.startswith("frag_") or k.startswith("val_cer_")}
        D["train"] = {"series": ser, "best_epoch": ck.get("epoch"), "n_strata": sum(1 for k in H if k.startswith("frag_theta_s"))}
    D["frag"] = X.get("frag_audit")
    D["wsd"] = ph.get("wsd") if isinstance(ph.get("wsd"), dict) and ph["wsd"].get("test_by_cluster") else None
    D["traces"] = X.get("traces") or []
    D["nbest"] = X.get("nbest")
    D["interp"] = M.get("interpretability") or {}
    ins: Dict[str, str] = {}
    fs = (M.get("error_structure") or {}).get("form_structure") or {}
    r5, r20 = shares["5"], shares["20"]
    ins["terrain"] = (f"The 5 % of the lines with the most edits hold {r5 * 100:.0f} % of all character edits, the top 20 % hold {r20 * 100:.0f} %."
                      + (f" Neighbouring lines are alike (lag-1 autocorrelation of line CER {fs['lag1_autocorrelation_of_line_CER']:.2f}): the model fails in stretches - "
                         "a hand, a form - not at random." if fs.get("lag1_autocorrelation_of_line_CER") is not None else ""))
    a20 = D["abstain"]
    ins["abstain"] = (f"Declining the 20 % least reliable lines (by the model's own doubt) leaves a CER of {a20['conf'][20] * 100:.2f} % on the rest (from {cer_all * 100:.2f} %); "
                      f"those lines carry {a20['share'][20] * 100:.0f} % of all edits. An oracle that knew the errors would reach {a20['oracle'][20] * 100:.2f} %.")
    sb = next((b for b in bins if b["key"] == "stroke_width"), None)
    gb = next((b for b in bins if b["key"] == "gap_ratio"), None)
    parts = []
    for b, nm in ((sb, "ink thickness"), (gb, "word-gap ratio")):
        if b and b["rows"][0]["cer"] is not None and b["rows"][-1]["cer"] is not None:
            parts.append(f"{nm}: {b['rows'][0]['cer'] * 100:.1f} % in the lowest quintile vs {b['rows'][-1]['cer'] * 100:.1f} % in the highest")
    lb = D["lenbins"]["rows"]
    if lb[-1]["cer"] is not None:
        parts.append(f"lines of more than 65 characters: {lb[-1]['cer'] * 100:.1f} % ({lb[-1]['n']} lines)")
    ins["bins"] = "CER by measured property - " + "; ".join(parts) + "." if parts else ""
    if "train" in D:
        s_ = D["train"]["series"]
        tr_ = [v for v in s_.get("train_eval_cer", []) if v is not None]
        va_ = [v for v in (s_.get("val_sel_cer") or s_.get("val_cer", [])) if v is not None]
        if tr_ and va_:
            ins["train"] = (f"At the end of training the labeled lines are read at CER {tr_[-1] * 100:.2f} % while held-out validation lines are read at {va_[-1] * 100:.2f} % - "
                            f"a {va_[-1] / max(tr_[-1], 1e-6):.0f}x gap: the labeled set is memorised, so more passes over it cannot help; only new information can.")
    adopted_opts = [e["label"] for e in led if e["kind"] == "option" and e["adopted"]]
    ins["ledger"] = (f"System CER {final_cer * 100:.2f} % vs {cer_all * 100:.2f} % for the plain greedy reading. Options adopted on validation: "
                     + (", ".join(adopted_opts) if adopted_opts else "none") + ". Options whose validation evidence did not exclude zero are shown hollow and are not applied.")
    D["insights"] = ins
    looks = []
    lp_ = os.path.join(run_dir, "protocol", "test_looks.jsonl")
    if os.path.exists(lp_):
        with open(lp_) as f:
            looks = [json.loads(x) for x in f if x.strip()]
    D["integrity"] = {"protocol": prot, "looks": looks, "provenance": M.get("provenance"), "model_selection": M.get("model_selection"), "checkpoint": ck,
                      "posthoc_rule": ph.get("rule")}
    D["caveats"] = [
        "Writer-independence of the test split is not verified here: if train and test share writers, the transductive gains (WSD, AdaBN) are optimistic for unseen writers.",
        "A single run is a single draw: claims about a method need the seed-paired suite (labeled_only / legacy_pl / full over >= 3 seeds), not one run.",
        "The external language model was trained on web text that may contain the corpus the benchmark is drawn from: report LLM-rescored numbers as 'with an external language model'.",
        "The form-clustered intervals treat 20 consecutive lines as one cluster; the corpora have no writer id, so the true cluster structure is unknown.",
        "Hardness strata, style clusters and the FRAG-RC evidence use measured pen properties only - never the transcription.",
        "Right-to-left corpora are shown in logical order; the recogniser reads them in visual order (the metrics are identical)."]
    if synthetic:
        D["caveats"].insert(0, "SYNTHETIC SMOKE TEST: generated fake handwriting; the numbers have no scientific meaning.")
    return D


def export_report_folders(D: Dict[str, Any], out: str) -> None:
    L = D["lines"]
    os.makedirs(out, exist_ok=True)
    _write_csv(os.path.join(out, "01_ledger", "ledger.csv"), ["step", "CER", "ci_lo", "ci_hi", "adopted", "kind", "note"],
               [[e["label"], e.get("cer"), (e.get("ci") or [None, None])[0], (e.get("ci") or [None, None])[1], e.get("adopted"), e.get("kind"), e.get("note")] for e in D["ledger"]])
    with open(os.path.join(out, "01_ledger", "headline.json"), "w") as f:
        json.dump(json_safe({"meta": D["meta"], "head": D["head"], "insight": D["insights"].get("ledger")}), f, indent=2, allow_nan=False)
    keys = ["idx", "corpus", "cer", "conf", "dis", "ed", "rl", "edf", "strat"] + list(STYLE_NAMES) + ["gt", "pr", "pf"] + (["edw", "pw", "clus"] if L.get("edw") else [])
    _write_csv(os.path.join(out, "02_error_terrain", "lines.csv"), keys, [[L[k][i] for k in keys] for i in range(len(L["idx"]))])
    _write_csv(os.path.join(out, "02_error_terrain", "blocks.csv"), ["first_line", "lines", "CER"], [[a, D["blocks"]["size"], c] for a, c in zip(D["blocks"]["first"], D["blocks"]["cer"])])
    _write_csv(os.path.join(out, "03_concentration_abstention", "lorenz.csv"), ["share_of_lines_worst_first", "share_of_edits"], list(zip(D["lorenz"]["x"], D["lorenz"]["y"])))
    A = D["abstain"]
    _write_csv(os.path.join(out, "03_concentration_abstention", "risk_coverage.csv"), ["abstained_share", "CER_of_kept_by_model_doubt", "CER_of_kept_oracle", "share_of_edits_removed"],
               list(zip(A["a"], A["conf"], A["oracle"], A["share"])))
    rows = []
    for b in D["bins"] + [D["lenbins"]]:
        for k, r in enumerate(b["rows"]):
            rows.append([b["label"], k + 1, r["lo"], r["hi"], r["n"], r["cer"], (r["ci"] or [None, None])[0], (r["ci"] or [None, None])[1], r["cer_final"]])
    _write_csv(os.path.join(out, "04_who_is_hard", "style_bins.csv"), ["property", "quintile", "from", "to", "lines", "CER", "ci_lo", "ci_hi", "CER_final"], rows)
    if D.get("train"):
        s = D["train"]["series"]
        ks = list(s.keys())
        _write_csv(os.path.join(out, "05_training_story", "history.csv"), ks, list(zip(*[s[k] for k in ks])))
        fk = [k for k in ks if k.startswith("frag_") or k.startswith("train_frag") or k == "epoch"]
        if len(fk) > 1:
            _write_csv(os.path.join(out, "06_trust_audit", "frag_timeline.csv"), fk, list(zip(*[s[k] for k in fk])))
    if D.get("frag"):
        fa = D["frag"]
        _write_csv(os.path.join(out, "06_trust_audit", "frag_audit.csv"), ["stratum", "lines", "characters", "trusted_share", "error_of_trusted", "error_of_doubtful", "target"],
                   [[s["stratum"], s["lines"], s["chars"], s["trusted_share"], s["err_trusted"], s["err_doubtful"], fa.get("target")] for s in fa["strata"]])
    if D.get("wsd"):
        w = D["wsd"]
        _write_csv(os.path.join(out, "07_writer_adaptation", "wsd_clusters.csv"), ["split", "cluster", "lines", "CER_before", "CER_after"],
                   [[sp, g, c["lines"], c["CER_before"], c["CER_after"]] for sp, key in (("val_gate", "val_by_cluster"), ("test", "test_by_cluster")) for g, c in (w.get(key) or {}).items()])
    if D.get("traces"):
        with open(os.path.join(out, "08_reading_traces", "traces.json"), "w", encoding="utf-8") as f:
            json.dump(json_safe(D["traces"]), f, indent=1, ensure_ascii=False, allow_nan=False)
    I = D["integrity"]
    os.makedirs(os.path.join(out, "09_integrity"), exist_ok=True)
    with open(os.path.join(out, "09_integrity", "protocol.json"), "w") as f:
        json.dump(json_safe(I), f, indent=2, allow_nan=False)
    with open(os.path.join(out, "09_integrity", "test_looks.jsonl"), "w") as f:
        f.write("".join(json.dumps(x) + "\n" for x in I.get("looks") or []))
    with open(os.path.join(out, "09_integrity", "caveats.md"), "w", encoding="utf-8") as f:
        f.write("# What this report does not establish\n\n" + "\n".join(f"- {c}" for c in D["caveats"]) + "\n")


def build_run_report(infer_dir: str, run_dir: Optional[str] = None, out_dir: Optional[str] = None, curve_dir: Optional[str] = None) -> str:
    """Write <infer_dir>/12_report/ (index.html + the numbered folders).  Returns the path of index.html."""
    D = collect_report_data(infer_dir, run_dir, curve_dir)
    out = out_dir or os.path.join(infer_dir, "12_report")
    export_report_folders(D, out)
    data = json.dumps(json_safe(D), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    page = _REPORT_HTML.replace("/*__CSS__*/", _REPORT_CSS).replace("/*__JS__*/", _REPORT_JS).replace("/*__DATA__*/null", data)
    path = os.path.join(out, "index.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page)
    LOG.info("Run report written: %s", path)
    return path


_REPORT_CSS = r''':root{--paper:#f7f4ee;--card:#fffdf9;--ink:#1d2330;--muted:#6b7280;--line:#ddd6c8;--accent:#b5532f;--good:#1f7a6d;--warn:#c58a1a;--cool:#3b6ea8;--purple:#7a4fa3;--shadow:0 1px 2px rgba(30,25,15,.06),0 8px 24px rgba(30,25,15,.05)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--paper:#14161a;--card:#1b1e24;--ink:#ebe8e1;--muted:#9aa1ad;--line:#343944;--accent:#e0805a;--good:#46b5a3;--warn:#e0aa42;--cool:#6fa3dc;--purple:#b08ad8;--shadow:0 1px 2px rgba(0,0,0,.4)}}
:root[data-theme="dark"]{--paper:#14161a;--card:#1b1e24;--ink:#ebe8e1;--muted:#9aa1ad;--line:#343944;--accent:#e0805a;--good:#46b5a3;--warn:#e0aa42;--cool:#6fa3dc;--purple:#b08ad8;--shadow:0 1px 2px rgba(0,0,0,.4)}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 -apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
.mono,.txt{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
h1,h2,h3{font-family:Georgia,"Iowan Old Style","Times New Roman",serif;font-weight:600;letter-spacing:-.01em;margin:0}
.layout{display:grid;grid-template-columns:210px minmax(0,1fr);gap:0;max-width:1480px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;padding:28px 16px 24px 24px;overflow:auto;border-right:1px solid var(--line)}
nav .brand{font-family:Georgia,serif;font-size:18px;font-weight:600;margin-bottom:2px}nav .sub{font-size:12px;color:var(--muted);margin-bottom:22px}
nav a{display:flex;gap:9px;align-items:baseline;padding:5px 0;color:var(--muted);text-decoration:none;font-size:13.5px}nav a b{font-variant-numeric:tabular-nums;color:var(--accent);font-weight:600;width:18px}nav a:hover,nav a.on{color:var(--ink)}
main{padding:30px 40px 90px;min-width:0}
header.top{padding-bottom:26px;border-bottom:1px solid var(--line);margin-bottom:12px}
header.top h1{font-size:34px;margin-bottom:6px}header.top p{margin:0;color:var(--muted);max-width:78ch}
.badges{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 4px}.badge{font-size:12px;padding:3px 10px;border-radius:99px;border:1px solid var(--line);background:var(--card);color:var(--muted)}
.badge.warn{background:#b00020;color:#fff;border-color:#b00020;font-weight:700}.badge.good{color:var(--good);border-color:var(--good)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:14px;margin-top:22px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:14px 16px;box-shadow:var(--shadow)}.kpi .v{font:600 30px/1.1 Georgia,serif;font-variant-numeric:tabular-nums}.kpi .l{font-size:12px;color:var(--muted);margin-top:4px}.kpi .s{font-size:12px;color:var(--muted);font-variant-numeric:tabular-nums}
section{padding:46px 0 10px;scroll-margin-top:8px}
section>.hd{display:flex;gap:14px;align-items:baseline}section>.hd .n{font:600 14px Georgia,serif;color:var(--accent);border:1px solid var(--accent);border-radius:99px;width:28px;height:28px;display:grid;place-items:center;flex:none}
section h2{font-size:26px}.insight{font-size:16.5px;margin:10px 0 18px;max-width:84ch;border-left:3px solid var(--accent);padding:2px 0 2px 14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px 18px;box-shadow:var(--shadow);margin:14px 0}
.howto{font-size:13px;color:var(--muted);max-width:90ch;margin:8px 2px 0}.howto b{color:var(--ink)}
svg.chart{width:100%;height:auto;display:block;overflow:visible}svg text{fill:var(--ink);font:12px -apple-system,"Segoe UI",sans-serif}svg .m{fill:var(--muted)}svg .ax{stroke:var(--line);stroke-width:1}svg .gr{stroke:var(--line);stroke-width:1;stroke-dasharray:2 4}
.row2{display:grid;grid-template-columns:repeat(auto-fit,minmax(380px,1fr));gap:16px}.grid5{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{font-weight:600;color:var(--muted);font-size:12px}
td.r,th.r{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.tip{position:fixed;z-index:50;pointer-events:none;background:var(--ink);color:var(--paper);font-size:12px;padding:8px 10px;border-radius:8px;max-width:420px;display:none;box-shadow:0 6px 24px rgba(0,0,0,.25)}.tip .txt{white-space:pre-wrap;word-break:break-all}
canvas{display:block;max-width:100%}.cv{position:relative}.cv canvas+canvas{position:absolute;left:0;top:0}
button,select,input[type=text]{font:inherit;font-size:13px;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:8px;padding:5px 10px}button{cursor:pointer}button:hover{border-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent)}
.ctrl{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin:8px 0 12px}
b.sub{background:rgba(181,83,47,.22);border-radius:3px;padding:0 1px}b.del{background:rgba(197,138,26,.35);border-radius:3px;padding:0 1px}b.ins{background:rgba(122,79,163,.30);border-radius:3px;padding:0 1px}
.diff{font-size:13px;line-height:1.55;word-break:break-word}.diff .k{display:inline-block;width:46px;color:var(--muted);font:11px sans-serif}
.bar{height:8px;border-radius:4px;background:var(--line);overflow:hidden;min-width:60px}.bar i{display:block;height:100%;background:var(--accent)}
.trace{display:grid;gap:8px}.trace .imgwrap{position:relative;background:#fff;border-radius:8px;overflow:hidden;border:1px solid var(--line)}.trace img{display:block;width:100%;height:auto}
.trace .ch{position:absolute;top:0;bottom:0;border-left:2px solid;opacity:.9}.trace .ch.t{border-color:var(--good);background:rgba(31,122,109,.10)}.trace .ch.d{border-color:var(--warn);background:rgba(197,138,26,.16)}.trace .ch.w{box-shadow:inset 0 -4px 0 #c0392b}
.legend{display:flex;flex-wrap:wrap;gap:14px;font-size:12px;color:var(--muted);margin:6px 0}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-2px}
.note{font-size:12.5px;color:var(--muted)}.pill{font-size:11px;padding:1px 8px;border-radius:99px;border:1px solid var(--line);color:var(--muted)}.pill.ok{color:var(--good);border-color:var(--good)}.pill.no{color:var(--warn);border-color:var(--warn)}
.cav li{margin:6px 0}
@media (max-width:900px){.layout{grid-template-columns:1fr}nav{display:none}main{padding:20px 16px 60px}}
'''

_REPORT_JS = r'''(function(){
"use strict";
const NS="http://www.w3.org/2000/svg";
function mk(ns,tag,props,kids){
  const e=ns?document.createElementNS(NS,tag):document.createElement(tag);
  for(const k in (props||{})){const v=props[k];
    if(v===null||v===undefined||v===false)continue;
    if(k==="class")e.setAttribute("class",v);
    else if(k==="html")e.innerHTML=v;
    else if(k.slice(0,2)==="on")e.addEventListener(k.slice(2),v);
    else e.setAttribute(k,v===true?"":v);}
  for(const c of kids.flat(Infinity)){if(c===null||c===undefined||c===false)continue;e.append(c.nodeType?c:document.createTextNode(String(c)));}
  return e;}
const h=(tag,props,...kids)=>mk(false,tag,props,kids);
const s=(tag,props,...kids)=>mk(true,tag,props,kids);
const esc=t=>String(t).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
const fin=v=>v!==null&&v!==undefined&&isFinite(v);
const pct=(x,d=2)=>fin(x)?(x*100).toFixed(d)+" %":"n/a";
const pt=(x,d=2)=>fin(x)?(x>=0?"+":"")+(x*100).toFixed(d)+" pt":"n/a";
const f3=(x,d=3)=>fin(x)?Number(x).toFixed(d):"n/a";
const cv=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const rgb=hex=>{const v=parseInt(hex.replace("#",""),16);return [v>>16&255,v>>8&255,v&255];};
const mix=(a,b,t)=>{t=Math.max(0,Math.min(1,t));return "rgb("+a.map((x,i)=>Math.round(x+(b[i]-x)*t)).join(",")+")";};
const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
const mean=a=>{let t=0,n=0;for(const v of a)if(fin(v)){t+=v;n++;}return n?t/n:NaN;};
const quant=(a,q)=>{const b=a.filter(fin).sort((x,y)=>x-y);return b.length?b[Math.min(b.length-1,Math.floor(q*b.length))]:NaN;};
const mounts=[];
const D=window.__D;
const tip=h("div",{class:"tip"});
function showTip(html,ev){tip.innerHTML=html;tip.style.display="block";const w=tip.offsetWidth,hh=tip.offsetHeight;let x=ev.clientX+14,y=ev.clientY+14;if(x+w>innerWidth-8)x=ev.clientX-w-14;if(y+hh>innerHeight-8)y=ev.clientY-hh-14;tip.style.left=x+"px";tip.style.top=y+"px";}
const hideTip=()=>{tip.style.display="none";};
function diff(ref,hyp){
  const a=Array.from(ref),b=Array.from(hyp),n=a.length,m=b.length;
  if(n*m>40000)return [esc(ref),esc(hyp)];
  const dp=[];for(let i=0;i<=n;i++){dp.push(new Uint16Array(m+1));dp[i][0]=i;}
  for(let j=0;j<=m;j++)dp[0][j]=j;
  for(let i=1;i<=n;i++)for(let j=1;j<=m;j++)dp[i][j]=Math.min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1));
  let i=n,j=m;const ra=[],rb=[];const vis=c=>c===" "?"␣":esc(c);
  while(i>0||j>0){
    if(i>0&&j>0&&dp[i][j]===dp[i-1][j-1]+(a[i-1]===b[j-1]?0:1)){
      if(a[i-1]===b[j-1]){ra.push(esc(a[i-1]));rb.push(esc(b[j-1]));}
      else{ra.push('<b class="sub">'+vis(a[i-1])+"</b>");rb.push('<b class="sub">'+vis(b[j-1])+"</b>");}
      i--;j--;}
    else if(i>0&&dp[i][j]===dp[i-1][j]+1){ra.push('<b class="del">'+vis(a[i-1])+"</b>");i--;}
    else{rb.push('<b class="ins">'+vis(b[j-1])+"</b>");j--;}}
  return [ra.reverse().join(""),rb.reverse().join("")];}
const diffBlock=(gt,pr,extra)=>{const [r,p]=diff(gt,pr);let html='<div class="diff txt"><div><span class="k">truth</span>'+r+'</div><div><span class="k">reads</span>'+p+"</div>";
  if(extra){const [r2,p2]=diff(gt,extra.text);html+='<div><span class="k">'+esc(extra.label)+"</span>"+p2+"</div>";}return html+"</div>";};
const SEC=[];
function frame(id,title,insight,body,how){
  const n=SEC.length+1;SEC.push([id,title]);
  return h("section",{id},h("div",{class:"hd"},h("span",{class:"n"},n),h("h2",null,title)),
    insight?h("p",{class:"insight"},insight):null,...body,how?h("p",{class:"howto",html:how}):null);}
function lineChart(o){
  const W=o.w||980,Hh=o.h||300,m={l:56,r:18,t:22,b:34},pw=W-m.l-m.r,ph=Hh-m.t-m.b;
  const xs=o.xs,all=[];o.series.forEach(se=>se.vals.forEach(v=>{if(fin(v)&&(!o.ylog||v>0))all.push(v);}));
  if(!all.length||!xs.length)return null;
  let y0=o.ymin!==undefined?o.ymin:(o.ylog?Math.min(...all)*.8:Math.min(0,Math.min(...all))),y1=o.ymax!==undefined?o.ymax:Math.max(...all)*1.06;
  if(o.ylog){y0=Math.max(y0,1e-4);}
  const xa=Math.min(...xs),xb=Math.max(...xs)||1;
  const X=v=>m.l+pw*(v-xa)/Math.max(xb-xa,1e-9),Y=v=>o.ylog?m.t+ph*(1-(Math.log(v)-Math.log(y0))/(Math.log(y1)-Math.log(y0))):m.t+ph*(1-(v-y0)/(y1-y0));
  const svg=s("svg",{viewBox:`0 0 ${W} ${Hh}`,class:"chart"});
  const ticks=[];if(o.ylog){let t=Math.pow(10,Math.floor(Math.log10(y0)));for(;t<=y1*1.01;t*=10){[1,2,5].forEach(k=>{const v=t*k;if(v>=y0&&v<=y1)ticks.push(v);});}}else{for(let k=0;k<=4;k++)ticks.push(y0+(y1-y0)*k/4);}
  ticks.forEach(v=>{svg.append(s("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:"gr"}),s("text",{x:m.l-8,y:Y(v)+4,"text-anchor":"end",class:"m"},o.yfmt?o.yfmt(v):v.toFixed(2)));});
  const step=Math.max(1,Math.ceil(xs.length/12));
  xs.forEach((v,i)=>{if(i%step===0)svg.append(s("text",{x:X(v),y:Hh-12,"text-anchor":"middle",class:"m"},v));});
  svg.append(s("line",{x1:m.l,x2:W-m.r,y1:m.t+ph,y2:m.t+ph,class:"ax"}));
  (o.markers||[]).forEach(mk_=>{if(mk_.x===null||mk_.x===undefined)return;svg.append(s("line",{x1:X(mk_.x),x2:X(mk_.x),y1:m.t,y2:m.t+ph,stroke:"var(--ink)","stroke-dasharray":"3 3",opacity:.6}),s("text",{x:X(mk_.x)+4,y:m.t+ph-6,class:"m"},mk_.label));});
  o.series.forEach(se=>{let d="",pen=false,cnt=0;se.vals.forEach((v,i)=>{if(!fin(v)||(o.ylog&&v<=0))return;d+=(pen?"L":"M")+X(xs[i]).toFixed(1)+" "+Y(v).toFixed(1);pen=true;cnt++;});
    if(d)svg.append(s("path",{d,fill:"none",stroke:se.color,"stroke-width":se.w||2.2,"stroke-dasharray":se.dash||null,"stroke-linejoin":"round"}));
    if(cnt&&cnt<se.vals.length*.6)se.vals.forEach((v,i)=>{if(fin(v)&&!(o.ylog&&v<=0))svg.append(s("circle",{cx:X(xs[i]),cy:Y(v),r:3,fill:se.color}));});});
  let lx=m.l;o.series.forEach(se=>{svg.append(s("line",{x1:lx,x2:lx+18,y1:10,y2:10,stroke:se.color,"stroke-width":3,"stroke-dasharray":se.dash||null}),s("text",{x:lx+23,y:14},se.name));lx+=34+se.name.length*6.3;});
  return svg;}
function secLedger(){
  const rows=D.ledger.filter(e=>fin(e.cer));if(!rows.length)return null;
  const W=1000,rowH=50,L=250,R=110,top=30,H=top+rows.length*rowH+34;
  const mx=Math.max(...rows.map(e=>e.ci?e.ci[1]:e.cer),...(D.refs||[]).map(r=>r.cer))*1.1;
  const X=v=>L+(W-L-R)*v/mx;const svg=s("svg",{viewBox:`0 0 ${W} ${H}`,class:"chart"});
  for(let k=0;k<=5;k++){const v=mx*k/5;svg.append(s("line",{x1:X(v),x2:X(v),y1:top-8,y2:H-30,class:"gr"}),s("text",{x:X(v),y:H-12,"text-anchor":"middle",class:"m"},(v*100).toFixed(1)+" %"));}
  const col=e=>e.kind==="ceiling"?"var(--muted)":e.kind==="option"?(e.adopted?"var(--good)":"var(--warn)"):e.kind==="base"?"var(--ink)":"var(--cool)";
  rows.forEach((e,i)=>{const y=top+i*rowH,c=col(e),filled=e.kind!=="ceiling"&&!(e.kind==="option"&&!e.adopted);
    svg.append(s("text",{x:L-14,y:y+17,"text-anchor":"end","font-weight":600},e.label),
      s("text",{x:L-14,y:y+33,"text-anchor":"end",class:"m"},e.kind==="option"?(e.adopted?"adopted":"not adopted"):e.kind==="ceiling"?"reference ceiling":e.kind==="milestone"?"milestone":"baseline"),
      s("rect",{x:L,y:y+6,width:Math.max(2,X(e.cer)-L),height:26,rx:5,fill:filled?c:"none",stroke:c,"stroke-width":2,"stroke-dasharray":filled?null:"5 4",opacity:filled?.88:1,
        onmousemove:ev=>showTip("<b>"+esc(e.label)+"</b><br>CER "+pct(e.cer)+"<br>"+esc(e.note||"")+(e.val_hi_adj!==undefined&&e.val_hi_adj!==null?"<br>validation gain, upper bound (Bonferroni-adjusted): "+pt(e.val_hi_adj):"")+(fin(e.test_gain)?"<br>test change vs. what it was applied to: "+pt(e.test_gain):""),ev),onmouseleave:hideTip}));
    if(e.ci)svg.append(s("line",{x1:X(e.ci[0]),x2:X(e.ci[1]),y1:y+19,y2:y+19,stroke:"var(--ink)","stroke-width":1.6}),s("line",{x1:X(e.ci[0]),x2:X(e.ci[0]),y1:y+13,y2:y+25,stroke:"var(--ink)"}),s("line",{x1:X(e.ci[1]),x2:X(e.ci[1]),y1:y+13,y2:y+25,stroke:"var(--ink)"}));
    svg.append(s("text",{x:X(e.ci?e.ci[1]:e.cer)+10,y:y+24,"font-weight":600},pct(e.cer)));});
  (D.refs||[]).forEach(r=>{svg.append(s("line",{x1:X(r.cer),x2:X(r.cer),y1:top-8,y2:H-30,stroke:"var(--purple)","stroke-width":1.6,"stroke-dasharray":"2 4"}),s("text",{x:X(r.cer)+5,y:top-12,fill:"var(--purple)"},"literature: "+pct(r.cer,1)));});
  const det=h("table",null,h("thead",null,h("tr",null,h("th",null,"step"),h("th",{class:"r"},"CER"),h("th",null,"evidence / what it is"))),
    h("tbody",null,rows.map(e=>h("tr",null,h("td",null,e.label),h("td",{class:"r"},pct(e.cer)),h("td",null,(e.note||"")+(e.val_hi_adj!==undefined&&e.val_hi_adj!==null?"  [validation gain upper bound, adjusted: "+pt(e.val_hi_adj)+"]":"")+(fin(e.test_gain)?"  [test change: "+pt(e.test_gain)+"]":""))))));
  const body=[h("div",{class:"card"},svg),h("div",{class:"card"},det)];
  const pc=D.head.per_corpus;if(pc&&Object.keys(pc).length>1){body.push(h("div",{class:"card"},h("h3",null,"Per corpus (greedy reading)"),h("table",null,h("thead",null,h("tr",null,["corpus","lines","CER","95 % CI","WER","exact lines"].map((t,i)=>h("th",{class:i?"r":""},t)))),
    h("tbody",null,Object.entries(pc).map(([k,v])=>h("tr",null,h("td",null,k),h("td",{class:"r"},v.lines),h("td",{class:"r"},pct(v.CER)),h("td",{class:"r"},v.CER_ci95?pct(v.CER_ci95[0],1)+" - "+pct(v.CER_ci95[1],1):""),h("td",{class:"r"},pct(v.WER,1)),h("td",{class:"r"},pct(v.exact_line_accuracy,1))))))));}
  return frame("ledger","The ledger",D.insights.ledger,body,
    "<b>Solid bars</b> are applied; <b>hollow bars</b> were evaluated but not adopted because their validation evidence (a Bonferroni-adjusted, form-clustered paired bootstrap on <b>val_gate</b>) did not exclude zero - the test split never decides. "+((D.refs||[]).length?"The dotted purple line is a published number for a different label budget, shown for context only.":""));}
function secTerrain(){
  const Ls=D.lines,n=Ls.idx.length;if(n<20)return null;
  const wrap=h("div",{class:"card"}),sel=h("div",{class:"card"},h("p",{class:"note"},"Drag across the terrain to select a stretch of lines, or use the buttons."));
  const tracks=[
    {name:"line CER",v:Ls.cer,kind:"err"},
    {name:"model doubt",v:Ls.conf.map(c=>fin(c)?1-c:null),kind:"doubt"},
    {name:"ink thickness",v:Ls.stroke_width,kind:"div"},
    {name:"word-gap ratio",v:Ls.gap_ratio,kind:"div"},
    {name:"hardness cell",v:Ls.strat,kind:"cat"}];
  const dEd=Ls.ed.map((e,i)=>e-Ls.edf[i]);if(dEd.some(v=>v!==0))tracks.push({name:"edits saved: options",v:dEd,kind:"delta"});
  if(Ls.edw){const dw=Ls.edf.map((e,i)=>e-Ls.edw[i]);if(dw.some(v=>v!==0))tracks.push({name:"edits saved: WSD",v:dw,kind:"delta"});}
  const ML=150,TH=19,GAP=4,BH=64;let state={a:null,b:null,drag:false,hover:-1};
  const base=h("canvas"),over=h("canvas"),box=h("div",{class:"cv"},base,over);wrap.append(box,h("div",{class:"legend"},
    h("span",null,h("i",{style:"background:var(--accent)"}),"hard / thin / tight"),h("span",null,h("i",{style:"background:var(--cool)"}),"easy / thick / wide"),
    h("span",null,h("i",{style:"background:var(--good)"}),"edits saved"),h("span",null,h("i",{style:"background:var(--purple)"}),"edits added")),
    h("div",{class:"ctrl"},h("button",{onclick:()=>pick("worst")},"jump to the worst 20-line block"),h("button",{onclick:()=>pick("best")},"jump to the best block"),h("button",{onclick:()=>{state.a=state.b=null;draw();sel.replaceChildren(h("p",{class:"note"},"Selection cleared."));}},"clear")));
  let W=900,Ht=0;
  function layout(){W=Math.max(640,wrap.clientWidth-36);Ht=tracks.length*(TH+GAP)+BH+30;const dpr=window.devicePixelRatio||1;
    [base,over].forEach(c=>{c.width=W*dpr;c.height=Ht*dpr;c.style.width=W+"px";c.style.height=Ht+"px";c.getContext("2d").setTransform(dpr,0,0,dpr,0,0);});}
  const paper=()=>rgb(cv("--paper")),A=()=>rgb(cv("--accent")),C=()=>rgb(cv("--cool")),G=()=>rgb(cv("--good")),P=()=>rgb(cv("--purple"));
  const stats=tracks.map(t=>{const m=mean(t.v);const sd=Math.sqrt(mean(t.v.map(x=>fin(x)?(x-m)*(x-m):null)))||1;return {m,sd,lo:quant(t.v,.03),hi:quant(t.v,.97)};});
  function colour(t,st,v){if(!fin(v))return null;const p=paper();
    if(t.kind==="err")return mix(p,A(),Math.sqrt(clamp(v/.35,0,1)));
    if(t.kind==="doubt")return mix(p,A(),clamp((v-st.lo)/Math.max(st.hi-st.lo,1e-6),0,1));
    if(t.kind==="div"){const z=clamp((v-st.m)/st.sd/2.2,-1,1);return z<0?mix(p,A(),-z):mix(p,C(),z);}
    if(t.kind==="cat"){return mix(p,A(),clamp(v/Math.max(st.hi,1),0,1)*.9+.05);}
    if(t.kind==="delta"){const z=clamp(v/3,-1,1);return z>0?mix(p,G(),z):z<0?mix(p,P(),-z):mix(p,p,0);}
    return null;}
  function draw(){const ctx=base.getContext("2d"),pw=W-ML-8;ctx.clearRect(0,0,W,Ht);ctx.font="12px -apple-system,Segoe UI,sans-serif";ctx.textBaseline="middle";
    tracks.forEach((t,k)=>{const y=k*(TH+GAP)+2;ctx.fillStyle=cv("--ink");ctx.fillText(t.name,0,y+TH/2);
      for(let c=0;c<pw;c++){const i0=Math.floor(c*n/pw),i1=Math.max(i0+1,Math.floor((c+1)*n/pw));let tot=0,cnt=0;for(let i=i0;i<i1&&i<n;i++){const v=t.v[i];if(fin(v)){tot+=v;cnt++;}}
        if(cnt){const col=colour(t,stats[k],tot/cnt);if(col){ctx.fillStyle=col;ctx.fillRect(ML+c,y,1.25,TH);}}}});
    const y0=tracks.length*(TH+GAP)+6;ctx.fillStyle=cv("--ink");ctx.fillText("20-line block CER",0,y0+BH/2);
    const bw=D.blocks;const mxb=Math.max(.2,quant(bw.cer,.98));
    bw.cer.forEach((v,b)=>{if(!fin(v))return;const x0=ML+pw*bw.first[b]/n,x1=ML+pw*Math.min(n,bw.first[b]+bw.size)/n;const hh=BH*clamp(v/mxb,0,1);ctx.fillStyle=cv("--accent");ctx.globalAlpha=.85;ctx.fillRect(x0,y0+BH-hh,Math.max(1,x1-x0-.5),hh);ctx.globalAlpha=1;});
    ctx.strokeStyle=cv("--line");ctx.beginPath();ctx.moveTo(ML,y0+BH+.5);ctx.lineTo(W-8,y0+BH+.5);ctx.stroke();
    ctx.fillStyle=cv("--muted");ctx.fillText("line 0",ML,y0+BH+14);ctx.textAlign="right";ctx.fillText("line "+Ls.idx[n-1],W-8,y0+BH+14);ctx.textAlign="left";
    const oc=over.getContext("2d");oc.clearRect(0,0,W,Ht);
    if(state.a!==null){const i0=Math.min(state.a,state.b),i1=Math.max(state.a,state.b),x0=ML+pw*i0/n,x1=ML+pw*(i1+1)/n;oc.fillStyle="rgba(59,110,168,.20)";oc.fillRect(x0,0,Math.max(2,x1-x0),Ht-16);oc.strokeStyle=cv("--cool");oc.strokeRect(x0+.5,.5,Math.max(2,x1-x0),Ht-17);}
    if(state.hover>=0){const x=ML+pw*state.hover/n;oc.strokeStyle=cv("--ink");oc.globalAlpha=.55;oc.beginPath();oc.moveTo(x+.5,0);oc.lineTo(x+.5,Ht-16);oc.stroke();oc.globalAlpha=1;}}
  const lineAt=ev=>{const r=base.getBoundingClientRect(),pw=W-ML-8;return clamp(Math.floor((ev.clientX-r.left-ML)/pw*n),0,n-1);};
  const inPlot=ev=>{const r=base.getBoundingClientRect();return ev.clientX-r.left>=ML;};
  over.addEventListener("mousemove",ev=>{if(!inPlot(ev)){hideTip();return;}const i=lineAt(ev);state.hover=i;if(state.drag){state.b=i;}
    const pw=W-ML-8,i0=Math.floor(Math.floor((i/n)*pw)*n/pw),i1=Math.min(n-1,Math.max(i0,Math.floor((Math.floor((i/n)*pw)+1)*n/pw)-1));let wi=i0;for(let q=i0;q<=i1;q++)if(Ls.ed[q]>Ls.ed[wi])wi=q;
    showTip("<b>lines "+Ls.idx[i0]+(i1>i0?"-"+Ls.idx[i1]:"")+"</b>"+(Ls.corpus[i]?" ("+esc(Ls.corpus[i])+")":"")+"<br>CER "+pct(mean(Ls.cer.slice(i0,i1+1)))+" - doubt "+f3(mean(Ls.conf.slice(i0,i1+1).map(c=>fin(c)?1-c:null)))+"<br>stroke "+f3(Ls.stroke_width[i],3)+" - gap "+f3(Ls.gap_ratio[i],3)+'<div class="txt">worst here (line '+Ls.idx[wi]+", CER "+pct(Ls.cer[wi],1)+")<br>"+esc(Ls.gt[wi].slice(0,70))+"<br>"+esc(Ls.pr[wi].slice(0,70))+"</div>",ev);draw();});
  over.addEventListener("mouseleave",()=>{state.hover=-1;hideTip();draw();});
  over.addEventListener("mousedown",ev=>{if(!inPlot(ev))return;state.a=state.b=lineAt(ev);state.drag=true;draw();});
  window.addEventListener("mouseup",()=>{if(state.drag){state.drag=false;summarise();draw();}});
  function pick(which){const bw=D.blocks;let k=0;bw.cer.forEach((v,b)=>{if(fin(v)&&(which==="worst"?v>bw.cer[k]:v<bw.cer[k]))k=b;});state.a=bw.first[k];state.b=Math.min(n-1,bw.first[k]+bw.size-1);summarise();draw();}
  function summarise(){const i0=Math.min(state.a,state.b),i1=Math.max(state.a,state.b);let es=0,ls=0,et=0,lt=0;for(let i=0;i<n;i++){et+=Ls.ed[i];lt+=Ls.rl[i];if(i>=i0&&i<=i1){es+=Ls.ed[i];ls+=Ls.rl[i];}}
    const inside=[];for(let i=i0;i<=i1;i++)inside.push(i);inside.sort((a,b)=>Ls.ed[b]-Ls.ed[a]);
    const zs=(arr)=>{const m=mean(arr),sd=Math.sqrt(mean(arr.map(x=>fin(x)?(x-m)*(x-m):null)))||1;return (mean(arr.slice(i0,i1+1))-m)/sd;};
    sel.replaceChildren(h("div",{class:"row2"},
      h("div",null,h("h3",null,"lines "+Ls.idx[i0]+" - "+Ls.idx[i1]+" ("+(i1-i0+1)+" lines)"),
        h("p",null,"CER here ",h("b",null,pct(es/ls)),", everywhere else ",h("b",null,pct((et-es)/(lt-ls))),". This stretch holds ",h("b",null,(100*es/et).toFixed(1)+" %")," of all edits in ",(100*(i1-i0+1)/n).toFixed(1)+" % of the lines."),
        h("p",{class:"note"},"Mean ink thickness is ",f3(zs(Ls.stroke_width),2)," sd and word-gap ratio ",f3(zs(Ls.gap_ratio),2)," sd from the corpus mean (negative = thinner / tighter).")),
      h("div",null,h("h3",null,"worst lines in the selection"),...inside.slice(0,4).map(i=>h("div",{html:'<span class="note">line '+Ls.idx[i]+" - "+Ls.ed[i]+" edits</span>"+diffBlock(Ls.gt[i],Ls.pr[i])})))));}
  mounts.push(()=>{layout();draw();});window.addEventListener("resize",()=>{layout();draw();});
  return frame("terrain","Error terrain",D.insights.terrain,[wrap,sel],
    "Each column is a few consecutive test lines (corpora are stored form by form, so neighbours usually share a writer). <b>Rows 1-2</b>: where the reading is wrong and where the model itself doubts it. <b>Rows 3-4</b>: pen properties measured from the pixels (never from the text). <b>Bottom</b>: CER per block of 20 lines.");}
function secAbstain(){
  const Lz=D.lorenz,A=D.abstain;if(!Lz||!A)return null;
  const W=480,Hh=340,m={l:52,r:14,t:20,b:40},pw=W-m.l-m.r,ph=Hh-m.t-m.b;
  const lor=s("svg",{viewBox:`0 0 ${W} ${Hh}`,class:"chart"});
  const X=v=>m.l+pw*v,Y=v=>m.t+ph*(1-v);
  [0,.25,.5,.75,1].forEach(v=>{lor.append(s("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:"gr"}),s("text",{x:m.l-7,y:Y(v)+4,"text-anchor":"end",class:"m"},Math.round(v*100)+"%"),s("text",{x:X(v),y:Hh-22,"text-anchor":"middle",class:"m"},Math.round(v*100)+"%"));});
  lor.append(s("line",{x1:X(0),y1:Y(0),x2:X(1),y2:Y(1),stroke:"var(--muted)","stroke-dasharray":"4 4"}));
  lor.append(s("path",{d:"M"+Lz.x.map((x,i)=>X(x).toFixed(1)+" "+Y(Lz.y[i]).toFixed(1)).join("L"),fill:"none",stroke:"var(--accent)","stroke-width":2.6}));
  [["5","5 %"],["10","10 %"],["20","20 %"]].forEach(([k,l])=>{const x=+k/100,y=Lz.shares[k];lor.append(s("circle",{cx:X(x),cy:Y(y),r:4.5,fill:"var(--accent)"}),s("text",{x:X(x)+8,y:Y(y)+16,"font-weight":600},"worst "+l+" of lines = "+Math.round(y*100)+" % of edits"));});
  lor.append(s("text",{x:m.l+pw/2,y:Hh-4,"text-anchor":"middle",class:"m"},"share of lines, worst first"),s("text",{x:12,y:m.t+ph/2,transform:`rotate(-90 12 ${m.t+ph/2})`,"text-anchor":"middle",class:"m"},"share of all character edits"));
  const rc=s("svg",{viewBox:`0 0 ${W} ${Hh}`,class:"chart"});const mxy=Math.max(A.base*1.12,.05);const Y2=v=>m.t+ph*(1-v/mxy),X2=v=>m.l+pw*v/.9;
  for(let k=0;k<=4;k++){const v=mxy*k/4;rc.append(s("line",{x1:m.l,x2:W-m.r,y1:Y2(v),y2:Y2(v),class:"gr"}),s("text",{x:m.l-7,y:Y2(v)+4,"text-anchor":"end",class:"m"},(v*100).toFixed(1)+"%"));}
  [0,.3,.6,.9].forEach(v=>rc.append(s("text",{x:X2(v),y:Hh-22,"text-anchor":"middle",class:"m"},Math.round(v*100)+"%")));
  rc.append(s("line",{x1:m.l,x2:W-m.r,y1:Y2(A.base),y2:Y2(A.base),stroke:"var(--muted)","stroke-dasharray":"2 4"}));
  const path=arr=>"M"+arr.map((v,i)=>X2(A.a[i]).toFixed(1)+" "+Y2(v).toFixed(1)).join("L");
  rc.append(s("path",{d:path(A.oracle),fill:"none",stroke:"var(--muted)","stroke-width":2,"stroke-dasharray":"6 4"}),s("path",{d:path(A.conf),fill:"none",stroke:"var(--cool)","stroke-width":2.8}));
  if(A.frag)rc.append(s("path",{d:path(A.frag),fill:"none",stroke:"var(--good)","stroke-width":2.8}),s("text",{x:m.l+8,y:m.t+ph-29,fill:"var(--good)","font-weight":600},"green: ranked by FRAG-RC expected character error"));
  const mk_=s("circle",{r:6,fill:"var(--accent)",stroke:"var(--card)","stroke-width":2});rc.append(mk_);
  rc.append(s("text",{x:m.l+8,y:m.t+ph-46,fill:"var(--cool)","font-weight":600},"blue: ranked by "+A.risk_name),s("text",{x:m.l+8,y:m.t+ph-12,class:"m"},"dashed: an oracle that knows the errors"),s("text",{x:m.l+pw/2,y:Hh-4,"text-anchor":"middle",class:"m"},"share of lines the system declines to read"),s("text",{x:12,y:m.t+ph/2,transform:`rotate(-90 12 ${m.t+ph/2})`,"text-anchor":"middle",class:"m"},"CER on the lines it does read"));
  const out=h("div",{class:"note",style:"font-size:14px;margin-top:8px"}),rng=h("input",{type:"range",min:0,max:90,value:20,"aria-label":"share of lines declined"});
  const upd=()=>{const k=+rng.value;mk_.setAttribute("cx",X2(A.a[k]));mk_.setAttribute("cy",Y2(A.conf[k]));out.innerHTML="Decline the <b>"+k+" %</b> least reliable lines &rarr; CER on the rest <b>"+pct(A.conf[k])+"</b>"+(A.frag?" by model doubt, <b>"+pct(A.frag[k])+"</b> by FRAG-RC risk":"")+" (oracle "+pct(A.oracle[k])+", no declining "+pct(A.base)+"). The declined lines hold <b>"+(A.share[k]*100).toFixed(0)+" %</b> of all edits.";};
  rng.addEventListener("input",upd);upd();
  return frame("abstain","Where the errors live",D.insights.abstain,[h("div",{class:"row2"},h("div",{class:"card"},h("h3",null,"Concentration of the errors"),lor),h("div",{class:"card"},h("h3",null,"Read less, err less"),rc,rng,out)),
    h("p",{class:"note"},"Area under the risk-coverage curve (lower is better): model doubt "+f3(A.aurc_conf,4)+(A.frag?", FRAG-RC expected error "+f3(A.aurc_frag,4):"")+", oracle "+f3(A.aurc_oracle,4)+".")],
    "<b>Left</b>: sort the lines from worst to best and add up their edits. <b>Right</b>: drag the slider - the system declines the lines it is least sure about and the remaining lines are read better.");}
function secBins(){
  const all=[...D.bins,D.lenbins].filter(b=>b&&b.rows&&b.rows.some(r=>r.cer!==null));if(!all.length)return null;
  const mx=Math.max(...all.flatMap(b=>b.rows.map(r=>r.ci?r.ci[1]:(r.cer||0))))*1.08;
  const grid=h("div",{class:"grid5"});
  all.forEach(b=>{const W=320,Hh=210,m={l:42,r:8,t:26,b:46},pw=W-m.l-m.r,ph=Hh-m.t-m.b,bw=pw/b.rows.length;const Y=v=>m.t+ph*(1-v/mx);
    const svg=s("svg",{viewBox:`0 0 ${W} ${Hh}`,class:"chart"});
    for(let k=0;k<=3;k++){const v=mx*k/3;svg.append(s("line",{x1:m.l,x2:W-m.r,y1:Y(v),y2:Y(v),class:"gr"}),s("text",{x:m.l-5,y:Y(v)+4,"text-anchor":"end",class:"m"},(v*100).toFixed(0)+"%"));}
    b.rows.forEach((r,k)=>{if(r.cer===null)return;const x=m.l+k*bw+bw*.16,w=bw*.68;
      svg.append(s("rect",{x,y:Y(r.cer),width:w,height:Math.max(1,m.t+ph-Y(r.cer)),rx:3,fill:"var(--accent)",opacity:.45+.5*(k/(b.rows.length-1||1)),onmousemove:ev=>showTip("<b>"+esc(b.label)+"</b> "+r.lo.toFixed(3)+" - "+(r.hi>1e8?"max":r.hi.toFixed(3))+"<br>"+r.n+" lines<br>CER "+pct(r.cer)+(r.ci?" ["+pct(r.ci[0],1)+", "+pct(r.ci[1],1)+"]":"")+(Math.abs(r.cer_final-r.cer)>1e-6?"<br>after validated options "+pct(r.cer_final):""),ev),onmouseleave:hideTip}));
      if(r.ci)svg.append(s("line",{x1:x+w/2,x2:x+w/2,y1:Y(r.ci[0]),y2:Y(r.ci[1]),stroke:"var(--ink)","stroke-width":1.5}),s("line",{x1:x+w/2-4,x2:x+w/2+4,y1:Y(r.ci[1]),y2:Y(r.ci[1]),stroke:"var(--ink)"}),s("line",{x1:x+w/2-4,x2:x+w/2+4,y1:Y(r.ci[0]),y2:Y(r.ci[0]),stroke:"var(--ink)"}));
      if(Math.abs(r.cer_final-r.cer)>1e-6)svg.append(s("path",{d:`M${x+w/2} ${Y(r.cer_final)-5}l5 5l-5 5l-5 -5z`,fill:"var(--card)",stroke:"var(--good)","stroke-width":2}));
      svg.append(s("text",{x:x+w/2,y:Hh-m.b+14,"text-anchor":"middle",class:"m"},b.key==="length"?(r.hi>1e8?">"+r.lo:(r.lo?(r.lo+1)+"-"+r.hi:"<="+r.hi)):(r.lo+(r.hi-r.lo)/2).toFixed(b.key==="slant_deg"?1:3)),s("text",{x:x+w/2,y:Hh-m.b+28,"text-anchor":"middle",class:"m"},"n="+r.n));});
    svg.append(s("text",{x:m.l,y:14,"font-weight":600},b.label+(fin(b.spearman)?"   rho = "+b.spearman.toFixed(2):"")),s("text",{x:W-m.r,y:Hh-4,"text-anchor":"end",class:"m"},b.direction+" (quintiles)"));
    grid.append(h("div",{class:"card"},svg));});
  return frame("bins","Who is hard to read",D.insights.bins,[grid],
    "Test lines are cut into quintiles of a <b>measured</b> property; bars are corpus CER with a form-clustered 95 % interval. A green diamond marks the CER after the options adopted on validation.");}
function secTrain(){
  const T=D.train;if(!T)return null;const S=T.series,xs=S.epoch.map(v=>Math.round(v)+ (S.epoch[0]===0?1:0));
  const perc=Object.keys(S).filter(k=>k.startsWith("val_cer_")).map((k,i)=>({name:"validation: "+k.slice(8),vals:S[k],color:["var(--purple)","var(--warn)","var(--good)","var(--muted)"][i%4],dash:"3 3"}));
  const cA=lineChart({xs,ylog:true,ymin:.002,ymax:1,yfmt:v=>(v*100).toFixed(v<.01?1:0)+"%",h:300,markers:[{x:T.best_epoch,label:"checkpoint"}],
    series:[{name:"labeled train subset",vals:S.train_eval_cer,color:"var(--good)"},{name:S.val_sel_cer?"val_sel (selects checkpoint)":"validation",vals:S.val_sel_cer||S.val_cer,color:"var(--accent)"},
      ...(S.val_gate_cer?[{name:"val_gate (never selects)",vals:S.val_gate_cer,color:"var(--cool)"}]:[]),...perc].filter(se=>se.vals)});
  const fk=Object.keys(S).filter(k=>k.startsWith("frag_unl_err_sel_s")).sort();
  let cB=null,capB="";
  if(S.train_frag_char_trust||fk.length){
    const ser=[];if(S.train_frag_char_trust)ser.push({name:"trusted share of characters (training batches)",vals:S.train_frag_char_trust,color:"var(--good)"});
    const cols=["var(--accent)","var(--warn)","var(--cool)","var(--purple)"];fk.forEach((k,i)=>ser.push({name:"error of trusted chars, cell "+k.slice(18)+" (unlabeled sample)",vals:S[k],color:cols[i%4],dash:"5 4"}));
    cB=lineChart({xs,ymin:0,ymax:Math.max(.12,...ser.flatMap(x=>x.vals.filter(fin)))*1.05,yfmt:v=>(v*100).toFixed(0)+"%",h:280,series:ser,markers:[{x:T.best_epoch,label:"checkpoint"}]});
    capB="<b>FRAG-RC</b>: the green line is the share of emitted characters the trust rule lets teach; the dashed lines are the realised error of those trusted characters on a sample of unlabeled lines (hidden text, diagnostic only) - they should stay near the target error.";}
  else if(S.pl_coverage){
    cB=lineChart({xs,ymin:0,ymax:Math.max(.2,...S.pl_coverage.filter(fin))*1.15,yfmt:v=>(v*100).toFixed(0)+"%",h:260,markers:[{x:T.best_epoch,label:"checkpoint"}],
      series:[{name:"line-level pseudo-label coverage",vals:S.pl_coverage,color:"var(--good)"},...(S.rc_calib_cov_raw?[{name:"raw coverage of the risk rule",vals:S.rc_calib_cov_raw,color:"var(--accent)",dash:"5 4"}]:[])]});
    capB="<b>Legacy line-level pseudo-labels</b>: the share of unlabeled lines that passed the rule.";}
  return frame("train","How it learned",D.insights.train,[h("div",{class:"card"},cA),cB?h("div",{class:"card"},cB):null],
    "<b>Top</b> (log scale): green = how well the model reads the labeled lines it trains on, orange/blue = lines it never trained on. "+(capB?"<br>"+capB:""));}
function secTrust(){
  const F=D.frag,S=D.train&&D.train.series;if(!F&&!(S&&Object.keys(S).some(k=>k.startsWith("frag_cal_chars"))))return null;
  const body=[];let ins="";
  if(F){
    ins=(F.trusted_share>0&&fin(F.err_trusted))
      ?"Applied unchanged to the evaluation split, the FRAG-RC rule trusts "+pct(F.trusted_share,1)+" of the emitted characters; the trusted ones are wrong "+pct(F.err_trusted,2)+" of the time, the doubtful ones "+pct(F.err_doubtful,1)+" (target "+pct(F.target,1)+"). AUROC of the learned risk "+f3(F.auroc_risk,3)+" vs peak probability "+f3(F.auroc_conf,3)+"."
      :"The FRAG-RC rule trusts none of the emitted characters on the evaluation split (the characters it reads are wrong "+pct(F.err_all,1)+" of the time, target "+pct(F.target,1)+"): either the model is not yet reliable enough for any threshold to meet the target, or the rule calibrated on the training writers does not transfer.";
    body.push(h("div",{class:"card"},h("table",null,h("thead",null,h("tr",null,["cell","lines","characters","trusted share","error of trusted","error of doubtful","target"].map((t,i)=>h("th",{class:i?"r":""},t)))),
      h("tbody",null,F.strata.map(r=>h("tr",null,h("td",null,String(r.stratum)),h("td",{class:"r"},r.lines),h("td",{class:"r"},r.chars),h("td",{class:"r"},pct(r.trusted_share,1)),h("td",{class:"r"},pct(r.err_trusted,2)),h("td",{class:"r"},pct(r.err_doubtful,1)),h("td",{class:"r"},pct(F.target,1)))))),
      h("p",{class:"note"},"Cell = corpus x hardness tertile (0 = least hard: thick strokes, wide gaps); cut points come from the training images.")));}
  if(S){const ks=Object.keys(S).filter(k=>k.startsWith("frag_cal_cov_s")).sort();const xs=S.epoch.map(v=>Math.round(v)+1);
    if(ks.length){const cols=["var(--accent)","var(--warn)","var(--cool)","var(--purple)"];
      const ch=lineChart({xs,ymin:0,ymax:1,yfmt:v=>(v*100).toFixed(0)+"%",h:260,series:ks.map((k,i)=>({name:"calibration coverage, cell "+k.slice(14),vals:S[k],color:cols[i%4]}))});
      if(ch)body.push(h("div",{class:"card"},h("h3",null,"Share of calibration characters the rule trusts, per cell"),ch));}}
  return frame("trust","What the model trusts",ins,body,"FRAG-RC replaces line-level pseudo-labels. A <b>character</b> is trusted when a label-free evidence score falls below a per-cell threshold chosen so that the error of the trusted characters stays under the target on held-out calibration lines.");}
function secWsd(){
  const W=D.wsd;if(!W)return null;const e=D.ledger.find(x=>x.id==="wsd");
  const slope=(title,by)=>{const keys=Object.keys(by||{});if(!keys.length)return null;const Wd=440,Hh=78+keys.length*26,m=110,xr=Wd-60;
    const vs=keys.flatMap(k=>[by[k].CER_before,by[k].CER_after]).filter(fin);if(!vs.length)return null;
    let lo=Math.min(...vs),hi=Math.max(...vs);const pad=Math.max((hi-lo)*0.18,0.002);lo=Math.max(0,lo-pad);hi=hi+pad;const X=v=>m+(xr-m)*(v-lo)/(hi-lo);
    const svg=s("svg",{viewBox:`0 0 ${Wd} ${Hh}`,class:"chart"});svg.append(s("text",{x:0,y:16,"font-weight":600},title),
      s("circle",{cx:m,cy:31,r:4,fill:"var(--muted)"}),s("text",{x:m+9,y:35,class:"m"},"before"),s("circle",{cx:m+70,cy:31,r:5,fill:"var(--good)"}),s("text",{x:m+80,y:35,class:"m"},"after (green = lower CER)"));
    keys.forEach((k,i)=>{const y=56+i*26,c=by[k],better=c.CER_after<c.CER_before;svg.append(s("text",{x:0,y:y+4,class:"m"},"cluster "+k+" ("+c.lines+")"),s("line",{x1:X(c.CER_before),x2:X(c.CER_after),y1:y,y2:y,stroke:better?"var(--good)":"var(--purple)","stroke-width":3}),
      s("circle",{cx:X(c.CER_before),cy:y,r:4,fill:"var(--muted)"}),s("circle",{cx:X(c.CER_after),cy:y,r:5,fill:better?"var(--good)":"var(--purple)"}),s("text",{x:Wd-52,y:y+4},pt(c.CER_after-c.CER_before)));});
    const ya=56+keys.length*26-6;svg.append(s("line",{x1:m,x2:xr,y1:ya,y2:ya,class:"ax"}),s("text",{x:m,y:ya+16,class:"m"},pct(lo,1)),s("text",{x:xr,y:ya+16,class:"m","text-anchor":"end"},pct(hi,1)),s("text",{x:(m+xr)/2,y:ya+16,class:"m","text-anchor":"middle"},"CER (axis does not start at 0)"));return svg;};
  const ins=e?("Writer self-distillation was "+(e.adopted?"ADOPTED":"not adopted")+" on val_gate (adjusted upper bound of the gain "+pt(e.val_hi_adj)+"). Test CER "+pct(e.cer)+(fin(e.test_gain)?" ("+pt(e.test_gain)+" vs. the reading it was compared with).":".")):"";
  return frame("wsd","Learning a hand",ins,[h("div",{class:"row2"},h("div",{class:"card"},slope("val_gate lines (decides adoption)",W.val_by_cluster)),h("div",{class:"card"},slope("test lines (reports only)",W.test_by_cluster)))],
    "The evaluation lines are grouped by <b>measured</b> pen style (k-means). For each group a copy of the model takes a few label-free steps on that group's own images. Steps: "+W.chosen_steps+"; transductive.");}
function secNbest(){
  const N=D.nbest;if(!N||!N.examples||!N.examples.length)return null;
  const cards=N.examples.map(e=>h("div",{class:"card"},h("div",{class:"note"},"line "+e.line+" - "+e.tag+" ("+(e.gain_edits>0?"-":"+")+Math.abs(e.gain_edits)+" edits)"),
      h("table",null,h("thead",null,h("tr",null,["CTC rank","hypothesis (differences from the truth marked)","CTC log-prob","LM log-prob","fused score","edits"].map((t,i)=>h("th",{class:i>1?"r":""},t)))),
        h("tbody",null,e.items.map((x,k)=>{const [,hy]=diff(e.gt,x.text);return h("tr",{style:k===e.chosen?"background:rgba(31,122,109,.13)":null},
          h("td",null,"#"+(x.rank+1)+(x.rank===0?" (top-1)":"")+(k===e.chosen?" - chosen":"")),h("td",{class:"txt",html:hy}),h("td",{class:"r"},f3(x.ctc,2)),h("td",{class:"r"},f3(x.lm,2)),
          h("td",{class:"r"},f3(x.fused,2)),h("td",{class:"r"},x.ed));}))),
      h("div",{class:"txt note",style:"margin-top:6px"},"truth: "+e.gt)));
  const ins="The "+(N.llm||"language model")+" "+(N.adopted?"was adopted":"was NOT adopted")+" on val_gate. It changed the top-1 reading of "+pct(N.changed_share,1)+" of the "+N.lines+" lines: it helped "+N.helped_lines+" lines, hurt "+N.hurt_lines+", net "+N.net_edits_saved+" edits saved (beta = "+N.beta+", rho = "+N.rho+").";
  return frame("nbest","How the language model chooses",ins,[...cards],"Each table is the recogniser's own N-best list for one line. The language model never writes anything; it only re-ranks. <b>Green</b> = the hypothesis the fused score picked.");}
function secInterp(){
  const I=D.interp;if(!I||!Object.keys(I).length)return null;const body=[];
  if(I.layer_probes&&I.layer_probes.layers){const L=I.layer_probes;body.push(h("div",{class:"card"},h("h3",null,"Where does the reading form? (linear probes per layer)"),h("table",null,h("thead",null,h("tr",null,["layer","character probe accuracy","slant probe R2","frames"].map((t,i)=>h("th",{class:i?"r":""},t)))),
    h("tbody",null,L.layers.map(r=>h("tr",null,h("td",null,r.name),h("td",{class:"r"},pct(r.char_probe_accuracy,1)),h("td",{class:"r"},f3(r.slant_probe_r2,2)),h("td",{class:"r"},r.n_frames))))),h("p",{class:"note"},"majority class accuracy "+pct(L.majority_class_accuracy,1)+". "+L.note)));}
  if(I.uncertainty_decomposition&&I.uncertainty_decomposition.signals){const U=I.uncertainty_decomposition;body.push(h("div",{class:"card"},h("h3",null,"Which uncertainty signal predicts a wrong line?"),h("table",null,h("thead",null,h("tr",null,["signal","AUROC (line has error)","CER of kept lines at 80 % coverage"].map((t,i)=>h("th",{class:i?"r":""},t)))),
    h("tbody",null,Object.entries(U.signals).map(([k,v])=>h("tr",null,h("td",null,k),h("td",{class:"r"},f3(v.auroc_line_has_error,3)),h("td",{class:"r"},pct(v.CER_of_kept_lines_at_80pct_coverage)))))),h("p",{class:"note"},U.n_lines+" lines, "+pct(U.line_error_rate,1)+" with at least one error.")));}
  if(I.longline_stress&&fin(I.longline_stress.CER_concatenated)){const S=I.longline_stress;body.push(h("div",{class:"card"},h("h3",null,"Long-line stress test"),h("p",null,S.pairs+" concatenated pairs (mean "+S.mean_length_chars.toFixed(0)+" chars): CER "+pct(S.CER_concatenated)+" vs "+pct(S.CER_same_lines_read_alone)+" for the same lines read alone (gap "+pt(S.generalisation_gap)+").")));}
  if(I.style_neighbours&&I.style_neighbours.neighbour_vs_random_descriptor_gap_ratio){const R=I.style_neighbours.neighbour_vs_random_descriptor_gap_ratio;body.push(h("div",{class:"card"},h("h3",null,"What the style space groups"),h("p",null,"Mean |descriptor difference| of embedding neighbours / random pairs: "+Object.entries(R).map(([k,v])=>k+" "+f3(v,2)).join(", ")+" (below 1 = neighbours have similar measured style).")));}
  if(!body.length)return null;
  return frame("interp","Inside the model",null,body,"Details, CSVs and figures: 10_explanations/, 02_uncertainty/, 07_robustness/, 05_style/ next to this report.");}
function secTraces(){
  const T=D.traces;if(!T||!T.length)return null;
  const cards=T.map(t=>{const wrap=h("div",{class:"imgwrap"},h("img",{src:t.image,alt:"line "+t.line}));
    t.chars.forEach(c=>{const d=h("div",{class:"ch "+(c.trusted?"t":"d")+(c.wrong?" w":""),style:`left:${(100*c.x0).toFixed(2)}%;width:${(100*(c.x1-c.x0)).toFixed(2)}%`});
      d.addEventListener("mousemove",ev=>showTip("<b>'"+esc(c.ch)+"'</b> "+(c.trusted?"trusted":"doubtful")+" - risk "+f3(c.risk,3)+"<br>peak prob "+f3(c.p,3)+(c.alt?"<br>runner-up '"+esc(c.alt)+"' "+f3(c.palt,3):"")+(c.wrong?"<br><b>wrong</b>":""),ev));d.addEventListener("mouseleave",hideTip);wrap.append(d);});
    return h("div",{class:"card trace"},h("div",{class:"note"},"line "+t.line+" - "+esc(t.tag)+" - CER "+pct(t.cer,1)+(t.rtl?" - right-to-left (text shown in logical order)":"")),wrap,h("div",{html:diffBlock(t.gt,t.pr,t.pw!==null&&t.pw!==undefined?{label:"after WSD",text:t.pw}:null)}));});
  return frame("traces","Reading traces","Every emitted character is drawn over the ink that produced it: green = the trust rule would let it teach, amber = doubtful; a red underline marks a character that is actually wrong.",[...cards],
    "Frames are 4 pixels wide. Hover a box for its probability and the runner-up. The gallery holds the worst lines, the longest perfectly-read lines, random lines and the lines the writer self-distillation helped / hurt most.");}
function secExplore(){
  const Ls=D.lines,n=Ls.idx.length;const st={sort:"cer",q:"",page:0,minlen:0,strat:"all",ling:true};const PS=15;const isLing=g=>/[\p{L}\p{N}]/u.test(g);
  const cells=[...new Set(Ls.strat)].sort((a,b)=>a-b);
  const sorts={cer:["worst CER",i=>-Ls.cer[i]],ed:["most edits",i=>-Ls.ed[i]],doubt:["most doubt",i=>-(fin(Ls.conf[i])?1-Ls.conf[i]:0)],gain:["biggest gain from options / WSD",i=>-(Ls.ed[i]-(Ls.edw?Ls.edw[i]:Ls.edf[i]))],loss:["biggest loss from options / WSD",i=>(Ls.ed[i]-(Ls.edw?Ls.edw[i]:Ls.edf[i]))],idx:["line index",i=>Ls.idx[i]]};
  const list=h("div"),pg=h("div",{class:"ctrl"});
  const ctrl=h("div",{class:"ctrl"},h("select",{onchange:e=>{st.sort=e.target.value;st.page=0;render();}},Object.keys(sorts).map(k=>h("option",{value:k},sorts[k][0]))),
    h("input",{type:"text",placeholder:"search a word or a line number",oninput:e=>{st.q=e.target.value.toLowerCase();st.page=0;render();}}),
    h("select",{onchange:e=>{st.strat=e.target.value;st.page=0;render();}},h("option",{value:"all"},"all cells"),...cells.map(k=>h("option",{value:k},"cell "+k))),
    h("select",{onchange:e=>{st.minlen=+e.target.value;st.page=0;render();}},h("option",{value:0},"any length"),h("option",{value:40},">= 40 chars"),h("option",{value:65},">= 65 chars")),
    h("select",{onchange:e=>{st.ling=e.target.value==="1";st.page=0;render();}},h("option",{value:1},"handwriting lines only"),h("option",{value:0},"include separator lines")));
  function render(){let ix=[];for(let i=0;i<n;i++){if(Ls.rl[i]<st.minlen)continue;if(st.ling&&!isLing(Ls.gt[i]))continue;if(st.strat!=="all"&&Ls.strat[i]!==+st.strat)continue;
      if(st.q&&!(String(Ls.idx[i])===st.q||Ls.gt[i].toLowerCase().includes(st.q)||Ls.pr[i].toLowerCase().includes(st.q)))continue;ix.push(i);}
    const f=sorts[st.sort][1];ix.sort((a,b)=>f(a)-f(b)||a-b);const np=Math.max(1,Math.ceil(ix.length/PS));st.page=Math.min(st.page,np-1);
    list.replaceChildren(...ix.slice(st.page*PS,(st.page+1)*PS).map(i=>h("div",{class:"card",style:"margin:8px 0;padding:10px 14px"},
      h("div",{class:"note"},"line "+Ls.idx[i]+(Ls.corpus[i]?" ["+esc(Ls.corpus[i])+"]":"")+" - "+Ls.ed[i]+" edits / "+Ls.rl[i]+" chars - CER "+pct(Ls.cer[i],1)+" - confidence "+f3(Ls.conf[i],2)+" - cell "+Ls.strat[i]+(Ls.edw?" - WSD edits "+Ls.edw[i]:"")),
      h("div",{html:diffBlock(Ls.gt[i],Ls.pr[i],Ls.pw&&Ls.pw[i]!==Ls.pr[i]?{label:"after WSD",text:Ls.pw[i]}:(Ls.pf[i]!==Ls.pr[i]?{label:"after options",text:Ls.pf[i]}:null))}))));
    pg.replaceChildren(h("button",{onclick:()=>{st.page=Math.max(0,st.page-1);render();}},"< prev"),h("span",{class:"note"},"page "+(st.page+1)+" / "+np+" - "+ix.length+" lines"),h("button",{onclick:()=>{st.page=Math.min(np-1,st.page+1);render();}},"next >"));}
  mounts.push(render);
  return frame("explore","Line explorer","Every test line, with the reading laid over the truth.",[h("div",{class:"card"},ctrl,list,pg)],
    "<span style='background:rgba(181,83,47,.22);padding:0 4px;border-radius:3px'>substitution</span> <span style='background:rgba(197,138,26,.35);padding:0 4px;border-radius:3px'>missing in the reading</span> <span style='background:rgba(122,79,163,.30);padding:0 4px;border-radius:3px'>extra in the reading</span>; a visible space mark (&#9251;) shows errors on word gaps.");}
function secIntegrity(){
  const I=D.integrity,P=I.protocol,body=[];
  if(P){body.push(h("div",{class:"card"},h("h3",null,"Which split is used for what"),h("table",null,h("tbody",null,Object.entries(P.roles||{}).map(([k,v])=>h("tr",null,h("td",{style:"width:32%;font-weight:600"},k),h("td",null,v))))),
    h("p",{class:"note"},"validation = "+(P.val_sel_lines)+" lines for checkpoint selection (val_sel) + "+(P.val_gate_lines)+" lines for every post-hoc decision (val_gate), disjoint whole blocks of "+P.val_block_lines+" lines"+(P.val_split_enabled?"":" - SPLIT SWITCHED OFF in this run")+"."+(P.corpora?" Corpora: "+P.corpora.join(", ")+".":""))));}
  else body.push(h("div",{class:"card"},h("p",null,"No protocol.json was found for this run (an older run).")));
  body.push(h("div",{class:"card"},h("h3",null,"Looks at the evaluation split"),(I.looks&&I.looks.length)?h("table",null,h("thead",null,h("tr",null,["time","split","checkpoint","epoch"].map(t=>h("th",null,t)))),h("tbody",null,I.looks.map(l=>h("tr",null,h("td",null,l.time),h("td",null,l.split),h("td",{class:"mono"},String(l.checkpoint||"")),h("td",null,String(l.epoch===null||l.epoch===undefined?"":l.epoch)))))):h("p",{class:"note"},"No log of looks available.")));
  body.push(h("div",{class:"card cav"},h("h3",null,"What this report does not establish"),h("ul",null,D.caveats.map(c=>h("li",null,c)))));
  return frame("integrity","Integrity",null,body,"Every panel above is computed from the files in this folder. The numbers behind each panel sit in the numbered sub-folders next to this page.");}
function build(){
  const meta=D.meta,H=D.head;const app=document.getElementById("app");
  const secs=[secLedger(),secTerrain(),secAbstain(),secBins(),secTrain(),secTrust(),secWsd(),secNbest(),secInterp(),secTraces(),secExplore(),secIntegrity()].filter(Boolean);
  const kpi=(v,l,sub)=>h("div",{class:"kpi"},h("div",{class:"v"},v),h("div",{class:"l"},l),sub?h("div",{class:"s"},sub):null);
  const top=h("header",{class:"top"},h("h1",null,meta.title),
    h("p",null,"A reading of one run: where the model is right, where it is not, what it believes about its own characters, and what each step of the pipeline contributed. Everything on this page is computed from the run's own files."),
    h("div",{class:"badges"},meta.synthetic?h("span",{class:"badge warn"},"SYNTHETIC SMOKE TEST - fake handwriting - NOT FOR PUBLICATION"):h("span",{class:"badge good"},"real data"),
      h("span",{class:"badge"},"split: "+meta.split+" ("+meta.n+" lines)"),h("span",{class:"badge"},meta.version),(meta.corpora&&meta.corpora.length)?h("span",{class:"badge"},"corpora: "+meta.corpora.join(", ")):null,
      H.looks?h("span",{class:"badge"},"looks at this split so far: "+H.looks):null,H.ckpt_epoch?h("span",{class:"badge"},"checkpoint epoch "+H.ckpt_epoch+(H.weights_from?" ("+H.weights_from+")":"")):null,h("span",{class:"badge"},"built "+meta.generated)),
    h("div",{class:"kpis"},kpi(pct(H.cer),"character error rate",H.ci?"95 % form-clustered CI "+pct(H.ci[0],1)+" - "+pct(H.ci[1],1):null),
      kpi(pct(H.final_cer),"system CER with every validated option"),kpi(pct(H.wer,1),"word error rate",H.wer_ci?"CI "+pct(H.wer_ci[0],1)+" - "+pct(H.wer_ci[1],1):null),
      kpi(pct(H.exact,1),"lines read exactly"),H.labeled_fraction!==null&&H.labeled_fraction!==undefined?kpi(Math.round(H.labeled_fraction*100)+" %","of the training lines labeled",H.calibration_fraction?"+ "+Math.round(H.calibration_fraction*100)+" % held out to calibrate trust":null):null));
  const nav=h("nav",null,h("div",{class:"brand"},"Reading Atlas"),h("div",{class:"sub"},meta.run),...SEC.map(([id,t],i)=>h("a",{href:"#"+id,"data-id":id},h("b",null,i+1),t)));
  app.append(h("div",{class:"layout"},nav,h("main",null,top,...secs)),tip);
  mounts.forEach(f=>f());
  const links=[...nav.querySelectorAll("a")];
  const io=new IntersectionObserver(es=>{es.forEach(e=>{if(e.isIntersecting){links.forEach(a=>a.classList.toggle("on",a.dataset.id===e.target.id));}});},{rootMargin:"-30% 0px -60% 0px"});
  SEC.forEach(([id])=>{const el=document.getElementById(id);if(el)io.observe(el);});
}
window.addEventListener("DOMContentLoaded",()=>{try{build();}catch(e){document.body.append(h("pre",{style:"color:#b00020;padding:20px"},"Report error: "+e.message+"\n"+e.stack));console.error(e);}});
})();
'''

_REPORT_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>InkLingua - Reading Atlas</title>
<style>/*__CSS__*/</style></head>
<body><div id="app"></div>
<script>window.__D = /*__DATA__*/null;</script>
<script>/*__JS__*/</script>
</body></html>
'''
