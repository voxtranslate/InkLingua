# InkLingua

Style-disentangled, uncertainty-aware handwriting-to-translation: a line of handwriting is read by a CTC recogniser whose
unlabeled lines teach it through **fragment-level, risk-controlled self-distillation (FRAG-RC)**, adapted to each writer by
**writer self-distillation (WSD)**, rescored by an LLM that may only choose among the recogniser's own hypotheses, translated
into any language, and rendered back in the predicted handwriting style.  Every option is validated on a split the test set
never sees, and every number of an evaluation comes with the file it was computed from.

This repository is **v11**: the v10 single-file script restructured into a readable package, audited line by line
(`docs/AUDIT_v10_to_v11.md`), with multi-corpus / multi-script training, warm starts on new corpora, a structured
interpretability suite and a stronger experiment suite.  Nothing in it has been run on real IAM yet: every claim about a
component's benefit is **unverified until the suite is run**.

## Pipeline

```
handwritten line image
   -> InkLinguaNet  (CNN -> dual-granularity VQ bottleneck -> style-FiLM Transformer -> CTC)
   -> text + per-character confidence + N-best hypotheses
   -> pretrained LLM (Qwen-style causal LLM, or Marian seq2seq) -> translation
   -> style-matched rendering of the translation (closed loop, uses the predicted style)
```

Connected task: the writer's **style** (slant, stroke width, ink density, word gap, body ratio, measured on the image, never
invented) is predicted from the same sample, used to modulate the recogniser (FiLM), to stratify the trust rule, to cluster
writers at test time and to render the translation.

## Quick start

```bash
pip install -r requirements.txt
python inklingua.py                 # the default IAM-line experiment (downloads Teklia/IAM-line, pinned commit)
```

In a notebook cell:

```python
from inklingua import Config, main
main()                                                 # IAM-line, 30 % labels, FRAG-RC + WSD
cfg = Config(); cfg.run.mode = "suite"; main(cfg)      # baselines + ablations x 3 seeds -> suite_summary.md (what a paper needs)
main(Config.multi(("iam", "rimes", "khatt")))          # English + French + Arabic (right-to-left) at once
cfg = Config(); cfg.run.mode = "finetune"              # continue a trained checkpoint on another corpus (alphabet grows)
cfg.train.init_from = "inklingua_runs/inklingua_iam/checkpoints/best.pt"; cfg.data.sources = ("esposalles",); main(cfg)
```

**The article's experiments (Kaggle GPU T4 x2):** `notebooks/InkLingua_Kaggle_T4x2.ipynb` is ONE readable cell that contains the whole
code -- every class and function of the package, one section per module in reading order, with a table of contents -- and one entry
point (`notebook_main`).  It is self-contained: it needs neither this repository nor any package of it, and contains no test code
and no generated data.  Each run executes in its own process from the cell's own source, written to `_plan/inklingua_notebook.py`.  It runs
the complete experiment plan on the real corpora (5 core variants, 14 training ablations and a label-budget sweep, for every corpus and 3
seeds; then a joint model and a zero-shot adaptation stage), two runs in parallel on the two T4s, and resumes in the next session.  It
writes the paper tables (Markdown, LaTeX, CSV), figures, statistics and a reproducibility report.  See `docs/EXPERIMENT_PLAN.md`,
including its cost: the full plan is about 2,350 T4 GPU-hours (`PLAN_OVERRIDES = {"dry_run": True}` checks the GPUs and prepares the
first corpus in minutes).  The package is the source of truth; regenerate the notebook after changing it with
`python tools/build_kaggle_notebook.py`.

There is no argparse: every option is a dataclass default in `inklingua/config.py`.  Modes: `single`, `suite`,
`suite_summary`, `finetune`, `adapt` / `adapt_curve` / `source_calib` (zero-shot adaptation to a folder of line images in a
new language), `report` (rebuild the report of an existing run).  Repository tests: `pytest` (fast) and `pytest -m slow` (end-to-end smoke).

## What is where

| module | question it answers |
|---|---|
| `inklingua/config.py` | every option; `Config.labeled_only()`, `Config.ablation(...)`, `Config.multi(...)` |
| `inklingua/datasets_registry.py` | 12 free line-level corpora with the IAM-line schema, verified on the Hub (`docs/DATASETS.md`) |
| `inklingua/text.py` | grapheme-cluster codec, right-to-left handling, IAM detokenisation, CER / WER / chrF, form-clustered bootstrap |
| `inklingua/data.py` | preprocessing, Hub / local / synthetic sources, the dataset serving both tasks, MixLine, the multi-corpus data module |
| `inklingua/model.py` | InkLinguaNet, losses (CTC, inter-CTC, style, leakage, consistency, fragment distillation), EMA teacher, SES augmentation |
| `inklingua/frag.py`, `riskcontrol.py` | FRAG-RC: character evidence, cross-fitted trust rule, per-cell risk control with line-clustered bounds |
| `inklingua/pseudo.py` | the v9 line-level pseudo-labels (kept as the `legacy_pl` baseline) |
| `inklingua/trainer.py` | training loop, checkpoint selection on `val_sel`, resume, warm start from any checkpoint |
| `inklingua/posthoc.py` | validation-gated options: view ensemble, style AdaBN, writer self-distillation, space bonus |
| `inklingua/interpret.py` | layer probes, uncertainty decomposition, character counterfactuals, style neighbours, alignments, long-line stress |
| `inklingua/inferencer.py` | the evaluation and its 13 output folders (`docs/INFERENCE_OUTPUTS.md`) |
| `inklingua/adapt.py` | `finetune` on a labeled corpus; zero-shot adaptation ([ALPHA-SPLIT], [LEX-TIE], [XL-RC]) |
| `inklingua/report.py` | the self-contained run report (`12_report/index.html`) |
| `inklingua/suite.py` | baselines, ablations, seed-paired comparison, component-contribution table |
| `inklingua/experiments.py` | the article's experiment plan: corpus after corpus, GPU workers, resumable ledger, session deadline |
| `inklingua/schedule.py` | the training-budget rule (epochs from corpus size, script and era; IAM = the v10 defaults) |
| `inklingua/paper.py` | paper tables (md / tex / csv), figures, pooled paired bootstrap, Holm, sign test, reproducibility report |
| `inklingua/selftest.py` | 19 fast invariant tests (incl. exact mid-epoch resume) + the synthetic smoke (repository tests only; real runs and the notebook run neither) |

## Data, caching and resuming

* **Download once.**  Every corpus is read from a pinned Hub commit; once its parquet files are on disk they are found without any
  network request.  Set `HF_TOKEN` (or, on Kaggle, a secret named `HF_TOKEN`) for higher rate limits; public corpora work without it.
* **Preprocess once.**  Lines are decoded in parallel processes into a line cache (`<work_dir>/cache/lines`, on by default) that also
  stores split sizes and transcriptions: a later run of the same corpus needs neither the download nor the decoding.  Images that are
  mostly blank page (KHATT) are cut to their text band first (`data.ink_band`; tight line crops such as IAM are unchanged).
* **Resume exactly.**  Every training writes a step checkpoint at least every `cfg.train.ckpt_every_minutes` (10) and whenever it is
  stopped (SIGTERM, the stop button of a notebook, `cfg.train.deadline_unix`).  Re-running the same configuration continues at that step
  with the same batches and random state; a self-test checks that the result is bit-identical to an uninterrupted run.  The zero-shot
  adapter checkpoints every round.

## The protocol (why the numbers can be trusted)

* **Hidden labels never reach training.**  Unlabeled train lines are images only; their text is read by diagnostics alone.
  Calibration lines (part of the labeled budget) are held out of the loss; their text only measures the trust rule.
* **Two validation halves.**  `val_sel` (whole form blocks) selects the checkpoint and stops training; `val_gate` is the only
  data on which post-hoc options, LLM weights, temperature and `[?]` marks are tuned (Bonferroni-adjusted, form-clustered
  paired bootstrap).  The test split decides nothing; every look at it is logged (`protocol/test_looks.jsonl`).
* **Form-clustered intervals.**  Lines are stored form by form and errors cluster by writer; intervals resample blocks of 20
  lines (a line bootstrap is about half as wide as it should be).
* **A single run proves nothing.**  `cfg.run.mode = "suite"` trains `labeled_only` (same labeled lines, no unlabeled data),
  `ctc_only`, `legacy_pl`, and `full` minus each component, over 3 seeds, and reports seed-paired deltas.

## Datasets

IAM-line is small (6,482 training lines, English, Latin script).  The registry adds eleven corpora with the same Hub schema:
RIMES (fr), Belfort (fr), Esposalles (ca), Himanis (la/fr), HOME-Alcar (la), NewsEye (de), NorHand v1/v3 (nb), POPP (fr),
CASIA-HWDB2 (zh, Han script), KHATT (ar, right-to-left).  Each entry pins a commit and the published split sizes.  The
licences of the underlying corpora are research-only in several cases: check `docs/DATASETS.md` before publishing.

## Status and honesty

* Verified here: 19 self-tests, a real-IAM run of the experiment plan on CPU, the preprocessing of IAM + RIMES + KHATT (cold and offline-warm), and the synthetic end-to-end smoke (train, resume, inference, report) on CPU.  Facts about the
  datasets and the literature were checked online on 2026-11-04 (`docs/AUDIT_v10_to_v11.md`, section C).
* **Unverified**: the real-IAM benefit of every component, including whether unlabeled data helps at all.  Run the suite.
* The LLM translation evaluation has no human reference; the LLM's translation of the true text is the proxy.
* IAM is distributed for non-commercial research; the LLM may have seen the LOB corpus IAM is drawn from (report LLM-rescored
  numbers as "with an external language model").

See `docs/CHANGELOG.md` for the v10 -> v11 changes and `docs/AUDIT_v10_to_v11.md` for the defects that were fixed.
