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
main(Config.smoke())                                   # synthetic plumbing test, a few minutes on CPU
main()                                                 # IAM-line, 30 % labels, FRAG-RC + WSD
cfg = Config(); cfg.run.mode = "suite"; main(cfg)      # baselines + ablations x 3 seeds -> suite_summary.md (what a paper needs)
main(Config.multi(("iam", "rimes", "khatt")))          # English + French + Arabic (right-to-left) at once
cfg = Config(); cfg.run.mode = "finetune"              # continue a trained checkpoint on another corpus (alphabet grows)
cfg.train.init_from = "inklingua_runs/inklingua_iam/checkpoints/best.pt"; cfg.data.sources = ("esposalles",); main(cfg)
```

**Kaggle (GPU T4 x2):** upload `notebooks/InkLingua_Kaggle_T4x2.ipynb`; it is self-contained (section 2 writes the package), puts the
recogniser on `cuda:0` and the LLM on `cuda:1`, and resumes from `/kaggle/working` across sessions.  Regenerate it after changing the
package with `python tools/build_kaggle_notebook.py`.

There is no argparse: every option is a dataclass default in `inklingua/config.py`.  Modes: `single`, `suite`,
`suite_summary`, `finetune`, `adapt` / `adapt_curve` / `source_calib` (zero-shot adaptation to a folder of line images in a
new language), `report` (rebuild the report of an existing run), `smoke`.

## What is where

| module | question it answers |
|---|---|
| `inklingua/config.py` | every option; `Config.labeled_only()`, `Config.ablation(...)`, `Config.multi(...)`, `Config.smoke()` |
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
| `inklingua/selftest.py` | 17 fast invariant tests (run before every real run) + the synthetic smoke |

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

* Verified here: 17 self-tests and the synthetic end-to-end smoke (train, resume, inference, report) on CPU.  Facts about the
  datasets and the literature were checked online on 2026-11-04 (`docs/AUDIT_v10_to_v11.md`, section C).
* **Unverified**: the real-IAM benefit of every component, including whether unlabeled data helps at all.  Run the suite.
* The LLM translation evaluation has no human reference; the LLM's translation of the true text is the proxy.
* IAM is distributed for non-commercial research; the LLM may have seen the LOB corpus IAM is drawn from (report LLM-rescored
  numbers as "with an external language model").

See `docs/CHANGELOG.md` for the v10 -> v11 changes and `docs/AUDIT_v10_to_v11.md` for the defects that were fixed.
