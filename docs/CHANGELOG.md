# Changelog

## v11.0.0 (2026-11-04) -- restructuring, audit fixes, multi-corpus, interpretability suite

Structure
* The v10 single file (~6,000 lines) is now the `inklingua/` package: one module per question (see `inklingua/__init__.py`), a
  thin `inklingua.py` entry point, `tests/` wrappers for pytest.  Behaviour is the v10 behaviour except for the fixes below.

Fixed (full list with locations in `docs/AUDIT_v10_to_v11.md`)
* The self-tests no longer contain a synthetic training run; the real run's log handlers are no longer redirected into a deleted
  temporary directory.  The smoke is `mode="smoke"` / `tests/test_smoke.py`.
* The suite runs the self-tests once instead of once per variant x seed.
* `KeyError` after an epoch in which every step was skipped.
* Out-of-range cell index when a FRAG-RC rule meets data with more cells (clipped; strata cuts stored in `best.pt`).
* Alphabet units are grapheme clusters (combining marks stay attached); checkpoints store `units`.
* Character classes use Unicode categories (uncased scripts were counted as punctuation).
* IAM detokenisation only for corpora flagged `detokenize=True`.
* The primary translation column follows the configured prompt; arms are named.
* Dead computation removed (`style_pred` in `Trainer.evaluate`); style attention is now shown instead of being discarded.
* One DataLoader iterator per epoch (tqdm.auto started two: twice the workers).
* The line cache is a memory-mapped file shared by parallel runs and loader workers (no per-run copy of a corpus in RAM).
* KHATT: label-free ink-band crop of mostly blank line images (53 test lines were unreadable); no `max_text_len` cap (dropped labeled lines 122 -> 3 on iam+rimes+khatt).
* Exact step-level resume of every training (mid-epoch, SIGTERM, KeyboardInterrupt, deadline); per-loader seeded generators; the zero-shot adapter checkpoints every round.
* Download and preprocessing cache: offline-first pinned snapshots, a self-sufficient line cache on by default, parallel decoding, caches carried across Kaggle sessions.
* Clean logs (no self-tests in real runs, no HTTP request lines, one HF token line); derived run names; the epoch budget rule by default.
* PyTorch practice: `weights_only=True` checkpoints, batched leakage penalty and VQ reset (no per-step host syncs), `inference_mode`, `torch.amp.GradScaler`; torch >= 2.5.
* The Kaggle notebook is one READABLE cell with the complete code (sections in reading order, table of contents), generated from the package; run processes execute the cell's own source.
* Plan robustness (Kaggle log: every run on GPU 0 aborted with exit code -6 within 20 s): GPU check per GPU, corpus preparation in a CPU-only process, spawn instead of fork for decoding, fault handler in run processes, the end of a failed run's log in the notebook, a stop after repeated failures, `dry_run`.
* Kaggle (torch 2.11, CUDA 13.0, T4): the leakage probe is fitted on the CPU (no GPU linear-algebra solver in training); worker processes record their error and exit without teardown (a broken CUDA context aborted them before the error was saved); the GPU check covers every operation of a training step.
* Training curves gained FRAG-RC panels (trusted share, per-cell coverage, realised error of trusted characters).
* The suite summary reports WER, post-hoc and system CER, a component-contribution table, and can re-aggregate without retraining.

Added
* `DatasetSpec` + a registry of 12 verified free line-level corpora; multi-corpus training with per-corpus metrics, per-corpus
  FRAG-RC cells (`strata_per_domain`) and `domain_balance`.
* Right-to-left support (`DatasetSpec.rtl`, `AdaptConfig.rtl`): visual order inside, logical order outside.
* `mode="finetune"` / `TrainConfig.init_from`: warm start from any checkpoint with function-preserving alphabet growth.
* Interpretability suite (`inklingua/interpret.py`): layer probes, uncertainty decomposition, character counterfactuals, style
  neighbours, alignment export, long-line stress test, group tables.
* Structured output folders (`docs/INFERENCE_OUTPUTS.md`); the report shows per-corpus tables and the interpretability results.
* Suite presets `quick` / `paper` / `all`; ablations `no_strata`, `no_sharpen`.
* The article's experiment plan (`inklingua/experiments.py`, `docs/EXPERIMENT_PLAN.md`): every corpus in turn, 5 core variants, 14 training
  ablations, a label-budget sweep, 3 seeds, a joint model and a zero-shot adaptation stage; one subprocess per run and per GPU; a resumable
  ledger; a session deadline with evaluation reserves; checkpoint pruning and cache cleanup.
* The training-budget rule (`inklingua/schedule.py`): epochs from corpus size, script and era, IAM reproducing the v10 defaults.
* Paper outputs (`inklingua/paper.py`): tables T1-T10 in Markdown, LaTeX and CSV, figures, pooled seed-paired bootstrap, Holm correction,
  sign test, reproducibility and status reports.
* A pyarrow parquet reader (no second Arrow copy on disk), block-preserving caps with a fixed subset seed, a shared on-disk line cache,
  cross-split decontamination for the joint stage, per-line languages for the LLM, `TrainConfig.deadline_unix`, and
  `checkpoints/source_calibration.npz` written by FRAG-RC runs (the zero-shot adapter needed it and only the legacy path wrote it).
* `notebooks/InkLingua_Kaggle_T4x2.ipynb` (generated by `tools/build_kaggle_notebook.py`): ONE cell that runs the plan on Kaggle GPU T4 x2;
  `LLMConfig.device` puts the LLM next to the run that needs it.
* Literature update in the FRAG-RC docstring and the audit (DocTTT, MetaWriter, few-shot ICL writer adaptation, Geometric Risk
  Control for VLM OCR, Selective Conformal Risk Control, OT pseudo-label selection, Learning Beyond Labels).

Unchanged on purpose
* Default hyper-parameters, the 2 % character-error target and z = 2 of FRAG-RC, the style descriptors, the LLM evaluation proxy.

## v10 and earlier
The history of v2 -> v10 (what each IAM run showed and what the next version did about it) is preserved in the original
single-file header; the essentials are summarised in `docs/AUDIT_v10_to_v11.md` and in the module docstrings.
