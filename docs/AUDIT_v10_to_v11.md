# Audit of the InkLingua v10 script and what v11 changes

This file records the line-by-line review of the single-file v10 script (`inklingua.py`, ~6,000 lines) that
preceded the restructuring into the `inklingua/` package.  Every item names the defect, where it was, and the fix.
Items marked **[verified online]** were checked against the Hugging Face Hub API or the cited paper on 2026-11-04.

## A. Correctness defects (fixed)

| # | Where (v10) | Defect | Fix in v11 |
|---|---|---|---|
| A1 | `run_self_tests` | The self-tests ended with a *full synthetic training run* (`main(c11)`, 12 epochs, 480 lines). Every real run therefore began with minutes of synthetic training, and `main()` inside the tests called `setup_logging()` again, so the real run's log handlers pointed into a temporary directory that the test then deleted (`shutil.rmtree`). Log lines of the real run were lost. | The end-to-end smoke is a separate function (`inklingua.selftest.run_smoke`, `mode="smoke"`, `tests/test_smoke.py`). `run_self_tests` keeps only the fast invariants. `setup_logging` is idempotent and restores the caller's handlers. |
| A2 | `run_suite` | `main(c)` was called per variant x seed with `self_test=True`, i.e. the self-tests ran 9 times. | The suite runs the self-tests once and switches them off for the member runs. |
| A3 | `Trainer.fit` | `tr["total"]` raises `KeyError` when every step of an epoch was skipped for a non-finite loss (the meter is empty). | Every running mean is read with a NaN default; a fully skipped epoch is logged as such. |
| A4 | `frag_trust_batch`, `_wsd_reads` | `theta[int(strat[b])]` indexed the per-stratum thresholds without a bound check: a checkpoint with S strata applied to data with more strata (another dataset, another registry) raises `IndexError`. | The stratum index is clipped to the rule's own number of strata; the strata cut points are stored in `best.pt` and reused by inference instead of being recomputed. |
| A5 | `CharCodec` | Alphabet units were Unicode **code points**. Combining marks (Devanagari matras, Arabic diacritics, `e + dot below + acute`) became separate CTC symbols, which (i) breaks CTC feasibility counts, (ii) makes the v8 adapter's grapheme-cluster `UnitCodec` inconsistent with the trained model. | One grapheme-cluster codec everywhere (`text.CharCodec`): units are grapheme clusters, `num_classes = len(units) + 1`, checkpoints store `units`; a joined `chars` string is kept for compatibility (re-clustering it gives the same units). |
| A6 | `_char_class` | `str.islower()/isupper()` are False for uncased scripts, so every Arabic or Chinese letter was counted as **punctuation** in the error breakdown. | Unicode categories (`Ll`, `Lu`, `Lo` -> "letter (uncased)", `Nd`, `P*`/`S*`, space). |
| A7 | `detokenize` | IAM/English-specific re-attachment of punctuation was applied to every language (and would mangle Chinese or Arabic strings for the LLM). | Applied only when the dataset spec says the transcriptions are IAM-tokenised (`DatasetSpec.detokenize`, True for IAM only). |
| A8 | translation arms | `_translate_block` always used the *uncertainty* prompt for the primary `translation` column although v9 made the plain prompt the default (`use_uncertainty_prompt=False`). | The primary column follows the configured prompt; the two arms are named explicitly (`plain_translation`, `uncertain_translation`). |
| A9 | `Trainer.evaluate` | Collected a `style_pred` array that no caller read (dead computation, one allocation per validation pass). | Removed. |
| A10 | `StyleEncoder` | `style_att` (which frames define the style) was computed, returned in `extras`, and never shown. | New dashboard panel `style_attention`. |
| A11 | `plot_training_curves` | No panel for the v10 main component (FRAG-RC trust, per-stratum coverage, realised error of trusted characters), so a run could not be monitored. | Three new panels (`frag_trust`, `frag_cov`, `frag_err`). |
| A12 | `run_adapt` log | Said "InkLingua v9". | Version string comes from `inklingua.__version__`. |
| A13 | `summarise_suite` | Read only the greedy CER from `predictions.csv`; ignored WER, the validated post-hoc CER and the final system CER; no component-contribution table; the default variant list (3) was too thin for a review. | Reads `metrics.json` too; reports CER / WER / posthoc / system CER per seed; a contribution table "full minus each ablation"; presets `quick`, `paper`, `all`; `mode="suite_summary"` re-aggregates finished runs without retraining. |
| A14 | data module | A single dataset only; IAM is small (6,482 train lines) and Latin-script data dominates. | `DataConfig.sources` accepts any number of `DatasetSpec`s (a registry of 12 verified free line-level corpora is shipped, see `docs/DATASETS.md`); lines carry a `domain` id; FRAG-RC Mondrian cells become domain x hardness; metrics are also reported per domain. |
| A15 | data module | No right-to-left support: KHATT (Arabic) stores text in logical order while CTC reads the image left to right. | `DatasetSpec.rtl`: text is converted to visual order (grapheme-cluster-wise reversal) for training and metrics (edit distance is invariant under reversing both strings) and converted back to logical order for every human-facing output and the LLM. |
| A16 | progress bars (found by the real-IAM run of the plan) | `tqdm.auto` resolves to tqdm's asyncio class, whose constructor already calls `iter()` on its argument; iterating it then called `iter()` again. Every epoch therefore started two DataLoader iterators: twice the configured workers, one idle pair holding prefetched batches. | `tqdm.autonotebook` (plain or notebook class, one iterator); a self-test counts the iterators. |
| A17 | line cache (found by the real-IAM run of the plan: a worker was OOM-killed) | The decoded lines were one pickle loaded into each run's RAM (about 3 GB for CASIA, 6 GB for HOME-Alcar or NorHand v3, 9 GB for the joint stage) and copied page by page into every forked loader worker; two parallel runs on Kaggle's 30 GB host could not hold that. | The pixels are streamed to a flat `.ink` file at decoding time and memory-mapped (`InkStore`): the OS page cache is shared by both runs and every worker, subsets and concatenations copy no pixels, a worker receives file paths only. Cache format version 2. |

## B. Protocol / evaluation issues (fixed)

| # | Issue | Fix |
|---|---|---|
| B1 | Inference wrote ~40 files flat into `inference/` and `figures/`, hard to navigate. | Numbered folders, one per question (`01_recognition`, `02_uncertainty`, ..., `12_report`), each with the CSV/JSON behind every figure (`docs/INFERENCE_OUTPUTS.md`). |
| B2 | No analysis of *where in the network* the reading forms, *which uncertainty signal* predicts errors, or *which ink* a character depends on. | New `InterpretMixin`: layer-wise linear probes, uncertainty decomposition (entropy / MC-dropout / view disagreement / FRAG-RC risk), character counterfactuals, style-space neighbours, CTC alignment export, long-line stress test, per-cluster / per-stratum / per-domain tables. |
| B3 | The `labeled_only` baseline could not be told apart from `full` on WER or on the system CER. | See A13. |
| B4 | `_REPORT_REFS` quoted HTR-VT at 4.7 % CER. **[verified online]** HTR-VT (arXiv 2409.08573) reports 4.7 % CER / 14.9 % WER on IAM lines, all training labels, no external data. Kept, with the caveat. | - |

## C. Facts verified online (2026-11-04)

* `Teklia/IAM-line`: commit `fbdad97500ce54635c0d1ba306bf535cb40656cf` (2024-03-14), columns `image`/`text`, splits 6,482 / 976 / 2,915, MIT licence on the Hub (the underlying IAM data remain non-commercial research data). **[verified online]**
* Eleven further line-level corpora with the *same* schema exist under `Teklia/*` (RIMES-2011, Belfort, Esposalles, Himanis, HOME-Alcar, NewsEye-Austrian, NorHand v1/v3, CASIA-HWDB2, POPP, KHATT v1.0); split sizes and commit hashes are recorded in `inklingua/datasets_registry.py`. **[verified online]**
* Literature that did not exist when v10 was written and is now cited in the header: DocTTT (WACV 2025, test-time training with meta-auxiliary learning), MetaWriter (CVPR 2025, meta-learned prompt tuning per writer), few-shot writer adaptation via multimodal in-context learning (arXiv 2603.29450), Geometric Risk Control for VLM OCR (arXiv 2603.19790), Selective Conformal Risk Control (arXiv 2512.12844), optimal-transport pseudo-label selection for low-resource HTR (arXiv 2509.16977), Learning Beyond Labels (WACV 2026). None of them selects *characters* of a CTC reading with a risk-controlled, style-stratified rule, so the FRAG-RC claim stands as "to our knowledge". **[verified online]**

## D. Not changed (and why)

* Default hyper-parameters, the 2 % character-error target and z = 2 of FRAG-RC: the v9 run gives no evidence for other values; the suite decides.
* The style descriptors (slant, stroke, density, gap, body) are script-agnostic measurements and stay the same for every dataset.
* The LLM translation evaluation still has no human reference; the oracle-translation proxy is documented as such.
