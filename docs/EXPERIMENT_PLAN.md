# The experiment plan of the article

`notebooks/InkLingua_Kaggle_T4x2.ipynb` is one readable code cell that contains the complete code (generated from the package by
`tools/build_kaggle_notebook.py`) and one entry point.  It runs `run_plan` on real data
only (no synthetic data, no dummy model, no smoke test), and it continues where it stopped in the next Kaggle session.  The plan, the
rules and the outputs are below; every number is produced by the code in `inklingua/experiments.py`, `inklingua/schedule.py` and
`inklingua/paper.py`.

## What runs, in which order

The corpora are processed one after the other: IAM first (the reference benchmark), then the others from the smallest to the largest,
so that complete results arrive early:

`iam, esposalles, popp, khatt, rimes, himanis, norhand1, belfort, casia, home_alcar, norhand3`, then the **joint** stage, with the
**adaptation** stage right after IAM.  NewsEye-Austrian has no test split, so it only enters the joint stage.

Inside a corpus, seed 1 is completed before seed 2 starts (all tables have a first complete column early).  For every seed:

| tier | runs | what they establish |
|---|---|---|
| core (5) | `labeled_only`, `ctc_only`, `legacy_pl`, `full`, `full_supervised` | the main table: does the unlabeled data help, compared with the previous method and with the 100 %-label upper bound? |
| ablation (14) | `no_frag`, `no_strata`, `no_sharpen`, `no_mixline`, `no_style_task`, `no_ses`, `no_vq`, `no_film`, `no_confusion`, `no_frame_cons`, `no_mask`, `no_aux_ctc`, `no_v9_reg`, `no_hscale` | the contribution of every trained component of the method |
| label budget (4) | `labeled_only` and `full` at 5 % and 10 % labels | the label-budget curves (30 % and 100 % come from the core tier) |

The joint stage trains `labeled_only` and `full` on all eleven corpora at once (per-corpus caps of 12,000 training and 1,000
validation lines, cross-split decontamination, corpus-balanced sampling 0.5).  The adaptation stage adapts the IAM `full` model, without
retraining from scratch, to French (RIMES) and Catalan (Esposalles) line images, and measures the CER against 0, 25, 100 and 200 labeled
target lines.

Default: 3 seeds (1337, 1338, 1339), 30 % labels, **767 runs**.

### What is deliberately not retrained

* `no_wsd`, `no_posthoc`, no LLM rescoring: these change only inference.  Every `full` run measures the reading before and after each
  validated option (view ensemble, style AdaBN, writer self-distillation, word-gap bonus, LLM N-best rescoring), so table T4 reads
  them from those runs at no extra cost.
* `no_consistency`, `no_llm`, `no_shadow`, `no_risk_smoothing`: they ablate the legacy baseline, not the proposed method.
* `no_pseudo` equals `labeled_only`, and `no_calibration` equals "FRAG-RC without a rule"; both are redundant.

## The training budget rule

The rule is anchored on IAM: 100 epochs, i.e. 40,600 optimisation steps at batch 16, where the validation CER of the v9 and v10 runs
plateaued between epoch 30 and 77.

```
S(d)      = 40,600 x sqrt(N_train(d) / 6,482) x k_script(d) x k_era(d)
epochs(d) = clip(ceil(S(d) / steps_per_epoch(d)), 20, 150)
k_script  : Latin, Cyrillic, Greek 1.0 | Hebrew 1.2 | Arabic, Devanagari 1.3 | Han 1.8
k_era     : modern 1.0 | historical 1.15
```

Patience, minimum epochs, FRAG-RC warm-up and refresh, legacy warm-up and refresh, auxiliary-loss and quantisation ramps, the
checkpoint interval and the learning-rate warm-up scale with the budget, using the fractions the IAM defaults already had.  IAM
therefore reproduces the published defaults exactly (a self-test checks it).  Early stopping on `val_sel` makes the budget an upper
bound.  Every variant of a corpus gets the same budget, so all comparisons are at equal optimisation cost.

| corpus | train lines | complexity | epoch budget | steps / epoch |
|---|---|---|---|---|
| iam | 6,482 | 1.00 | 100 | 406 |
| esposalles | 2,328 | 1.15 | 150 | 146 |
| popp | 3,835 | 1.15 | 150 | 240 |
| khatt | 4,672 | 1.30 | 150 | 292 |
| rimes | 10,188 | 1.00 | 80 | 637 |
| himanis | 18,504 | 1.15 | 69 | 1,157 |
| norhand1 | 19,653 | 1.15 | 67 | 1,229 |
| belfort | 25,800 | 1.15 | 58 | 1,613 |
| casia | 33,401 | 1.80 | 80 | 2,088 |
| home_alcar | 59,969 | 1.15 | 38 | 3,749 |
| norhand3 | 60,000 (capped) | 1.15 | 38 | 3,750 |
| joint | 99,505 (capped) | 1.21 | 31 | 6,220 |

The run computes its budget from its actual data (after label filtering); the table above is the estimate from the published split sizes.

## Data rules

* Every corpus is read from its pinned Hub commit, parquet files only (no second Arrow copy on disk); the resolved commit is recorded.
* Caps keep whole blocks of 20 consecutive lines, chosen with a fixed subset seed, so every variant and seed sees the same pool.
  Test sets are never capped.  Caps: NorHand v3 train 60,000 and validation 3,000; HOME-Alcar, CASIA validation 3,000; joint stage
  12,000 / 1,000 per corpus.
* The joint stage removes training lines whose image also occurs in a validation or test set, and validation lines that also occur
  in a test set (exact image digests, counted in the reproducibility report).  NorHand v3 is not in the joint stage because it
  probably contains NorHand v1.
* `max_text_len` is raised to 200 for the historical corpora; lines that cannot fit a CTC alignment are still dropped and counted.

## How a session runs on Kaggle (GPU T4 x2)

* Every GPU is first checked in its own process: three fp16 training steps of the real model (cuDNN convolutions, recurrent and
  attention layers, CTC, GradScaler) on random tensors.  A GPU that fails is reported with its log and not used; if none passes, the
  session stops before any run.
* Every corpus is downloaded and decoded once, in a CPU-only process, before its first run (decoding uses spawned worker processes,
  never `fork`).  Runs only read the caches.
* A failed run prints the end of its log in the notebook output.  Run processes use Python's fault handler, so a native crash leaves
  a Python stack in `<run>/stdout.log`.  After 4 failed attempts in a row with no run finished in the session, the session stops
  (a systematic error) instead of spending the plan.  A run stopped by the session end is not a failure.
* Two runs train in parallel, one per T4.  Each run is its own process pinned to its GPU, so memory is released between runs.  The
  LLM (Qwen2.5-1.5B) shares the GPU of the run that needs it.
* No run starts unless its evaluation reserve plus 30 minutes of training remain: 100 minutes for the full evaluation, 40 for the
  standard one, 15 for the core one, scaled by the test-set size.  A training run stops at the step where its deadline falls, with a step
  checkpoint (also written every 10 minutes and on SIGTERM).  The session ends about 40 minutes before Kaggle's 12-hour limit.
* After a run, its checkpoints are pruned.  `best.pt` is kept for the core tier only.  When no pending run needs a corpus any more,
  its downloads and line caches are deleted.
* Downloads are resolved offline once the pinned files are on disk; each corpus is decoded once (in parallel) into a line cache shared by
  all its runs.  At the end of a session the caches of unfinished corpora are saved in the output (`_cache/lines`, within the 20 GB
  limit) and restored by the next session.
* To continue: add the previous version's output as input, set `RESUME_FROM`, then Save & Run All.  Finished runs are skipped and
  interrupted runs resume exactly at the step where they stopped (a run stopped by the session end is not counted as a failed attempt).

## Evaluation depth

| profile | used for | contents |
|---|---|---|
| full | `full`, first seed, every corpus and the joint model | everything: post-hoc options incl. WSD, LLM rescoring, translation, interpretability, robustness, the run report |
| standard | the other core runs | metrics, every post-hoc option incl. WSD and LLM rescoring (the system CER) |
| core | ablations, label-budget runs | metrics and the cheap post-hoc options (no WSD, no LLM) |

## Outputs (`/kaggle/working/inklingua_runs/paper/`, rebuilt after every finished run)

| file | content |
|---|---|
| `RESULTS.md` | every table, with the statistics used |
| `tables/T1_main_cer` / `T1b_main_wer` | CER / WER per corpus and core variant, mean ± sd over seeds, plus the system CER |
| `tables/T2_main_claim` | InkLingua minus labeled_only and minus line-level PL, pooled seed-paired form-clustered bootstrap, Holm over corpora, sign test over (corpus, seed) |
| `tables/T3_ablations` + `figures/ablation_heatmap.png` | ablation minus full, Holm over the ablations of each corpus |
| `tables/T4_posthoc` | each validated option's test CER and adoption count, beam, LLM rescoring, oracle N-best |
| `tables/T5_label_budget` + `figures/label_budget.png` | CER versus labeled fraction |
| `tables/T6_joint` | joint model versus per-corpus models, per corpus, paired |
| `tables/T7_adaptation` | zero-shot adaptation CER versus labeled target lines |
| `tables/T8_translation`, `T9_uncertainty`, `T10_compute` | chrF of the translation arms; AUROC of the uncertainty signals; epochs, best epochs, GPU-hours |
| `results_long.csv` | one row per finished run, every number the tables use |
| `REPRODUCIBILITY.md` | datasets and commits, caps, decontamination counts, budget rule and budgets, environment, code archive sha256, protocol |
| `STATUS.md` | done / running / interrupted / pending / failed runs, with the last error line of each failure |

Every `.md` table also exists as `.tex` (booktabs) and `.csv`.

## Cost (be aware before starting)

With the prior of 0.15 s per step on a T4, and early stopping at about 75 % of the budget, the full plan is about **2,350 GPU-hours**.
With two GPUs per session that is roughly 1,150 session-hours, i.e. about 100 Kaggle sessions.  The free weekly GPU quota does not
cover that in less than many months.  The estimate printed at the start of every session uses measured rates as soon as runs finish.
`PLAN_OVERRIDES` in the notebook narrows the plan without touching the code, for example:

```python
PLAN_OVERRIDES = {"seeds": (1337,)}                                        # one seed: about one third of the cost
PLAN_OVERRIDES = {"ablation_corpora": ("iam", "khatt", "casia")}           # ablations on three scripts only
PLAN_OVERRIDES = {"order": "tier"}                                         # core runs on every corpus first, then ablations
```
