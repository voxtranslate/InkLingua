# Datasets

IAM-line alone is small (6,482 training lines), English and Latin-script.  v11 can train on any number of corpora at once
(`Config.multi(("iam", "rimes", "khatt"))`), and the registry (`inklingua/datasets_registry.py`) holds the free line-level
corpora that share the Hugging Face layout of `Teklia/IAM-line` (an `image` column, a `text` column, parquet files).  Every row
below was checked against the Hub API on **2026-11-04** (commit hash, columns, split sizes, licence tag).

| key | Hub dataset | language / script | train / val / test lines | Hub licence tag | note |
|---|---|---|---|---|---|
| `iam` | Teklia/IAM-line | en / Latin | 6,482 / 976 / 2,915 | MIT (IAM itself: non-commercial research) | modern English; IAM-tokenised transcriptions (`detokenize=True`) |
| `rimes` | Teklia/RIMES-2011-line | fr / Latin | 10,188 / 1,138 / 778 | MIT (RIMES: research) | modern French letters |
| `belfort` | Teklia/Belfort-line | fr / Latin | 25,800 / 3,102 / 3,819 | MIT | historical municipal registers |
| `esposalles` | Teklia/Esposalles-line | ca / Latin | 2,328 / 742 / 757 | MIT | 17th-century Catalan marriage licences |
| `himanis` | Teklia/Himanis-line | la, fr / Latin | 18,504 / 2,367 / 2,241 | MIT | medieval chancery registers |
| `home_alcar` | Teklia/HOME-Alcar-line | la / Latin | 59,969 / 7,905 / 6,932 | MIT | medieval cartularies |
| `newseye_de` | Teklia/NewsEye-Austrian-line | de / Latin | 51,588 / 4,379 / - | MIT | **no test split**: contributes train / validation only |
| `norhand1` | Teklia/NorHand-v1-line | nb / Latin | 19,653 / 2,286 / 1,793 | MIT | Norwegian letters and diaries |
| `norhand3` | Teklia/NorHand-v3-line | nb / Latin | 222,381 / 22,679 / 1,562 | MIT | large: cap it with `max_lines` |
| `popp` | Teklia/POPP-line | fr / Latin | 3,835 / 480 / 479 | MIT | Paris census tables |
| `casia` | Teklia/CASIA-HWDB2-line | zh / Han | 33,401 / 8,318 / 10,441 | MIT (CASIA: academic research) | thousands of output units |
| `khatt` | Teklia/KHATT_v1.0 | ar / Arabic, **right-to-left** | 4,672 / 963 / 1,038 | none on the Hub (KHATT: free research use) | `rtl=True`: text stored in logical order |

The Hub tags are those of the Teklia conversions; the **underlying corpora have their own terms** (IAM: non-commercial research;
CASIA: academic research; KHATT and RIMES: research).  Check them before publishing anything trained on them.

## How several corpora are combined

* Lines of every corpus are concatenated per split; each line carries a `domain` id, and every evaluation table is also given per
  corpus (`metrics.json -> per_corpus`, `03_error_structure/groups.csv`).
* The alphabet is the union of the grapheme clusters of the labeled lines of all corpora (built from labeled lines only).
* FRAG-RC's Mondrian cells become **corpus x hardness tertile** (`FragConfig.strata_per_domain`): each corpus gets its own
  trust thresholds at the same character-error target, so a small corpus is never silently excluded.
* `DataConfig.domain_balance` (0..1) repeats labeled lines of the smaller corpora so that they are visited more often.
* Right-to-left corpora: the recogniser reads the image left to right, so the transcription is converted to visual order
  (cluster-wise reversal) for training and metrics -- edit distance is invariant under reversing both strings -- and converted
  back to logical order in every CSV, panel, prompt and report.
* MixLine (concatenating two labeled lines) only pairs lines of the same corpus.

## Adding a corpus that is not in the registry

```python
from inklingua import Config
from inklingua.config import DatasetSpec
cfg = Config()
cfg.data.custom_sources = (DatasetSpec(key="my_corpus", hf_dataset="org/name", language="et", script="Latin",
                                       image_col="image", text_col="transcription", expected_sizes=None),)
cfg.data.sources = ("iam", "my_corpus")
```

A local parquet folder works through `DatasetSpec(local_dir=...)` (files named `train*.parquet`, `validation*.parquet`,
`test*.parquet`).  Corpora worth knowing that need a custom spec because their columns differ: Riksarkivet (Swedish),
aarhus-city-archives/historical-danish-handwriting (cc-by-4.0), Rahvusarhiiv/et_handwriting_recognition (Estonian, cc0),
fhswf/german_handwriting (afl-3.0), SSamDav/portuguese-handwriting-16th-19th (cc-by-4.0).

## Adapting a trained model to a new corpus

* **With transcriptions** -- `cfg.run.mode = "finetune"`, `cfg.train.init_from = <checkpoint>`, `cfg.data.sources = (...)`.
  The alphabet of the checkpoint is kept and extended: new units are appended as tied copies of the visually closest known
  unit, so the model's behaviour on the old alphabet is unchanged at step 0 ([ALPHA-SPLIT]).  Everything else (FRAG-RC,
  protocol, suite-compatible outputs) applies.
* **Without transcriptions** -- `cfg.run.mode = "adapt"` on a folder of line images (optionally a few labeled "anchor" lines):
  [LEX-TIE] language-prior resolution of the tied variants, [XL-RC] transfer-calibrated risk control of the pseudo-labels,
  translation into any `cfg.adapt.target_languages`.  `cfg.adapt.rtl = True` for right-to-left orthographies.
* **At inference time, per writer** -- writer self-distillation (`InferConfig.wsd`): label-free, transductive, adopted only
  on a significant `val_gate` gain.
