# Inference outputs

Every evaluation writes to `<work_dir>/<run>/inference/<split>/`, one numbered folder per question.  Every figure has the CSV or
JSON it was drawn from next to it.  Readings of right-to-left corpora are written in logical order.

| folder | contents | question |
|---|---|---|
| `00_summary/` | `metrics.json` (every number), `report.html` (flat table), `config_used.json`, `v11_extras.json`, `training_curves.png` | what are the results? |
| `01_recognition/` | `predictions.csv` (per line: corpus, truth, reading, CER, confidence, view disagreement, edits, post-hoc reading, measured and predicted style), `worst_lines.txt`, `alignments.jsonl` (per character: frames, pixel span, confidence), `cer_distribution.png`, `top_confusions.png` | what did it read, and where exactly? |
| `02_uncertainty/` | `calibration.png` (reliability diagram with temperature, risk-coverage curves), `word_marks.json` ([CAL-MARK]), `uncertainty_decomposition.png` + `uncertainty_signals.csv` | can the model tell when it is wrong, and which signal says so best? |
| `03_error_structure/` | `error_structure.png` (character classes, form blocks, style quintiles), `groups.csv` (CER + CI per corpus / hardness cell / style cluster) | where do the errors come from? |
| `04_posthoc/` | `posthoc.json`: view ensemble, style AdaBN, writer self-distillation, space bonus -- each with its `val_gate` gain interval, whether it was adopted, and its test effect | which label-free options help, with evidence? |
| `05_style/` | regression and confusion of the style task, embedding spaces, disentanglement probe, FiLM gamma, `style_neighbours.png` + `.csv` | does the model know the writer, and is style separated from content? |
| `06_codebooks/` | usage, code -> character (purity vs chance), PCA | what do the discrete codes stand for? |
| `07_robustness/` | `robustness_profile.png` (10 perturbations, CI), `longline_stress.png` + `.json` | how fragile is the reading? |
| `08_self_distillation/` | `pseudo_label_reliability.png` (legacy rule), `frag_trust_audit.json` (the FRAG-RC rule applied to the evaluation split) | is what the model trusts actually right? |
| `09_language_model/` | `llm_rescoring_tuning.png`, `llm_rescoring.json` | does the LLM help, and was it adopted on `val_gate`? |
| `10_explanations/` | `samples/<tag>/` dashboards (input, posterior, confidences, saliency, integrated gradients, occlusion, per-character attribution, attention rollout, VQ codes, **style attention**), `layer_probes.png` + `.csv`, `counterfactuals/` + `.json` | why did it read this, and where in the network does the reading form? |
| `11_translation/` | `translations.json` (plain / uncertainty-aware / rescored arms, prompt-matched oracles, selective translation), `translation_ablation.png`, `panels/` (style-matched renderings) | what reaches the user? |
| `12_report/` | `index.html` (the interactive Reading Atlas) + `01_ledger ... 09_integrity` folders with the numbers behind every panel | the whole story in one page |

## New in v11 (`InferConfig.interpret`)

* **Layer probes** -- linear probes on the encoder state after every layer: accuracy of the emitted character (vs the majority
  class) and R2 of the measured slant from the pooled state.  Shows where content forms and how much writer style the content
  stream still carries.
* **Uncertainty decomposition** -- for each line: 1 - sequence confidence, posterior entropy at the emitting frames, MC-dropout
  disagreement, view disagreement, FRAG-RC mean character risk.  AUROC for "line has an error", rank correlations, risk-coverage
  curves.
* **Character counterfactuals** -- the ink of one emitted character is erased (its CTC frames x 4 px) and the line is read again; the
  dependency matrix shows which characters change (own ink vs context).
* **Style neighbours** -- nearest lines in the style embedding with their measured descriptors; the neighbour / random descriptor-gap
  ratio says whether the space groups similar hands.
* **Alignments** -- `alignments.jsonl`, the CTC alignment of every line for downstream work (word boxes, keyword spotting).
* **Long-line stress test** -- pairs of test lines concatenated and read as one; the gap to the same lines read alone measures
  length generalisation.
* **Group tables** -- CER with form-clustered intervals per corpus, hardness cell and style cluster.

Every analysis is guarded: a failure in one is logged and never costs the rest of the evaluation.
