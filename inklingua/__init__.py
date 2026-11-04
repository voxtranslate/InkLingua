"""
InkLingua v11 -- style-disentangled, uncertainty-aware handwriting-to-translation with fragment-level
risk-controlled self-distillation, writer self-distillation, an LLM in the loop, multi-dataset / multi-script
training and a split validation protocol.

The package is the v10 single-file script restructured into readable modules (one question per module) with the
defects of the audit (`docs/AUDIT_v10_to_v11.md`) fixed and the additions listed in `docs/CHANGELOG.md`.

    inklingua.config            every option (dataclasses); nothing is parsed from the command line
    inklingua.datasets_registry the verified free line-level corpora that can complement IAM (12 entries)
    inklingua.text              grapheme-cluster codec, right-to-left handling, detokenisation, CER / WER / chrF, bootstrap
    inklingua.decoding          CTC greedy / prefix-beam decoding
    inklingua.style             the five measured style descriptors (targets of the connected task)
    inklingua.data              preprocessing, sources (Hub / local parquet / synthetic), datasets, loaders, MixLine
    inklingua.model             InkLinguaNet, losses, EMA teacher, style-equivariant GPU augmentation
    inklingua.llm               the pretrained LLM: translation + N-best rescoring
    inklingua.riskcontrol       isotonic risk curves, thresholds, cluster-robust bounds, transfer calibration
    inklingua.pseudo            legacy line-level pseudo-labels (kept as the `legacy_pl` baseline)
    inklingua.frag              FRAG-RC: fragment-level, style-stratified, risk-controlled self-distillation
    inklingua.trainer           the training loop (+ protocol files, curves)
    inklingua.render            closed-loop style-matched rendering of the translation
    inklingua.explain           per-sample explanation panels, calibration and selective-prediction helpers
    inklingua.posthoc           validation-gated post-hoc options (view ensemble, AdaBN, WSD, space bonus)
    inklingua.interpret         the interpretability suite (probes, uncertainty decomposition, counterfactuals, ...)
    inklingua.inferencer        evaluation: metrics, structured output folders, translation, report
    inklingua.adapt             zero-shot adaptation to a new written language + fine-tuning on a new dataset
    inklingua.report            the self-contained run report (index.html + the numbers behind every panel)
    inklingua.suite             the experiment suite (baselines, ablations, seed-paired comparison)
    inklingua.selftest          fast invariant tests + the synthetic end-to-end smoke
    inklingua.main              entry point: `main(cfg)` dispatches on cfg.run.mode
"""
from .config import Config  # noqa: F401
from .main import main  # noqa: F401

__version__ = "11.0.0"
__all__ = ["Config", "main", "__version__"]
