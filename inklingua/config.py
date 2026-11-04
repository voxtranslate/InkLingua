"""
Configuration.  EVERY option lives here; `main()` never parses command-line arguments.  Edit the dataclass defaults, or
build a Config in a notebook cell and pass it to `main(cfg)`.

The most used entry points:
    Config()                         the default IAM-line experiment (30 % labels, FRAG-RC + WSD)
    Config.labeled_only()            THE baseline: same labeled lines, no unlabeled data
    Config.ablation("no_frag")       one component removed (see Config.ablation for the list)
    Config.multi(("iam", "rimes"))   train on several corpora of the registry at once (docs/DATASETS.md)
    Config.smoke()                   a tiny synthetic configuration for plumbing tests
    cfg.run.mode = "suite"           train / evaluate every variant for every seed (the experiment a paper needs)
"""
from __future__ import annotations

import copy
import dataclasses
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

IAM_LINE_REVISION = "fbdad97500ce54635c0d1ba306bf535cb40656cf"      # Teklia/IAM-line, last modified 2024-03-14 (verified 2026-11-04)
IAM_LINE_SIZES = (6482, 976, 2915)                                    # train / validation / test


def _default_work_dir() -> str:
    return "/kaggle/working/inklingua_runs" if os.path.isdir("/kaggle/working") else "./inklingua_runs"


# ==============================================================================
# Datasets
# ==============================================================================
@dataclass
class DatasetSpec:
    """One line-level corpus.  The registry (inklingua/datasets_registry.py) holds verified specs; a custom one can be built
    for any Hugging Face dataset (or local parquet folder) that has an image column and a text column."""
    key: str                                   # short name used in cfg.data.sources and in every per-domain table
    hf_dataset: str = ""                       # Hub id, e.g. "Teklia/IAM-line" ("" with local_dir = local parquet only)
    hf_revision: Optional[str] = None          # pinned commit (None = latest)
    local_dir: Optional[str] = None            # folder with train/validation/test .parquet (takes precedence over the Hub)
    language: str = "en"                       # ISO code of the WRITTEN language (prompts, LM scoring, per-domain tables)
    script: str = "Latin"                      # "Latin" | "Arabic" | "Han" | "Cyrillic" | "Devanagari" | ...
    rtl: bool = False                          # right-to-left script: transcriptions are stored in LOGICAL order and are converted to
                                               # visual order (left -> right as the image is read) for training; outputs are converted back
    detokenize: bool = False                   # transcriptions are IAM-tokenised ("house , and"): re-attach punctuation for the LLM
    image_col: Optional[str] = None            # None = auto-detect (image / img / line_image)
    text_col: Optional[str] = None             # None = auto-detect (text / transcription / label / sentence / gt)
    train_split: str = "train"
    val_split: str = "validation"
    test_split: Optional[str] = "test"         # None = this corpus contributes no test lines (e.g. NewsEye has no test split)
    expected_sizes: Optional[Tuple[int, ...]] = None   # (train, val[, test]) as published; checked when strict_checks is on
    max_lines: Optional[Dict[str, int]] = None  # optional cap per split, e.g. {"train": 20000}: a seeded random subset is used
    licence: str = ""                          # for the provenance file
    note: str = ""


# ==============================================================================
# Run / data / model / losses
# ==============================================================================
@dataclass
class RunConfig:
    name: str = "inklingua_iam"
    work_dir: str = field(default_factory=_default_work_dir)   # checkpoints, curves, inference outputs
    seed: int = 1337
    device: str = "auto"                       # "auto" | "cpu" | "cuda" | "cuda:1"
    do_train: bool = True
    do_infer: bool = True
    self_test: bool = True                     # run the fast invariant tests (inklingua.selftest.run_self_tests) before anything else
    smoke_test: bool = False                   # True ONLY for Config.smoke(): permits synthetic data, stamps outputs
    log_level: str = "INFO"
    # "single"        train + evaluate this configuration
    # "suite"         train / evaluate every variant for every seed and write the seed-paired comparison (suite_summary.*)
    # "suite_summary" only re-aggregate the finished runs of a suite (no training)
    # "finetune"      start from cfg.train.init_from (any InkLingua checkpoint), grow the alphabet to the new data, train, evaluate
    # "adapt" / "adapt_curve" / "source_calib"  zero-shot adaptation to a new written language from a folder of line images (AdaptConfig)
    # "report"        rebuild inference/<split>/12_report/index.html of an EXISTING run from its files (no model, no download)
    # "smoke"         the synthetic end-to-end smoke (train -> resume -> inference) in a temporary directory
    mode: str = "single"
    suite_preset: str = "paper"                # "quick" | "paper" | "all" | "custom" (then suite_variants is used)
    suite_variants: Tuple[str, ...] = ("labeled_only", "legacy_pl", "full")
    suite_seeds: Tuple[int, ...] = (1337, 1338, 1339)


@dataclass
class DataConfig:
    source: str = "hf"                         # "hf" (download) | "local" (parquet folder, see local_dir) | "synthetic" (smoke test only)
    sources: Tuple[str, ...] = ("iam",)        # registry keys (inklingua.datasets_registry.DATASETS); several = multi-corpus training
    custom_sources: Tuple[DatasetSpec, ...] = ()   # specs that are not in the registry
    hf_cache_dir: Optional[str] = None         # default: <work_dir>/data
    local_dir: Optional[str] = None            # source="local" with ONE source: the parquet folder of that source
    strict_checks: bool = True                 # True: wrong split sizes / image overlap between splits raise
    img_height: int = 64                       # must be a multiple of 16
    max_img_width: int = 2560                  # Teklia lines are up to ~5000 px wide at 128 px height
    min_img_width: int = 32
    pad_x: int = 8                             # blank margin (px) left/right of every line (keeps augmentations from cropping ink)
    max_text_len: int = 90                     # applied to LABELED lines only (IAM-line maximum is 80)
    labeled_fraction: float = 0.30             # fraction of the train lines whose text is used (the annotation budget)
    calib_fraction: float = 0.10               # share of that budget HELD OUT (never trained on) to calibrate the trust rule of FRAG-RC /
                                               # the legacy pseudo-label selection; 0 = none (FRAG-RC then has no rule)
    drop_nonlinguistic: bool = True            # drop LABELED train lines whose transcription has no letter / digit (e.g. "-----")
    limit_train: Optional[int] = None          # quick experiments (applied to the concatenation of the sources)
    limit_val: Optional[int] = None
    limit_test: Optional[int] = None
    cache_images: bool = True                  # keep the preprocessed uint8 lines in RAM (~1 GB for all of IAM-line)
    batch_size: int = 16
    eval_batch_size: int = 32
    num_workers: int = 2
    bucket_mega: int = 50                      # width-bucketing window (in batches)
    train_eval_subset: int = 400               # labeled train lines used for the "train" curves
    style_num_clusters: int = 10               # K style classes
    style_label_column: Optional[str] = None   # e.g. "writer" if your dataset has writer ids
    val_gate_fraction: float = 0.5             # [PROTOCOL] share of the validation FORM BLOCKS reserved for post-hoc decisions (`val_gate`); the
                                               # rest (`val_sel`) selects the checkpoint.  0 or 1 = no split (validation does both jobs)
    val_block: int = 20                        # consecutive validation lines per block (a form / writer proxy: lines are stored form by form)
    domain_balance: float = 0.0                # multi-corpus: 0 = lines sampled in proportion to corpus size; 1 = every corpus equally often
    synthetic_sizes: Tuple[int, int, int] = (600, 120, 120)
    synthetic_words: Tuple[int, int] = (3, 6)  # words per synthetic line (offline smoke tests only)


@dataclass
class ModelConfig:
    channels: Tuple[int, int, int, int] = (32, 64, 128, 256)
    d_model: int = 256
    enc_layers: int = 4
    heads: int = 8
    ffn: int = 1024
    dropout: float = 0.1
    layer_drop: float = 0.1                    # stochastic depth: drop probability of the LAST encoder layer, linear in depth; 0 = off
    conv_kernel: int = 7
    low_dim: int = 128                         # stroke-level code dimension
    codes_low: int = 256                       # stroke-level codebook size
    codes_high: int = 512                      # component-level codebook size
    vq_decay: float = 0.99
    vq_dead_thresh: float = 0.5
    style_dim: int = 128
    style_input: str = "residual"              # "residual" (z - quantised z) | "continuous"
    film: bool = True
    film_detach_style: bool = True
    film_max: float = 0.5                      # |gamma|, |beta| of the style FiLM are bounded by this value (tanh, no bias)
    lstm_prefix: bool = False                  # optional BiLSTM before the Transformer
    style_grad_scale: float = 0.1              # fraction of the style-task gradient that reaches the shared backbone
    mask_prob: float = 0.25                    # masked-code modelling (span masking)
    mask_span: int = 8


@dataclass
class LossConfig:
    aux_ctc: float = 0.3                       # CTC on the quantised content stream (follows the aux curriculum)
    style_reg: float = 1.0
    style_cls: float = 0.2
    style_cls_smoothing: float = 0.1
    leak: float = 0.1                          # [CFL] closed-form leakage penalty on the content stream
    consistency: float = 0.2                   # weak->strong frame-posterior consistency to the EMA teacher
    masked_code: float = 0.2
    decorrelate: float = 0.05
    vq_commit: float = 0.25
    ctc_smooth: float = 0.02                   # [CTC-LS] label smoothing on EMITTING frames only; 0 = off
    frag: float = 0.5                          # [FRAG-RC] weight of the fragment-level self-distillation (x aux_scale); 0 = off


# ==============================================================================
# Pseudo-labels (legacy, line level) and FRAG-RC (character level)
# ==============================================================================
@dataclass
class PseudoConfig:
    """LEGACY (v3-v9) line-level pseudo-labelling.  Off by default; Config.ablation('legacy_pl') turns it on as a baseline."""
    enabled: bool = False
    warmup_epochs: int = 25
    refresh_every: int = 3
    target_cer: float = 0.02                   # intended CER of the selected pseudo-labels
    risk_z: float = 2.0                        # z of the upper bound (calibrated by simulation, not a confidence level)
    min_calib: int = 40
    min_selected_calib: int = 15
    shadow_calibration: bool = True            # [SHADOW] calibration lines are pseudo-labelled and trained on like unlabeled lines
    risk_folds: int = 5
    risk_smoothing: bool = True                # isotonic risk curve + stop at the FIRST bound violation
    cov_guard: bool = True
    max_cov_growth: float = 0.10
    min_cov_keep: float = 0.50
    cov_ref_window: int = 5                    # [GUARD-REF]
    save_source_calibration: bool = True       # checkpoints/source_calibration.npz, the source side of the zero-shot adapter's [XL-RC]
    guard_cer_factor: float = 1.5
    rel_views: Tuple[str, ...] = ("gauss noise", "shear +0.08", "shear -0.08")
    fallback_tau_conf: float = 0.97
    min_stability: int = 1
    min_chars: int = 2
    lops_drop: float = 0.05
    min_weight: float = 0.50
    consistency: bool = True                   # [PCG] perturbation-consistency gate
    cons_views: Tuple[str, ...] = ("gauss noise", "shear +0.08")
    tau_cons: float = 0.10
    llm_refine: bool = True
    llm_target_cer: float = 0.06
    llm_budget: int = 150
    llm_beam: int = 5
    llm_beta: float = 0.30
    llm_rho: float = 1.0
    llm_max_edit_ratio: float = 0.25
    llm_min_tok_lp: float = -4.5
    llm_weight_scale: float = 0.7


@dataclass
class FragConfig:
    """[FRAG-RC] fragment-level, style-stratified, risk-controlled self-distillation (inklingua/frag.py)."""
    enabled: bool = True
    warmup_epochs: int = 20                    # supervised-only epochs before the first calibration
    refresh_every: int = 4                     # epochs between calibrations of the trust rule
    target_err: float = 0.02                   # intended error rate of the TRUSTED characters (wrong, or next to an omission), per cell
    risk_z: float = 2.0                        # z of the line-clustered upper bound (heuristic, not a confidence level)
    min_chars: int = 400                       # fewer calibration characters in a cell -> nothing is trusted there
    folds: int = 5                             # cross-fitting folds (by line)
    calib_passes: int = 2                      # augmentation draws per calibration line
    guard_frames: int = 1                      # frames around an untrusted character that are never trusted
    sharpen: float = 0.6                       # tau: target = softmax(log p / tau) on trusted frames (1 = plain distillation)
    anchor_weight: float = 0.1                 # weight of the plain-teacher target on the untrusted frames of unlabeled lines
    unlabeled_ratio: float = 1.5               # unlabeled / calibration lines per labeled line in an epoch
    stratum_emphasis: float = 0.5              # labeled lines of harder cells are visited (err_s / mean err)^gamma times more often (1..3); 0 = off
    diag_lines: int = 600                      # unlabeled lines scored against their hidden text at each refresh (DIAGNOSTIC ONLY)
    strata: int = 3                            # hardness tertiles of the label-free score -(z thin stroke + z tight word gaps)
    strata_per_domain: bool = True             # multi-corpus: Mondrian cells = corpus x hardness tertile (one threshold per cell)


# ==============================================================================
# Training / LLM / adaptation / inference
# ==============================================================================
@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 6e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 1e-2
    warmup_steps: int = 600
    grad_clip: float = 5.0
    amp: bool = True
    strong_aug: bool = True
    aug_geom_prob: float = 0.5
    aug_morph_prob: float = 0.24
    aug_blur_prob: float = 0.3
    aug_shear_max: float = 0.10
    aug_vscale: Tuple[float, float] = (0.92, 1.08)
    aug_hscale: Tuple[float, float] = (0.85, 1.15)
    aug_hscale_prob: float = 0.5
    aug_elastic_px: float = 2.0
    aug_elastic_prob: float = 0.3
    aug_tilt_px: float = 3.0
    unlabeled_aux_ratio: float = 0.5           # lines WITHOUT a recognition target per line with one (when FRAG-RC is off)
    style_equivariant: bool = True             # [SES]
    ema_decay: float = 0.999
    aux_start_epoch: int = 3
    aux_ramp_epochs: int = 5
    vq_start_epoch: int = 3
    vq_anneal_epochs: int = 4
    patience: int = 25                         # early stopping on val_sel CER; 0 = never
    min_epochs: int = 30
    min_delta: float = 5e-4
    resume: bool = True
    extend_policy: str = "new_cycle"           # resume with a LARGER epochs: "new_cycle" (warm restart) | "error"
    restart_lr_scale: float = 0.3
    restart_warmup_steps: int = 200
    leak_queue: int = 1024
    leak_ridge: float = 0.3
    leak_slack: float = 0.02
    save_every: int = 5
    keep_last: int = 3
    noise_eval_samples: int = 300
    pseudo: PseudoConfig = field(default_factory=PseudoConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    frag: FragConfig = field(default_factory=FragConfig)
    mixline_prob: float = 0.15                 # [MIXLINE] probability that two adjacent labeled lines of a batch are concatenated (0 = off)
    mixline_gap: Tuple[int, int] = (0, 24)
    # --- warm start / fine-tuning on a new corpus (mode "finetune", or any run) -----------------------------------------------
    init_from: str = ""                        # path of an InkLingua checkpoint (best.pt / last.pt / adapted.pt); "" = train from scratch
    init_lr_scale: float = 0.5                 # peak lr relative to cfg.train.lr when starting from init_from
    init_strict_alphabet: bool = False         # False: the alphabet may GROW (new characters appended as tied copies, [ALPHA-SPLIT])


@dataclass
class LLMConfig:
    backend: str = "causal"                    # "causal" | "seq2seq" | "dummy"
    device: str = "auto"                       # "auto" = the SECOND GPU when there are two (Kaggle T4 x2: recogniser on cuda:0, LLM on cuda:1),
                                               # otherwise the recogniser's device; or an explicit "cuda:1" / "cpu"
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    seq2seq_models: Dict[str, str] = field(default_factory=lambda: {
        "fr": "Helsinki-NLP/opus-mt-en-fr", "es": "Helsinki-NLP/opus-mt-en-es",
        "de": "Helsinki-NLP/opus-mt-en-de", "it": "Helsinki-NLP/opus-mt-en-it",
        "pt": "Helsinki-NLP/opus-mt-tc-big-en-pt", "ru": "Helsinki-NLP/opus-mt-en-ru",
        "zh": "Helsinki-NLP/opus-mt-en-zh", "ar": "Helsinki-NLP/opus-mt-en-ar",
        "hi": "Helsinki-NLP/opus-mt-en-hi"})
    target_language: str = "fr"
    source_language: str = ""                  # "" = taken from the (first) dataset spec of cfg.data.sources
    max_new_tokens: int = 96
    nbest: int = 3
    word_conf_thresh: float = 0.70             # fallback only: the Inferencer calibrates it on validation words ([CAL-MARK])
    word_conf_agg: str = "mean"
    use_uncertainty_prompt: bool = False       # the '[?]' prompt LOST chrF against the plain prompt in the v7 IAM run
    score_batch_size: int = 8


@dataclass
class AdaptConfig:
    """[ZERO-SHOT] adaptation of a trained checkpoint to handwriting in a language / orthography the training data did not have
    (mode "adapt"; inklingua/adapt.py).  Nothing here is used by a normal run."""
    checkpoint: str = "best"
    source_calibration: str = ""
    lines_dir: str = ""
    anchors_file: str = ""
    eval_file: str = ""
    out_dir: str = ""
    target_languages: Tuple[str, ...] = ("fr",)
    charset: str = ""
    visual_base: Dict[str, str] = field(default_factory=dict)
    rtl: bool = False                          # the target orthography is written right to left (text files are in logical order)
    rounds: int = 4
    epochs_per_round: int = 3
    lr: float = 3e-4
    batch_size: int = 16
    unfreeze_layers: int = 1
    ema_decay: float = 0.9
    max_lines: Optional[int] = None
    target_cer: float = 0.03
    risk_z: float = 2.0
    min_selected: int = 15
    min_ess: float = 10.0
    zero_anchor_inflation: float = 1.5
    anchor_prior_edits: float = 20.0
    anchor_train_share: float = 0.5
    min_anchors_to_train: int = 40
    lex_beam: int = 4
    lex_vis_weight: float = 1.0
    lex_lm_weight: float = 1.0
    lex_rho: float = 0.0
    lex_tie_margin: float = 1.0
    lex_decided_prob: float = 0.98
    lex_max_options: int = 6
    lex_max_lines: int = 400
    translate_max: int = 200
    lex_final_lines: int = 2000
    anchor_limit: Optional[int] = None
    anchor_counts: Tuple[int, ...] = (0, 25, 100, 200)


@dataclass
class InferConfig:
    checkpoint: str = "best"                   # "best" | "last" | explicit path
    split: str = "test"
    max_eval_samples: Optional[int] = None
    beam: int = 5
    n_vis: int = 8
    n_translate: int = 160
    render_translation_max: int = 24
    ig_steps: int = 12
    occl_window: int = 24
    occl_stride: int = 12
    pl_diag_samples: int = 1500
    selective_val_lines: int = 976
    robustness_lines: int = 300
    translation_ablation: bool = True
    render_translation: bool = True
    html_report: bool = True
    llm_rescoring: bool = True
    rescore_beam: int = 8
    rescore_val_lines: int = 976
    rescore_test_lines: Optional[int] = None
    rescore_betas: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
    rescore_extend_grid: bool = True
    rescore_rhos: Tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)
    rescore_extend_rho: bool = True
    bootstrap: int = 2000
    bootstrap_block: int = 20                  # lines per resampling block (form-clustered bootstrap); 1 = plain line bootstrap
    mark_precision: float = 0.5                # [CAL-MARK]
    posthoc: bool = True
    tta: bool = True
    tta_clusters: int = 8
    tta_shrink: Tuple[float, ...] = (10.0, 30.0, 100.0)
    tta_min_lines: int = 10
    view_ensemble: bool = True
    view_sets: Tuple[Tuple[str, ...], ...] = (("shear +0.08", "shear -0.08"), ("thicker ink", "thinner ink"),
                                              ("shear +0.08", "shear -0.08", "thicker ink", "thinner ink"))
    space_bonus_grid: Tuple[float, ...] = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    error_analysis_lines: int = 3000
    wsd: bool = True                           # [WSD] writer self-distillation (transductive, label-free; adopted only on a val_gate gain)
    wsd_clusters: int = 8
    wsd_steps: Tuple[int, ...] = (20, 60)
    wsd_lr: float = 1e-4
    wsd_batch: int = 16
    wsd_min_lines: int = 24
    wsd_tau: float = 0.6
    wsd_anchor: float = 0.1
    report: bool = True                        # build 12_report/index.html at the end of the evaluation
    # --- the interpretability suite (inklingua/interpret.py); every analysis writes CSV/JSON + PNG into its own folder ---------
    interpret: bool = True
    probe_lines: int = 400                     # lines whose encoder states feed the layer-wise linear probes
    mc_dropout_samples: int = 8                # stochastic forward passes of the uncertainty decomposition (0 = off)
    uncertainty_lines: int = 600
    counterfactual_lines: int = 6              # lines whose characters are erased one by one
    neighbour_queries: int = 6                 # query lines of the style-space neighbour panel
    longline_pairs: int = 300                  # concatenated test pairs of the long-line stress test (0 = off)
    export_alignments: bool = True             # 01_recognition/alignments.jsonl: per-line CTC alignment (char, frames, confidence)


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    adapt: AdaptConfig = field(default_factory=AdaptConfig)

    # ---- derived paths -------------------------------------------------------
    @property
    def run_dir(self) -> str:
        return os.path.join(self.run.work_dir, self.run.name)

    @property
    def ckpt_dir(self) -> str:
        return os.path.join(self.run_dir, "checkpoints")

    @property
    def curve_dir(self) -> str:
        return os.path.join(self.run_dir, "curves")

    @property
    def infer_dir(self) -> str:
        """Evaluation outputs of the configured split: <run>/inference/<split>/ (one folder per split that was looked at)."""
        return os.path.join(self.run_dir, "inference", self.infer.split)

    @property
    def data_dir(self) -> str:
        return self.data.hf_cache_dir or os.path.join(self.run.work_dir, "data")

    def make_dirs(self) -> None:
        for d in (self.run_dir, self.ckpt_dir, self.curve_dir, self.infer_dir, self.data_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Optional[str] = None) -> None:
        path = path or os.path.join(self.run_dir, "config.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Config":
        return dc_from_dict(cls, d)

    # ---- variants ------------------------------------------------------------------------------------------------------------
    def with_seed(self, seed: int) -> "Config":
        c = copy.deepcopy(self)
        c.run.seed = seed
        c.run.name = f"{self.run.name}_s{seed}"
        return c

    @classmethod
    def multi(cls, sources: Tuple[str, ...], name: Optional[str] = None) -> "Config":
        """Train on several registry corpora at once (e.g. ("iam", "rimes", "khatt")).  The alphabet is the union of the labeled
        lines, every line carries its corpus id, FRAG-RC uses one trust threshold per corpus x hardness cell, and every metric is also
        reported per corpus."""
        c = cls()
        c.data.sources = tuple(sources)
        c.run.name = name or "inklingua_" + "_".join(sources)
        return c

    @classmethod
    def full_supervised(cls) -> "Config":
        """Upper-bound baseline: all training lines labeled, no self-distillation."""
        c = cls()
        c.run.name = "inklingua_iam_fullsup"
        c.data.labeled_fraction, c.data.calib_fraction = 1.0, 0.0
        c.train.pseudo.enabled = c.train.frag.enabled = False
        return c

    @classmethod
    def labeled_only(cls, base: Optional["Config"] = None) -> "Config":
        """THE baseline: identical model, augmentation (incl. MixLine) and labeled lines, but the unlabeled lines are never used
        (no pseudo-labels, no FRAG-RC, no calibration split -- it even trains on the lines `full` holds out for calibration)."""
        c = copy.deepcopy(base) if base is not None else cls()
        c.run.name = f"{c.run.name}_labeled_only"
        c.train.pseudo.enabled = c.train.frag.enabled = False
        c.data.calib_fraction = 0.0
        return c

    @classmethod
    def make_variant(cls, name: str, base: Optional["Config"] = None) -> "Config":
        """'full' -> a copy of `base`; 'labeled_only' | 'full_supervised' | any ablation kind (see `ablation`)."""
        b = base if base is not None else cls()
        if name == "full":
            c = copy.deepcopy(b)
            c.run.name = f"{c.run.name}_full"
            return c
        if name == "labeled_only":
            return cls.labeled_only(b)
        if name == "full_supervised":
            c = copy.deepcopy(b)
            c.run.name = f"{c.run.name}_full_supervised"
            c.data.labeled_fraction, c.data.calib_fraction = 1.0, 0.0
            c.train.pseudo.enabled = c.train.frag.enabled = False
            return c
        return cls.ablation(name, b)

    ABLATIONS = ("no_pseudo", "no_frag", "legacy_pl", "no_mixline", "no_wsd", "no_consistency", "no_llm", "no_style_task", "no_ses",
                 "no_vq", "no_film", "no_confusion", "no_frame_cons", "no_hscale", "no_calibration", "no_mask", "no_aux_ctc",
                 "no_shadow", "no_risk_smoothing", "no_v9_reg", "no_posthoc", "no_strata", "no_sharpen", "ctc_only")

    @classmethod
    def ablation(cls, kind: str, base: Optional["Config"] = None) -> "Config":
        """One component removed.  kind in Config.ABLATIONS:
          no_pseudo (no unlabeled data at all), no_frag (no FRAG-RC; labeled + calibration images only), legacy_pl (v9 line-level
          pseudo-labels instead of FRAG-RC), no_mixline, no_wsd, no_strata (one global FRAG-RC threshold), no_sharpen (plain
          distillation targets), no_consistency / no_llm (legacy gates), no_style_task, no_ses, no_vq, no_film, no_confusion
          (= no leakage penalty / decorrelation), no_frame_cons, no_hscale, no_calibration, no_mask, no_aux_ctc, no_shadow,
          no_risk_smoothing, no_v9_reg (no layer drop / CTC label smoothing), no_posthoc, ctc_only (plain CTC Transformer)."""
        c = copy.deepcopy(base) if base is not None else cls()
        c.run.name = f"{c.run.name}_abl_{kind}"
        L, P, Fg = c.train.loss, c.train.pseudo, c.train.frag
        if kind == "no_pseudo":
            P.enabled = Fg.enabled = False
        elif kind == "no_frag":
            Fg.enabled = False
        elif kind == "legacy_pl":
            Fg.enabled, P.enabled = False, True
        elif kind == "no_mixline":
            c.train.mixline_prob = 0.0
        elif kind == "no_wsd":
            c.infer.wsd = False
        elif kind == "no_strata":
            Fg.strata, Fg.strata_per_domain, Fg.stratum_emphasis = 1, False, 0.0
        elif kind == "no_sharpen":
            Fg.sharpen = 1.0
        elif kind == "no_consistency":
            P.consistency = False
        elif kind == "no_llm":
            P.llm_refine = False
        elif kind == "no_style_task":
            L.style_reg = L.style_cls = L.leak = L.decorrelate = 0.0
            c.model.film = False
        elif kind == "no_ses":
            c.train.style_equivariant = False
        elif kind == "no_vq":
            c.train.vq_start_epoch = 10 ** 6
        elif kind == "no_film":
            c.model.film = False
        elif kind == "no_confusion":
            L.leak = L.decorrelate = 0.0
        elif kind == "no_frame_cons":
            L.consistency = 0.0
        elif kind == "no_hscale":
            c.train.aug_hscale_prob = 0.0
        elif kind == "no_calibration":
            c.data.calib_fraction = 0.0
        elif kind == "no_mask":
            L.masked_code = 0.0
        elif kind == "no_aux_ctc":
            L.aux_ctc = 0.0
        elif kind == "no_shadow":
            P.shadow_calibration = False
        elif kind == "no_risk_smoothing":
            P.risk_smoothing, P.cov_guard = False, False
        elif kind == "no_v9_reg":
            c.model.layer_drop, L.ctc_smooth = 0.0, 0.0
        elif kind == "no_posthoc":
            c.infer.posthoc = False
        elif kind == "ctc_only":
            P.enabled = Fg.enabled = False
            c.infer.wsd = False
            c.data.calib_fraction = 0.0
            L.aux_ctc = L.style_reg = L.style_cls = L.leak = L.consistency = L.masked_code = L.decorrelate = L.vq_commit = 0.0
            c.train.vq_start_epoch = 10 ** 6
            c.model.film = False
        else:
            raise ValueError(f"unknown ablation '{kind}' (choose from {cls.ABLATIONS})")
        return c

    @classmethod
    def smoke(cls, work_dir: str = "./inklingua_smoke") -> "Config":
        """Tiny SYNTHETIC configuration used to validate the whole pipeline offline.  Its numbers mean nothing scientifically;
        outputs are stamped as such."""
        c = cls()
        c.run.name, c.run.work_dir, c.run.smoke_test = "smoke", work_dir, True
        c.data.source, c.data.sources = "synthetic", ("synthetic",)
        c.data.img_height, c.data.max_img_width, c.data.pad_x = 32, 512, 4
        c.data.batch_size, c.data.eval_batch_size, c.data.num_workers = 16, 16, 0
        c.data.synthetic_sizes, c.data.labeled_fraction = (1200, 48, 48), 0.5
        c.data.synthetic_words = (2, 3)
        c.data.train_eval_subset = 48
        c.model.channels, c.model.d_model, c.model.heads = (8, 16, 32, 48), 64, 4
        c.model.ffn, c.model.enc_layers = 128, 2
        c.model.low_dim, c.model.codes_low, c.model.codes_high, c.model.style_dim = 32, 32, 64, 32
        c.train.epochs, c.train.lr, c.train.warmup_steps = 24, 2e-3, 60
        c.train.aug_tilt_px = 1.5
        c.train.aux_start_epoch, c.train.aux_ramp_epochs = 5, 5
        c.train.vq_start_epoch, c.train.vq_anneal_epochs = 5, 5
        c.train.save_every, c.train.noise_eval_samples = 4, 30
        c.train.pseudo.warmup_epochs, c.train.pseudo.refresh_every = 10, 3
        c.train.pseudo.min_stability, c.train.pseudo.target_cer, c.train.pseudo.min_calib = 0, 0.12, 30
        c.train.frag.warmup_epochs, c.train.frag.refresh_every, c.train.frag.min_chars, c.train.frag.diag_lines = 8, 4, 30, 40
        c.train.frag.target_err, c.train.mixline_prob = 0.12, 0.2
        c.data.val_block = 8
        c.infer.wsd_steps, c.infer.wsd_min_lines, c.infer.wsd_clusters, c.infer.wsd_batch = (3, 6), 6, 2, 8
        c.train.patience, c.train.leak_queue = 0, 256
        c.data.calib_fraction = 0.15
        c.train.pseudo.llm_refine = False
        c.llm.backend = "dummy"
        c.infer.n_vis, c.infer.n_translate, c.infer.ig_steps = 3, 4, 4
        c.infer.pl_diag_samples, c.infer.occl_window, c.infer.occl_stride = 60, 16, 8
        c.infer.llm_rescoring, c.infer.bootstrap = False, 200
        c.infer.probe_lines, c.infer.mc_dropout_samples, c.infer.uncertainty_lines = 48, 3, 48
        c.infer.counterfactual_lines, c.infer.neighbour_queries, c.infer.longline_pairs = 2, 2, 20
        return c


# ==============================================================================
# (de)serialisation helpers
# ==============================================================================
def _tupled(v):
    """JSON gives lists; the config uses tuples (also nested ones, e.g. view_sets)."""
    return tuple(_tupled(x) if isinstance(x, list) else x for x in v)


def dc_from_dict(cls, d: Dict[str, Any]):
    """Rebuild a (nested) dataclass from a plain dict (JSON round trip).  Unknown keys are ignored, so a config saved by an
    older version still loads."""
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        dflt = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore
        if dataclasses.is_dataclass(dflt) and isinstance(v, dict):
            v = dc_from_dict(type(dflt), v)
        elif f.name == "custom_sources" and isinstance(v, (list, tuple)):
            v = tuple(dc_from_dict(DatasetSpec, x) if isinstance(x, dict) else x for x in v)
        elif isinstance(dflt, tuple) and isinstance(v, list):
            v = _tupled(v)
        kwargs[f.name] = v
    return cls(**kwargs)


SUITE_PRESETS: Dict[str, Tuple[str, ...]] = {
    # the cheapest decisive experiment: does the unlabeled data help at all?
    "quick": ("labeled_only", "full"),
    # what a reviewer needs: both baselines, the previous method, and each v10/v11 component removed once
    "paper": ("labeled_only", "ctc_only", "legacy_pl", "no_frag", "no_wsd", "no_mixline", "no_strata", "full"),
    # everything (expensive)
    "all": ("labeled_only", "ctc_only", "full_supervised", "legacy_pl") + tuple(k for k in Config.ABLATIONS if k not in ("legacy_pl", "no_pseudo")) + ("full",),
}
