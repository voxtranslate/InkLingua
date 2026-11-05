"""
Fast invariant tests (CPU, no download, no real data) and the end-to-end smoke on GENERATED lines.  Repository tests only: a real run, the
experiment plan and the Kaggle notebook never use this module or its generated data.

`run_self_tests()` guards what a reader of the paper has to be able to trust: decoding and metrics are right, padding does not change a
prediction, the style-equivariant targets follow the geometry, hidden transcriptions cannot reach training, the alphabet is grapheme-cluster
based and RTL-safe, the trust rules do what they say, warm starts preserve the function, a resumed run is bit-identical.

`run_smoke()` trains a tiny model on generated lines, resumes, evaluates, builds the report (minutes on CPU): `tests/test_smoke.py`.
The line generator and the tiny configuration (`smoke_config`) live here, registered in data.GENERATED_SOURCES at import.
"""
from __future__ import annotations

import copy
import csv
import json
import math
import os
import random
import shutil
import tempfile
import time
import unicodedata
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from .alphabet import extend_codec, extend_ctc_heads, resolve_variants, visual_base_of
from .config import SUITE_PRESETS, Config, DatasetSpec
from .data import GENERATED_SOURCES, Collator, HandwritingDataModule, SplitSource, font_paths, load_font, preprocess_line, to_u8
from .datasets_registry import DATASETS
from .decoding import ctc_prefix_beam_search, decode_with_space_bonus, greedy_decode, greedy_runs
from .explain import adapt_bn_state, bn_layers, code_char_stats, contrast_rollout, load_bn, snapshot_bn
from .frag import FragModel, cell_theta, char_evidence, char_wrong_labels, fragment_mask
from .llm import LLMEngine, calibrate_word_mark, mark_uncertain
from .model import ClosedFormLeakage, FiLM, GPUAugment, InkLinguaNet, body_ratio_t, frame_consistency, fragment_distillation, transform_slant_deg
from .protocol import log_test_look, write_protocol
from .pseudo import PseudoLabelBank, SelectionRule, reliability_features, reliability_views
from .render import _baseline_y, _draw_line, render_styled_text_info, style_mismatch
from .riskcontrol import auroc, char_risk_threshold, domain_shift, pava_increasing, risk_control_threshold, risk_curve, transfer_risk_control, weighted_ks, weighted_risk_curve
from .style import NUM_DESC, STYLE_FLOORS, body_ratio_from_rows, compute_style_descriptors, estimate_slant_deg
from .suite import suite_markdown, summarise_suite
from .text import (CharCodec, bootstrap_cer_ci, cer, char_class, chrf, ctc_feasible, detokenize, grapheme_clusters, paired_bootstrap_delta, ridge_cv_r2,
                   to_logical, to_visual, wer)
from .trainer import Trainer, warm_start
from .utils import LOG, restore_logging

RELIABILITY_N = 10



# ==============================================================================
# generated lines (tests only) and the tiny test configuration
# ==============================================================================
_SYN_WORDS = ("the of and to in is that it was for on are with as his they be at one have this from or had by "
              "hot word but what some we can out other were all there when up use your how said an each she "
              "which do their time if will way about many then them write would like so these her long make "
              "thing see him two has look more day could go come did number sound no most people my over know "
              "water than call first who may down side been now find head stand own page should country found "
              "answer school grow study still learn plant cover food sun four between state keep eye never last "
              "let thought city tree cross farm hard start might story saw far sea draw left late run").split()



def render_synthetic_line(text: str, rng: random.Random, fonts: List[str]) -> Image.Image:
    """Offline 'handwriting': varied fonts / slant / stroke / blur / noise (for smoke tests only)."""
    font = load_font(rng.choice(fonts) if fonts else None, rng.randint(40, 56))
    stroke = rng.choice([0, 0, 1, 2])
    ink = int(rng.uniform(0, 90))
    tmp = ImageDraw.Draw(Image.new("L", (8, 8)))
    x0, y0, x1, y1 = tmp.textbbox((0, 0), text, font=font, stroke_width=stroke)
    w = (x1 - x0) + 60
    canvas = Image.new("L", (w, 128), 255)
    ImageDraw.Draw(canvas).text((30 - x0, 64 - (y1 - y0) // 2 - y0), text, font=font, fill=ink, stroke_width=stroke, stroke_fill=ink)
    shear = rng.uniform(-0.35, 0.35)
    canvas = canvas.transform(canvas.size, Image.AFFINE, (1, shear, -shear * 64, 0, 1, 0), resample=Image.BILINEAR, fillcolor=255)
    arr = np.asarray(canvas, np.float32)
    arr = cv2.GaussianBlur(arr, (0, 0), rng.uniform(0.3, 1.1))
    arr = np.clip(arr + np.random.RandomState(rng.randint(0, 10 ** 6)).normal(0, rng.uniform(1, 6), arr.shape), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


class SyntheticSource(SplitSource):
    def __init__(self, n: int, seed: int, words: Tuple[int, int] = (3, 6)):
        rng, fonts = random.Random(seed), font_paths()
        self._texts = [" ".join(rng.choice(_SYN_WORDS) for _ in range(rng.randint(*words))) for _ in range(n)]
        self._imgs = [render_synthetic_line(t, rng, fonts) for t in self._texts]

    def __len__(self): return len(self._texts)
    def image(self, j): return self._imgs[int(j)]
    def text(self, j): return self._texts[int(j)]


class ListSource(SplitSource):
    """Tiny in-memory split rendered from a list of texts (self-tests)."""

    def __init__(self, texts: List[str], seed: int = 0):
        rng, fonts = random.Random(seed), font_paths()
        self._t = list(texts)
        self._im = [render_synthetic_line(t, rng, fonts) for t in texts]

    def __len__(self): return len(self._t)
    def image(self, j): return self._im[int(j)]
    def text(self, j): return self._t[int(j)]


def smoke_config(work_dir: str = "./inklingua_smoke") -> Config:
    """Tiny SYNTHETIC configuration used to validate the whole pipeline offline.  Its numbers mean nothing scientifically;
    outputs are stamped as such."""
    c = Config()
    c.run.name, c.run.work_dir, c.run.smoke_test = "smoke", work_dir, True
    c.train.auto_schedule = False
    c.data.line_cache_dir = None
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


def _generated_source(dm: HandwritingDataModule, spec: DatasetSpec) -> Dict[str, SplitSource]:
    """The 'synthetic' source of smoke_config(): generated lines, sizes / words per line from cfg.data.synthetic_* (test attributes)."""
    sizes, words, s = getattr(dm.dc, "synthetic_sizes", (600, 120, 120)), getattr(dm.dc, "synthetic_words", (3, 6)), dm.cfg.run.seed
    dm.provenance.setdefault("sources", {})[spec.key] = {"synthetic": True, "sizes": list(sizes)}
    return {"train": SyntheticSource(sizes[0], s, words), "val": SyntheticSource(sizes[1], s + 1, words), "test": SyntheticSource(sizes[2], s + 2, words)}


DATASETS.setdefault("synthetic", DatasetSpec(key="synthetic", hf_dataset="", language="en", script="Latin", note="generated test lines"))
GENERATED_SOURCES["synthetic"] = _generated_source


def run_self_tests(verbose: bool = True) -> None:
    t0, done = time.time(), []

    def ok(name: str) -> None:
        done.append(name)
        if verbose:
            LOG.info("self-test ok: %s", name)

    # ---- CTC decoding --------------------------------------------------------------------------------------------------------
    V = 4
    path = [0, 1, 1, 0, 1, 2, 2, 0, 3]
    lp = np.log(np.full((len(path), V), 0.02))
    for t, c in enumerate(path):
        lp[t, c] = np.log(0.94)
    ids, cf, fr = greedy_decode(lp)
    assert ids == [1, 1, 2, 3] and len(cf) == 4 and fr[0] < fr[1] < fr[2] < fr[3], (ids, fr)
    ids_r, st_r, en_r = greedy_runs(lp)
    assert ids_r == [1, 1, 2, 3] and st_r.tolist() == [1, 4, 5, 8] and en_r.tolist() == [3, 5, 7, 9]
    nb = ctc_prefix_beam_search(lp, beam=4)
    assert list(nb[0][0]) == [1, 1, 2, 3] and nb[0][1] <= 0.0 and all(nb[i][1] >= nb[i + 1][1] for i in range(len(nb) - 1))
    ok("CTC greedy / runs / prefix beam search")

    # ---- text: metrics, grapheme codec, RTL, detokeniser, bootstrap -------------------------------------------------------------
    assert abs(cer("abcd", "abxd") - 0.25) < 1e-9 and wer("a b c", "a x c") == 1 / 3 and cer("", "") == 0.0
    assert chrf("hello world", "hello world") == 100.0 and chrf("abc", "xyz") == 0.0
    for raw_, want_ in [('He said , " go . "', 'He said, "go."'), ("it 's do n't", "it's don't"), ("( a ) b , c ;", "(a) b, c;"),
                        ("`` hi '' , he said", "``hi'', he said"), ('say " a " and " b "', 'say "a" and "b"')]:
        got_ = detokenize(raw_)
        assert got_ == want_, (raw_, got_, want_)
    assert mark_uncertain("ab cd", [0.9, 0.9, 0.0, 0.2, 0.3], 0.7) == "ab cd[?]"
    Eg = unicodedata.normalize("NFC", "ẹ́")
    assert grapheme_clusters("ẹ́a") == [Eg, "a"] and grapheme_clusters("क्ष") == ["क्", "ष"]
    cd = CharCodec.from_texts(["ab " + Eg, "b"])
    assert cd.units == [" ", "a", "b", Eg] and cd.num_classes == 5 and cd.decode_ids(cd.encode("b" + Eg + "a")) == "b" + Eg + "a"
    assert CharCodec(cd.chars) == cd and cd.encode("z") == []                                  # chars round trip; unknown units dropped
    assert ctc_feasible("ab", 2) and not ctc_feasible("aa", 2) and ctc_feasible("aa", 3) and not ctc_feasible("", 5) and ctc_feasible(Eg + Eg, 3)
    ar = "سلام عليكم"
    vis = to_visual(ar, True)
    assert vis != ar and to_logical(vis, True) == ar and to_visual(ar, False) == ar and len(vis) == len(ar)
    assert cer(ar, ar.replace("م", "ن", 1)) == cer(to_visual(ar, True), to_visual(ar.replace("م", "ن", 1), True))      # metrics invariant under reversal
    assert char_class("a") == "lower" and char_class("A") == "upper" and char_class("س") == "letter" and char_class("中") == "letter" and char_class("٣") == "digit"
    assert char_class(".") == "punct" and char_class(" ") == "space"
    ed = np.array([0, 1, 2, 0, 3.0])
    rl = np.array([10, 10, 10, 10, 10.0])
    lo, hi = bootstrap_cer_ci(ed, rl, 300, 0)
    assert lo <= ed.sum() / rl.sum() <= hi
    d = paired_bootstrap_delta(ed * 0.5, ed, rl, 300, 0)
    assert d["delta"] < 0 and d["hi"] <= 0.0 and d["p_a_better"] > 0.9
    rg = np.random.RandomState(3)
    base_ = np.repeat(rg.rand(60) * 6, 10)
    edc = rg.poisson(base_).astype(np.float64)
    rlc = np.full(600, 40.0)
    lo_l, hi_l = bootstrap_cer_ci(edc, rlc, 600, 0)
    lo_g, hi_g = bootstrap_cer_ci(edc, rlc, 600, 0, groups=np.arange(600) // 10)
    assert (hi_g - lo_g) > 1.5 * (hi_l - lo_l) and lo_g <= edc.sum() / rlc.sum() <= hi_g
    dg = paired_bootstrap_delta(edc * 0.5, edc, rlc, 400, 0, groups=np.arange(600) // 10, n_tests=4)
    assert dg["hi_adj"] >= dg["hi"] and dg["lo_adj"] <= dg["lo"] and dg["delta"] < 0
    Xq = np.random.RandomState(4).randn(200, 3)
    Yq = np.column_stack([Xq[:, 0] + 0.1 * np.random.RandomState(5).randn(200)])
    r_a, r_b = ridge_cv_r2(Xq, Yq, np.ones_like(Yq, bool)), ridge_cv_r2(Xq, Yq + 50.0, np.ones_like(Yq, bool))
    assert r_a[0] > 0.8 and abs(r_a[0] - r_b[0]) < 1e-4
    ok("text metrics, grapheme-cluster codec, RTL round trip (metrics invariant), Unicode character classes, form-clustered bootstrap")

    # ---- [CAL-MARK] and the renderer baseline ----------------------------------------------------------------------------------
    rm_ = random.Random(2)
    preds_m, gts_m, confs_m = [], [], []
    for _ in range(120):
        gw = ["".join(rm_.choice("abcdef") for _ in range(rm_.randint(3, 7))) for _ in range(6)]
        pw, cw = [], []
        for w in gw:
            bad = rm_.random() < 0.3
            pw.append(("z" + w[1:]) if bad else w)
            c = (0.93 + 0.05 * rm_.random()) if bad else (0.985 + 0.014 * rm_.random())
            cw += [c] * len(w)
            cw.append(1.0)
        preds_m.append(" ".join(pw)), gts_m.append(" ".join(gw)), confs_m.append(cw[:-1])
    cal_m = calibrate_word_mark(preds_m, gts_m, confs_m, 0.5)
    assert cal_m["threshold"] is not None and cal_m["precision"] >= 0.5 and cal_m["recall"] > 0.3 and cal_m["auroc"] > 0.9 and 0.93 < cal_m["threshold"] < 0.999
    assert calibrate_word_mark(preds_m[:2], gts_m[:2], confs_m[:2])["threshold"] is None
    if font_paths():
        Hs_t, sz_t = 128, 60
        fnt_t = ImageFont.truetype(font_paths()[0], sz_t)
        inkp = lambda t: _draw_line(t, font_paths()[0], sz_t, 0.0, 10, 0, 0.0, Hs_t)  # noqa: E731
        rows_of = lambda a: np.flatnonzero(a.max(1) > 0.3)  # noqa: E731
        base_t = _baseline_y(fnt_t, Hs_t)
        rd, rp, rx = rows_of(inkp(".")), rows_of(inkp("p")), rows_of(inkp("x"))
        assert abs(rd.max() - base_t) <= 3 and rp.max() > base_t + 0.12 * sz_t and abs(rx.max() - base_t) <= 3
    ok("calibrated [?] marks, full stops and descenders on one baseline")

    # ---- preprocessing ---------------------------------------------------------------------------------------------------------
    img = Image.fromarray((255 - 200 * (np.random.RandomState(0).rand(128, 800) > 0.9)).astype(np.uint8))
    ink = preprocess_line(img, 64, 2560, 32, 8)
    assert ink.shape == (64, 400 + 16) and ink[:, :8].max() == 0 and ink[:, -8:].max() == 0 and 0 <= ink.min() <= ink.max() <= 1
    assert preprocess_line(Image.fromarray(np.full((128, 9000), 255, np.uint8)), 64, 2560, 32, 8).shape[1] <= 2560
    ok("line preprocessing")

    # ---- model: padding invariance (train and eval), hidden states ------------------------------------------------------------
    mc = copy.deepcopy(smoke_config().model)
    mc.dropout, mc.layer_drop = 0.0, 0.0
    torch.manual_seed(0)
    net = InkLinguaNet(mc, 30, NUM_DESC, 10, 32)
    x = torch.rand(4, 1, 32, 96)
    w = torch.tensor([96, 80, 60, 40])
    for i, ww in enumerate(w):
        x[i, :, :, ww:] = 0
    x2 = torch.cat([x, torch.zeros(4, 1, 32, 64)], -1)
    net.train()
    net(x, w, vq_lambda=0.7)
    for mode in ("train", "eval"):
        a, b = InkLinguaNet(mc, 30, NUM_DESC, 10, 32), InkLinguaNet(mc, 30, NUM_DESC, 10, 32)
        a.load_state_dict(net.state_dict()), b.load_state_dict(net.state_dict())
        getattr(a, mode)(), getattr(b, mode)()
        o1, o2 = a(x, w, vq_lambda=0.7), b(x2, w, vq_lambda=0.7)
        for k in range(4):
            L = int(o1["out_lens"][k])
            assert (o1["log_probs"][k, :L] - o2["log_probs"][k, :L]).abs().max().item() < 1e-3, f"padding changes the output in {mode} mode"
    net.eval()
    oh = net(x, w, keep_hidden=True, extras=True)
    assert len(oh["hidden"]) == mc.enc_layers + 1 and oh["style_att"].shape == (4, oh["log_probs"].shape[1])
    ok("padding invariance (train and eval), hidden states and style attention exposed")

    # ---- style descriptors -----------------------------------------------------------------------------------------------------
    H_, W_ = 64, 400

    def bars(angle: float, n: int = 10) -> np.ndarray:
        m = np.zeros((H_, W_), np.float32)
        t = math.tan(math.radians(angle))
        for k in range(n):
            for y in range(8, H_ - 8):
                xx = int(round(20 + 36 * k - t * (y - (H_ - 1) / 2.0)))
                m[y, xx:xx + 3] = 1.0
        return m

    for a_ in (-20, -8, 0, 12, 25):
        est = estimate_slant_deg((bars(a_) > 0.45).astype(np.uint8))
        assert abs(est - a_) < 1.5, (a_, est)
    assert math.isnan(estimate_slant_deg(np.zeros((H_, W_), np.uint8)))
    assert np.isnan(compute_style_descriptors(np.zeros((H_, W_), np.float32))).all()
    assert np.isfinite(compute_style_descriptors(bars(10))).all()
    xr = (torch.rand(5, 1, 64, 200, generator=torch.Generator().manual_seed(1)) > 0.8).float()
    for b_ in range(5):
        assert abs(float(body_ratio_t(xr)[b_]) - body_ratio_from_rows((xr[b_, 0] > 0.45).sum(1).numpy(), 64)) < 1e-6
    ok("style descriptors (slant estimator, NaN = not measurable, body ratio)")

    # ---- [SES] -----------------------------------------------------------------------------------------------------------------
    src = ListSource(["the quick brown fox", "jumps over the lazy dog", "handwriting recognition", "pack my box"] * 6, 3)
    inks, wid = [], []
    for i in range(len(src)):
        u = to_u8(preprocess_line(src.image(i), 64, 400, 32, 8)).astype(np.float32) / 255.0
        wid.append(min(u.shape[1], 384))
        inks.append(np.pad(u, ((0, 0), (0, max(0, 384 - u.shape[1]))))[:, :384])
    inks = np.stack(inks)
    X = torch.from_numpy(inks)[:, None]
    d0 = np.stack([compute_style_descriptors(i) for i in inks])
    B, _, Hh, Ww = X.shape
    for shear in (0.10, -0.10):
        sh, rot, vs = torch.full((B,), shear), torch.zeros(B), torch.ones(B)
        th = torch.zeros(B, 2, 3)
        th[:, 0, 0], th[:, 0, 1], th[:, 1, 1] = 1, (sh - rot) / (Ww / Hh), vs
        Y = F.grid_sample(X, F.affine_grid(th, list(X.shape), align_corners=False), padding_mode="zeros", align_corners=False)
        d1 = np.stack([compute_style_descriptors(y[0].numpy()) for y in Y])
        pred = transform_slant_deg(torch.from_numpy(d0[:, 0]), sh, rot, vs).numpy()
        fin = np.isfinite(d1[:, 0]) & np.isfinite(pred)
        assert fin.sum() >= B // 2 and abs((d1[fin, 0] - pred[fin]).mean()) < 1.5
    tc = copy.deepcopy(smoke_config().train)
    tc.aug_geom_prob, tc.aug_hscale_prob, tc.aug_elastic_prob, tc.aug_morph_prob, tc.aug_blur_prob = 1.0, 1.0, 0.0, 0.0, 0.0
    with np.errstate(all="ignore"):
        mu_, sd_ = np.nanmean(d0, 0), np.maximum(np.nanstd(d0, 0), STYLE_FLOORS)
    aug = GPUAugment(tc, mu_, sd_)
    okd = np.isfinite(d0)
    z = torch.from_numpy(np.clip(np.nan_to_num((d0 - mu_) / sd_), -3, 3)).float()
    torch.manual_seed(0)
    xs_, xw_, wd_, st_, sm_, cok_ = aug(X, z, torch.from_numpy(okd), torch.tensor(wid), torch.full((B,), 300))
    assert xs_.shape == X.shape and float(xs_.min()) >= 0 and float(xs_.max()) <= 1 and int(wd_.max()) <= Ww and int(wd_.min()) >= 300
    assert (sm_[~torch.from_numpy(okd)] == 0).all()
    e_sl, e_body = [], []
    for b_ in range(B):
        dm_ = compute_style_descriptors(xw_[b_, 0, :, :int(wd_[b_])].numpy())
        if sm_[b_, 0] > 0 and np.isfinite(dm_[0]):
            e_sl.append(dm_[0] - float(st_[b_, 0] * sd_[0] + mu_[0]))
        if sm_[b_, 4] > 0 and abs(float(st_[b_, 4])) < 2.9:
            e_body.append(dm_[4] - float(st_[b_, 4] * sd_[4] + mu_[4]))
    assert len(e_sl) >= B // 2 and abs(np.mean(e_sl)) < 1.5 and np.mean(np.abs(e_sl)) < 3.0
    assert len(e_body) >= B // 2 and np.mean(np.abs(e_body)) < 0.03
    ok("style-equivariant targets (slant closed form, re-measured body ratio, widths, masks)")

    # ---- FiLM, consistency, leakage --------------------------------------------------------------------------------------------
    fl = FiLM(8, 16, 0.5)
    xf = torch.randn(2, 5, 16)
    yf, g_, b_ = fl(xf, torch.randn(2, 8))
    assert torch.allclose(yf, xf)
    torch.nn.init.normal_(fl.lin.weight, std=10.0)
    _, g_, b_ = fl(xf, torch.randn(2, 8) * 10)
    assert float(g_.detach().abs().max()) <= 0.5 + 1e-6
    lpa = F.log_softmax(torch.randn(2, 10, 5), -1)
    lpb = F.log_softmax(torch.randn(2, 10, 5), -1)
    lens_ = torch.tensor([10, 6])
    assert abs(float(frame_consistency(lpa, lpa, lens_))) < 1e-6 and float(frame_consistency(lpb, lpa, lens_)) > 0
    gen = torch.Generator().manual_seed(0)
    act = np.array([1, 1, 0, 0, 0], bool)
    for leaky, floor, expect in ((False, 0.0, "low"), (True, 0.0, "high"), (True, 0.95, "low")):
        cfl = ClosedFormLeakage(8, 5, 256, 0.3, 0.02, act, np.full(5, floor), torch.device("cpu"))
        pen = None
        for _ in range(40):
            yy = torch.randn(16, 5, generator=gen)
            cc = torch.randn(16, 8, generator=gen)
            if leaky:
                cc[:, 0] = yy[:, 0] + 0.1 * torch.randn(16, generator=gen)
            cc.requires_grad_(True)
            pen = cfl(cc, yy, torch.ones(16, 5))
        assert (float(pen) > 0.2) if expect == "high" else (float(pen) < 0.1), (leaky, floor, float(pen))
    ok("FiLM bounded, teacher-student consistency, closed-form leakage penalty")

    # ---- data module: hidden labels, calibration lines, alphabet, multi-corpus + RTL, provenance, synthetic guard ------------------
    texts = [f"{chr(0x100 + i) * 3} ab" for i in range(40)]
    tmp = tempfile.mkdtemp()
    c = smoke_config(tmp)
    c.data.labeled_fraction, c.data.max_text_len = 0.5, 30
    dm = HandwritingDataModule(c)
    dm._load_sources = lambda: [{"train": ListSource(texts, 1), "val": ListSource(texts[:8], 2), "test": ListSource(texts[8:16], 3)}]   # type: ignore
    dm.prepare()
    tr = dm.train_ds
    assert abs(int(tr.labeled.sum() + tr.calib.sum()) - len(tr) / 2) <= 1 and tr.calib.sum() >= 2 and not (tr.labeled & tr.calib).any()
    for i in range(len(tr)):
        it = tr[i]
        if tr.labeled[i]:
            assert it["text"] == tr.oracle[i] != "" and it["weight"] == 1.0 and it["oracle"] == tr.oracle[i]
        else:
            assert it["text"] == "" and it["oracle"] == "" and it["weight"] == 0.0, "hidden / calibration text leaked into a batch"
            assert tr.reference(i) == (tr.oracle[i] if tr.calib[i] else "")
    n_tr = len(tr)
    tr.set_pseudo(["ab"] * n_tr, np.ones(n_tr, np.float32), include_calib=False)
    assert all(tr.tgt_weight[i] == 0.0 for i in np.flatnonzero(tr.calib))
    tr.set_pseudo(["ab"] * n_tr, np.ones(n_tr, np.float32), include_calib=True)
    assert all(tr.tgt_text[i] == "ab" and tr.tgt_weight[i] == 1.0 for i in np.flatnonzero(tr.calib)) and all(tr.tgt_text[i] == tr.oracle[i] for i in np.flatnonzero(tr.labeled))
    assert all(tr[i]["oracle"] == "" for i in np.flatnonzero(tr.calib))
    tr.set_pseudo([""] * n_tr, np.zeros(n_tr, np.float32), include_calib=True)
    pool = tr.labeled | tr.calib
    lab_chars = set("".join(t for t, l in zip(tr.oracle, pool) if l))
    unl_only = set("".join(t for t, l in zip(tr.oracle, pool) if not l)) - lab_chars
    assert unl_only and not (unl_only & set(dm.codec.units)), "alphabet leaks characters of unlabeled lines"
    batch = Collator(dm.codec)([tr[i] for i in range(len(tr))])
    assert int(batch["target_len"].sum()) == sum(len(dm.codec.encode(tr.oracle[i])) for i in np.flatnonzero(tr.labeled))
    assert all(o == "" for o, l in zip(batch["oracle"], batch["labeled"].tolist()) if not l)
    assert dm.val_ds.reference(0) == texts[0] and os.path.exists(os.path.join(c.run_dir, "provenance.json"))
    c2 = smoke_config(tmp)
    c2.run.smoke_test = False
    try:
        HandwritingDataModule(c2)._load_sources()
        raise AssertionError("synthetic data must be refused outside smoke_config()")
    except RuntimeError:
        pass
    # two corpora, one of them right-to-left: domains, visual order, display, per-domain strata, MixLine within a corpus only
    ar_lines = ["سلام عليكم يا صديقي", "كتاب على الطاولة", "ذهب الولد الى المدرسة", "القط ينام في الحديقة"] * 10
    en_lines = ["the cat sat on the mat", "a quick brown fox", "hello world again", "pack my box with jugs"] * 10
    c3 = smoke_config(tmp)
    c3.data.sources, c3.data.labeled_fraction, c3.data.max_text_len = ("synthetic", "arabic_test"), 0.6, 40
    c3.data.custom_sources = (DatasetSpec(key="arabic_test", language="ar", script="Arabic", rtl=True),)
    c3.train.frag.strata = 2
    dm3 = HandwritingDataModule(c3)
    dm3._load_sources = lambda: [{"train": ListSource(en_lines, 1), "val": ListSource(en_lines[:12], 2), "test": ListSource(en_lines[:8], 3)},        # type: ignore
                                 {"train": ListSource(ar_lines, 4), "val": ListSource(ar_lines[:12], 5), "test": ListSource(ar_lines[:8], 6)}]
    dm3.prepare()
    t3 = dm3.train_ds
    assert set(t3.domain.tolist()) == {0, 1} and t3.domain_names == ["synthetic", "arabic_test"] and dm3.n_strata == 4
    ar_rows = np.flatnonzero(t3.domain == 1)
    assert all(t3.rtl[i] for i in ar_rows) and not t3.rtl[np.flatnonzero(t3.domain == 0)[0]]
    i_ar = int(ar_rows[0])
    assert t3.oracle[i_ar] == to_visual(ar_lines[int(t3.indices[i_ar])], True) and t3.display(i_ar) == ar_lines[int(t3.indices[i_ar])]
    assert set(t3.stratum[t3.domain == 0]) <= {0, 1} and set(t3.stratum[t3.domain == 1]) <= {2, 3}
    assert dm3.test_ds.domain_names == t3.domain_names and dm3.val_ds.rtl.sum() == 12 and "ar" in [s.language for s in dm3.specs]
    col3 = Collator(dm3.codec, mix_prob=1.0, max_w=4096, gap=(2, 2), seed=1)
    la, lb = [int(i) for i in np.flatnonzero(t3.labeled & (t3.domain == 0))[:1]], [int(i) for i in np.flatnonzero(t3.labeled & (t3.domain == 1))[:1]]
    bmix = col3([t3[la[0]], t3[lb[0]]])
    assert bmix["image"].shape[0] == 2, "MixLine must not concatenate lines of different corpora"
    assert bmix["domain"].tolist() == [0, 1]
    shutil.rmtree(tmp, ignore_errors=True)
    ok("data module: hidden labels / calibration / alphabet / synthetic guard; two corpora with one RTL (visual order, display, per-corpus cells, MixLine)")

    # ---- risk control, reliability model, pseudo-label gates ------------------------------------------------------------------
    rng_ = np.random.RandomState(0)
    n_ = 600
    s_ = rng_.rand(n_)
    ed_ = rng_.binomial(40, 0.4 * s_ ** 3).astype(np.float64)
    rl_ = np.full(n_, 40.0)
    assert np.allclose(pava_increasing(np.array([3.0, 1.0, 2.0, 5.0, 4.0]), np.ones(5)), [2, 2, 2, 4.5, 4.5])
    cv = risk_curve(s_, ed_, rl_, True)
    assert np.all(np.diff(cv["smooth"]) >= -1e-12)
    th1, k1, c1 = risk_control_threshold(s_, ed_, rl_, 0.02, 1.0, 15)
    th2, k2, _ = risk_control_threshold(s_, ed_, rl_, 0.08, 1.0, 15)
    assert k1 >= 15 and k2 > k1 and th2 >= th1 and risk_control_threshold(s_, ed_ + 5, rl_, 0.02, 1.0, 15)[1] == 0
    assert auroc(np.array([1, 2, 3, 4.0]), np.array([0, 0, 1, 1], bool)) == 1.0
    Xr = np.column_stack([s_ + 0.15 * rng_.randn(n_)] + [rng_.randn(n_) for _ in range(RELIABILITY_N - 1)])
    from .config import PseudoConfig
    rule = SelectionRule.fit(Xr, ed_, rl_, PseudoConfig(), seed=0)
    assert rule.info["calib_auroc_error_detection"] > 0.7 and rule.predict(Xr).shape == (n_,)
    pcg = PseudoConfig()
    free = SelectionRule.fit(Xr, ed_, rl_, pcg, seed=0)
    capped = SelectionRule.fit(Xr, ed_, rl_, pcg, seed=0, prev_cov=0.0)
    assert capped.info["calib_selected"] <= int(math.floor(pcg.max_cov_growth * n_)) and free.info["calib_selected"] > capped.info["calib_selected"]
    lp_ = F.log_softmax(torch.randn(30, 6), -1).numpy()
    ids_, cf_, fr_ = greedy_decode(lp_)
    assert reliability_features(lp_, ids_, cf_, fr_, [0.0, 0.2]).shape == (RELIABILITY_N,) and len(reliability_views(PseudoConfig())) == 3
    pc = PseudoConfig()
    pc.min_stability, pc.lops_drop, pc.fallback_tau_conf = 0, 0.0, 0.9
    bank = PseudoLabelBank(7, np.ones(7, bool), ["hello"] * 7, np.full(7, 50))
    bank.begin_round()
    bank.update_predictions(range(7), ["hello", "hello", "h", "hello", "hello", "x" * 100, "hello"], [0.99, 0.99, 0.99, 0.99, 0.50, 0.99, 0.62],
                            [0.0, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0], [0.05, 0.05, 0.05, 0.05, 0.90, 0.05, 0.40])
    bank.llm_ok[6], bank.llm_text[6] = True, "hello"
    bank.llm_ok[1], bank.llm_text[1] = True, "hello"
    bank.llm_ok[4], bank.llm_text[4] = True, "hello"
    bank.select(pc, 0.30, 0.50)
    assert bank.sel.tolist() == [True, False, False, True, False, False, True] and bank.source.tolist() == [1, 0, 0, 1, 0, 0, 2]
    bank.select(pc, None)
    assert bank.sel.tolist() == [True, False, False, True, False, False, False]
    ok("line-level risk control, reliability model, pseudo-label gates (legacy baseline)")

    # ---- test-time adaptation helpers, space bonus -----------------------------------------------------------------------------
    src_ = {"a": (np.array([3.0]), np.array([1.0]))}
    acc_ = {"a": {"n": 10.0, "s1": np.array([10.0]), "s2": np.array([20.0])}}
    ad_ = adapt_bn_state(src_, acc_, 10, 10.0)
    assert abs(ad_["a"][0][0] - 2.0) < 1e-9 and abs(ad_["a"][1][0] - 2.0) < 1e-9
    lps_ = np.log(np.array([[0.1, 0.8, 0.1], [0.5, 0.1, 0.4], [0.9, 0.05, 0.05]]))
    assert decode_with_space_bonus(lps_, 2, 0.0) == [1] and decode_with_space_bonus(lps_, 2, 1.5) == [1, 2] and decode_with_space_bonus(lps_, None, 9.0) == [1]
    cm = smoke_config().model
    net_ = InkLinguaNet(cm, 24, NUM_DESC, 4, 32).eval()
    for _, m_ in bn_layers(net_):
        m_.running_mean.normal_(0, 0.5), m_.running_var.uniform_(0.5, 2.0)
    x_, w_ = torch.rand(4, 1, 32, 64), torch.tensor([64, 60, 52, 64])
    snap_ = snapshot_bn(net_)
    with torch.no_grad():
        base_o = net_(x_, w_)["log_probs"].clone()
        for _, m_ in bn_layers(net_):
            m_.collect = {}
        net_(x_, w_)
        acc2_ = {n: m_.collect for n, m_ in bn_layers(net_)}
        for _, m_ in bn_layers(net_):
            m_.collect = None
        after_ = snapshot_bn(net_)
        assert all(np.array_equal(snap_[k][0], after_[k][0]) for k in snap_)
        load_bn(net_, adapt_bn_state(snap_, acc2_, 4, 0.0))
        moved_ = net_(x_, w_)["log_probs"]
        load_bn(net_, adapt_bn_state(snap_, acc2_, 4, 1e15))
        same_ = net_(x_, w_)["log_probs"]
        load_bn(net_, snap_)
    assert (moved_ - base_o).abs().max() > 1e-4 and (same_ - base_o).abs().max() < 1e-4
    ok("AdaBN helpers (collection pass leaves the model untouched), space bonus")

    # ---- closed-loop renderer ----------------------------------------------------------------------------------------------------
    if font_paths():
        tg_ = [([24.5, 0.061, 0.106, 0.356, 0.28], "Parfois ils ont parle du theme"), ([-4.2, 0.056, 0.094, 0.356, 0.35], "et esthetique aussi")]
        e0_, e1_ = 0.0, 0.0
        for d_, x_t in tg_:
            inf_ = render_styled_text_info(x_t, np.array(d_), 64, iters=6)
            assert inf_["mismatch"] <= inf_["mismatch_round0"] + 1e-12 and inf_["image"].shape[0] == 64
            e0_, e1_ = e0_ + inf_["mismatch_round0"], e1_ + inf_["mismatch"]
        assert e1_ < 0.6 * e0_, (e0_, e1_)
        assert style_mismatch(np.array([np.nan] * 5), np.array([1.0] * 5)) == float("inf")
        ok("closed-loop style renderer (re-measured, corrected, never worse than round 0)")

    # ---- post-hoc options on a fake model (validation-gated adoption) ------------------------------------------------------------
    from .inferencer import Inferencer

    class _DS:
        def __init__(self, texts):
            self.oracle = texts
            self.style = np.zeros((len(texts), NUM_DESC), np.float32)
        def __len__(self): return len(self.oracle)

    codec_ = CharCodec("ab ")
    rs4 = np.random.RandomState(4)

    def fake_lp(text: str) -> np.ndarray:
        rows = []
        for ch in text:
            p = np.full(4, 1e-3)
            if ch == " ":
                p[codec_.encode(" ")[0]], p[0] = 0.30, 0.69
            else:
                p[codec_.encode(ch)[0]], p[0] = 0.95, 0.049
            rows.append(p / p.sum())
            blank = np.full(4, 1e-3)
            blank[0] = 0.997
            rows.append(blank / blank.sum())
        return np.log(np.array(rows, np.float32))

    def rand_text() -> str:
        return " ".join("".join(rs4.choice(list("ab"), rs4.randint(2, 5))) for _ in range(rs4.randint(2, 4)))

    v_txt, t_txt = [rand_text() for _ in range(80)], [rand_text() for _ in range(80)]
    inf_ = object.__new__(Inferencer)
    inf_.ic = copy.deepcopy(Config().infer)
    inf_.ic.tta, inf_.ic.bootstrap = False, 200
    inf_.data = SimpleNamespace(val_ds=_DS(v_txt), codec=codec_, val_gate_idx=np.arange(len(v_txt)))               # type: ignore
    inf_.model, inf_.frag_model, inf_.frag_state = None, None, None
    inf_._lp_pass = lambda ds, ids, assign=None, states=None, view=None: {int(i): fake_lp(ds.oracle[int(i)]) for i in ids}   # type: ignore
    R_ = {"idx": list(range(80))}
    ph_ = inf_._posthoc_block(_DS(t_txt), R_)
    assert ph_["test_CER_base"] > 0.05 and ph_["space_bonus"]["adopted"] and ph_["space_bonus"]["chosen"] >= 1.0
    assert ph_["test_CER_final"] == 0.0 and [R_["pred_posthoc"][i] for i in range(80)] == t_txt
    v2, t2 = ["ab" * 3] * 40, ["ba" * 3] * 40
    inf_.data = SimpleNamespace(val_ds=_DS(v2), codec=codec_, val_gate_idx=np.arange(len(v2)))                  # type: ignore
    ph2_ = inf_._posthoc_block(_DS(t2), {"idx": list(range(40))})
    assert not ph2_["adopted"]["tta"] and ph2_["adopted"]["space_bonus"] == 0.0 and ph2_["view_ensemble"]["adopted"] is False and ph2_["test_CER_final"] == 0.0
    ok("post-hoc options (validation-gated adoption of the space bonus; no adoption without evidence)")

    # ---- [ALPHA-SPLIT] / [LEX-TIE] / warm start --------------------------------------------------------------------------------
    kn = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .,")
    assert visual_base_of("é", kn) == "e" and visual_base_of(Eg, kn) == "e" and visual_base_of("ɛ̀", kn) == "e" and visual_base_of("ж", kn) is None
    uc0 = CharCodec(sorted(kn))
    uc1, gr = extend_codec(uc0, "éèà " + Eg + "ɛж́")
    assert uc1.units[:len(uc0.units)] == uc0.units and gr.orphans == [uc1.c2i["ж́"]]
    net8 = InkLinguaNet(smoke_config().model, uc0.num_classes, NUM_DESC, 4, 32).eval()
    x8, w8 = torch.rand(2, 1, 32, 96), torch.tensor([96, 80])
    with torch.no_grad():
        lp_old = net8(x8, w8)["log_probs"].numpy()
    extend_ctc_heads(net8, gr)
    with torch.no_grad():
        lp_new = net8(x8, w8)["log_probs"].numpy()
    assert lp_new.shape[-1] == uc1.num_classes and np.abs(gr.collapse(lp_new)[..., :uc0.num_classes] - lp_old).max() < 1e-4
    assert np.abs(lp_new[..., uc1.c2i["e"]] - lp_new[..., uc1.c2i["é"]]).max() < 1e-5
    Vg = gr.V

    def frames_for(text, probs_fn):
        lp = np.full((2 * len(text), Vg), math.log(1e-6))
        lp[:, 0] = math.log(0.9)
        ids, fr = [], []
        for j, ch in enumerate(text):
            i, t = uc1.c2i[ch], 2 * j
            mem = gr.variants(i)
            lp[t, :] = math.log(1e-6)
            lp[t, mem] = np.log(probs_fn(ch, mem))
            ids.append(i), fr.append(t)
        return ids, fr, lp
    lex = {"été", "élève", "école"}
    lm_fake = lambda texts: (np.array([sum(-1.0 if w_ in lex else -9.0 for w_ in t_.split()) for t_ in texts]), np.array([len(t_) for t_ in texts], float))  # noqa: E731
    tied = lambda ch, mem: np.ones(len(mem)) / len(mem) * 0.99          # noqa: E731
    for word, want in (("ete", "été"), ("eleve", "élève"), ("ecole", "école")):
        ids_, fr_, lp_ = frames_for(word, tied)
        assert resolve_variants(ids_, fr_, lp_, gr, lm_fake, beam=4)[0] == want
    ids_, fr_, lp_ = frames_for("ete", tied)
    assert resolve_variants(ids_, fr_, lp_, gr, None)[0] == "ete" and resolve_variants(ids_, fr_, lp_, gr, lm_fake, beam=1)[0] == "été"
    # warm start of a model whose alphabet EXTENDS the checkpoint's: function preserved on the old units
    tmpw = tempfile.mkdtemp()
    small = InkLinguaNet(smoke_config().model, uc0.num_classes, NUM_DESC, 4, 32).eval()
    torch.save({"weights": small.state_dict(), "units": uc0.units, "epoch": 1}, os.path.join(tmpw, "best.pt"))
    big = InkLinguaNet(smoke_config().model, uc1.num_classes, NUM_DESC, 4, 32).eval()
    info_w = warm_start(big, os.path.join(tmpw, "best.pt"), uc1)
    with torch.no_grad():
        lp_s, lp_b = small(x8, w8)["log_probs"].numpy(), big(x8, w8)["log_probs"].numpy()
    assert info_w["appended_units"] == uc1.num_classes - uc0.num_classes and np.abs(gr.collapse(lp_b)[..., :uc0.num_classes] - lp_s).max() < 1e-4
    try:
        warm_start(InkLinguaNet(smoke_config().model, uc1.num_classes, NUM_DESC, 4, 32), os.path.join(tmpw, "best.pt"), uc1, strict_alphabet=True)
        raise AssertionError("strict warm start must refuse a different alphabet")
    except RuntimeError:
        pass
    shutil.rmtree(tmpw, ignore_errors=True)
    ok("[ALPHA-SPLIT] alphabet growth preserves the function; [LEX-TIE] language-prior tie-breaking; warm start with alphabet extension")

    # ---- [XL-RC] ----------------------------------------------------------------------------------------------------------------
    rs8 = np.random.RandomState(0)
    Z0 = rs8.randn(400, 3)
    w0, au0, es0 = domain_shift(Z0[:200], Z0[200:])
    Zs, Zt = rs8.randn(400, 3), rs8.randn(400, 3) + np.array([1.2, 0.0, 0.0])
    w1, au1, es1 = domain_shift(Zs, Zt)
    assert au0 < 0.62 and au1 > 0.7 and abs(w1.mean() - 1) < 1e-9
    e8, l8 = rs8.poisson(0.05 * 40, 300).astype(float), np.full(300, 40.0)
    s8 = rs8.rand(300)
    cv1, cv0 = weighted_risk_curve(s8, e8, l8, np.ones(300)), risk_curve(s8, e8, l8, True)
    assert np.allclose(cv1["smooth"], cv0["smooth"]) and np.allclose(cv1["se"], cv0["se"])

    def xl_trial(seed, conc=1.0, n_anchor=0):
        r_ = np.random.RandomState(seed)
        xs, xt = r_.randn(500), r_.randn(900) + 1.0
        rate = lambda x_: 0.004 * np.exp(0.9 * x_)                                                  # noqa: E731
        ed_s, ed_t = r_.poisson(rate(xs) * 40).astype(float), r_.poisson(conc * rate(xt) * 40).astype(float)
        sa_ = ea_ = None
        if n_anchor:
            xa = r_.randn(n_anchor) + 1.0
            sa_, ea_ = xa, r_.poisson(conc * rate(xa) * 40).astype(float)
        res = transfer_risk_control(xs[:, None], xs, ed_s, np.full(500, 40.0), xt[:, None], xt, 0.01, 2.0, 15, 10.0, score_a=sa_, ed_a=ea_,
                                    rl_a=None if sa_ is None else np.full(n_anchor, 40.0), zero_anchor_inflation=1.0)
        th_, k_, _ = risk_control_threshold(xs, ed_s, np.full(500, 40.0), 0.01, 2.0, 15, True)
        out = {}
        for tag, (theta, k) in (("weighted", (res["theta"], res["k"])), ("plain", (th_, k_))):
            sel = xt <= theta if k > 0 else np.zeros(900, bool)
            out[tag] = (ed_t[sel].sum() / max(40.0 * sel.sum(), 1.0) if sel.any() else float("nan"), res)
        return out
    viol = {"weighted": [], "plain": []}
    for sd in range(40):
        o_ = xl_trial(sd)
        for tag in viol:
            if np.isfinite(o_[tag][0]):
                viol[tag].append(o_[tag][0] > 0.01)
    assert np.mean(viol["weighted"]) < np.mean(viol["plain"])
    f_noshift = np.median([xl_trial(sd, 1.0, 60)["weighted"][1]["factor"] for sd in range(10)])
    f_hard = np.median([xl_trial(sd, 2.5, 60)["weighted"][1]["factor"] for sd in range(10)])
    assert f_noshift < 1.35 and f_hard > 1.6, (f_noshift, f_hard)
    assert 0.0 <= weighted_ks(Zs[:, 0], Zt[:, 0]) <= 1.0 and weighted_ks(Zs[:, 0], Zs[:, 0]) == 0.0
    ok("[XL-RC] transfer-calibrated risk control (domain weights help under covariate shift, anchors detect a harder target)")

    # ---- FRAG-RC: evidence, labels, masks, character-level risk control, distillation --------------------------------------------
    Vf, path_f = 5, [0, 1, 1, 0, 2, 0, 3, 3]
    lpf = np.full((len(path_f), Vf), 0.01 / (Vf - 1))
    for t_, c_ in enumerate(path_f):
        lpf[t_, c_] = 0.99
    lpf = np.log(lpf)
    ids_f, st_f, en_f, Ff = char_evidence(lpf, lpf, lpf)
    assert ids_f == [1, 2, 3] and st_f.tolist() == [1, 4, 6] and en_f.tolist() == [3, 5, 8] and np.isfinite(Ff).all() and (Ff[:, 2] == 1.0).all()
    lp_bad = lpf.copy()
    lp_bad[4] = np.log(np.array([0.05, 0.1, 0.45, 0.3, 0.1]))
    lp_alt = lpf.copy()
    lp_alt[4] = np.log(np.array([0.05, 0.1, 0.1, 0.65, 0.1]))
    _, _, _, Fb = char_evidence(lp_bad, lp_alt, lp_bad)
    assert Fb[1, 0] < Ff[1, 0] - 2 and Fb[1, 3] == 0.0 and Fb[0, 3] == 1.0 and Fb[0, 4] < Ff[0, 4]
    assert char_wrong_labels("abcd", "abxd").tolist() == [0, 0, 1, 0] and char_wrong_labels("abcde", "abde").tolist() == [0, 1, 1, 0]
    m0 = fragment_mask(10, np.array([1, 3, 7]), np.array([3, 5, 9]), np.array([False, True, False]), guard=0)
    assert m0.tolist() == [True, True, True, False, False, False, False, True, True, True]
    m1 = fragment_mask(10, np.array([1, 3, 7]), np.array([3, 5, 9]), np.array([False, True, False]), guard=1)
    assert (not m1[2]) and m1[0] and m1[7]
    assert not fragment_mask(6, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0, bool), 1).any()
    assert cell_theta([0.1, 0.2], 7) == 0.2 and cell_theta([], 0) == float("-inf")        # clipped cell index (audit A4)
    rg2 = np.random.RandomState(0)
    nch = 4000
    rk = rg2.rand(nch)
    ln_ = rg2.randint(0, 130, nch)
    wr = (rg2.rand(nch) < 0.002 + 0.45 * rk ** 4).astype(np.float64)
    th1, k1, e1 = char_risk_threshold(rk, wr, ln_, 0.02, 2.0, 100)
    th2, k2, e2 = char_risk_threshold(rk, wr, ln_, 0.05, 2.0, 100)
    assert k1 > 100 and e1 < 0.04 and k2 >= k1 and th2 >= th1 and char_risk_threshold(rk, np.ones(nch), ln_, 0.02, 2.0, 100)[1] == 0
    lines_c = np.repeat(np.arange(100), 30)
    rk_c = rg2.rand(3000)
    y_spread = (rg2.rand(3000) < 0.02).astype(np.float64)
    y_clust = np.isin(lines_c, [3, 71]).astype(np.float64)
    assert char_risk_threshold(rk_c, y_spread, lines_c, 0.04, 2.0, 400)[1] > char_risk_threshold(rk_c, y_clust, lines_c, 0.04, 2.0, 400)[1]
    fm_ = FragModel.fit(rg2.randn(500, Ff.shape[1]), (rg2.rand(500) < 0.3).astype(np.float64))
    xx_ = rg2.randn(20, Ff.shape[1])
    assert np.allclose(fm_.risk(xx_), FragModel.from_state(fm_.state()).risk(xx_))
    gB, gT, gV = 3, 7, 5
    t_lp_ = F.log_softmax(torch.randn(gB, gT, gV), -1)
    s_raw = torch.randn(gB, gT, gV, requires_grad=True)
    lens_g, trust_g, rows_g = torch.tensor([7, 5, 7]), torch.zeros(gB, gT, dtype=torch.bool), torch.tensor([True, True, False])
    trust_g[0, 1:4] = True
    fragment_distillation(F.log_softmax(s_raw, -1), t_lp_, lens_g, trust_g, rows_g, 0.6, 0.1).backward()
    gnorm = s_raw.grad.abs().sum(-1)
    assert float(gnorm[2].sum()) == 0.0 and float(gnorm[1, :5].min()) > 0 and float(gnorm[1, 5:].sum()) == 0.0
    s_same = t_lp_.clone().requires_grad_(True)
    assert abs(float(fragment_distillation(s_same, t_lp_, lens_g, torch.zeros(gB, gT, dtype=torch.bool), rows_g, 0.6, 0.1))) < 1e-5
    assert float(fragment_distillation(s_same, t_lp_, lens_g, torch.ones(gB, gT, dtype=torch.bool), rows_g, 0.6, 0.1)) > 1e-3
    assert float(fragment_distillation(s_same, t_lp_, lens_g, torch.ones(gB, gT, dtype=torch.bool), rows_g, 1.0, 0.1)) < 1e-5
    ok("[FRAG-RC] character evidence, correctness labels, fragment masks, cluster-robust character-level risk control, distillation gradients")

    # ---- protocol, strata, MixLine, lr schedule, suite, config ---------------------------------------------------------------------
    rs10 = random.Random(11)
    texts10 = ["".join(rs10.choice("abcdefgh") for _ in range(rs10.randint(8, 14))) for _ in range(400)]
    tmp10 = tempfile.mkdtemp()
    c10 = smoke_config(tmp10)
    c10.data.labeled_fraction, c10.data.max_text_len, c10.data.calib_fraction, c10.data.val_block = 0.4, 30, 0.2, 8
    dm10 = HandwritingDataModule(c10)
    dm10._load_sources = lambda: [{"train": ListSource(texts10[:320], 1), "val": ListSource(texts10[320:372], 2), "test": ListSource(texts10[372:], 3)}]   # type: ignore
    dm10.prepare()
    nv10 = len(dm10.val_ds)
    sel10, gate10 = set(dm10.val_sel_idx.tolist()), set(dm10.val_gate_idx.tolist())
    assert not (sel10 & gate10) and (sel10 | gate10) == set(range(nv10)) and len(sel10) > 0 and len(gate10) > 0
    assert not ({i // c10.data.val_block for i in sel10} & {i // c10.data.val_block for i in gate10})
    col = Collator(dm10.codec, mix_prob=1.0, max_w=4096, gap=(2, 2), seed=1)
    lab10 = [int(i) for i in dm10.labeled_idx[:4]]
    unl10 = [int(i) for i in dm10.unlabeled_idx[:2]]
    items10 = [dm10.train_ds[i] for i in lab10 + unl10]
    b10 = col(items10)
    sep10 = " " if " " in dm10.codec.c2i else ""
    assert b10["image"].shape[0] == 4 and int(b10["width"][0]) == int(items10[0]["width"]) + 2 + int(items10[1]["width"])
    assert int(b10["target_len"][0]) == len(items10[0]["text"]) + len(sep10) + len(items10[1]["text"]) and (b10["style_ok"][:2] == 0).all()
    assert Collator(dm10.codec)(items10)["image"].shape[0] == 6
    pr10 = write_protocol(c10, dm10)
    assert pr10["val_sel_lines"] + pr10["val_gate_lines"] == nv10 and log_test_look(c10, {"path": "x.pt", "epoch": 3}, "test") == 1 and log_test_look(c10, {"path": "x.pt"}, "test") == 2
    # the training loop with a FORCED all-trusting rule: trusted fragments reach the loss; without FRAG-RC nothing does
    c12 = copy.deepcopy(c10)
    c12.train.aux_start_epoch, c12.train.aux_ramp_epochs = 0, 1
    dev12 = torch.device("cpu")
    trn12 = Trainer(c12, dm10, InkLinguaNet(c12.model, dm10.codec.num_classes, NUM_DESC, dm10.n_clusters, c12.data.img_height), LLMEngine(c12.llm, dev12), dev12)
    nf12 = Ff.shape[1]
    trn12._frag_set_state({"model": FragModel(np.zeros(nf12), np.ones(nf12), np.zeros(nf12 + 1)).state(), "theta": [1.0] * dm10.n_strata, "alpha": 0.1, "epoch": 0, "n_strata": dm10.n_strata})
    trn12.global_step = 2 * trn12.steps_per_epoch
    tr12 = trn12._train_epoch(0)
    assert tr12.get("frag_trust", 0.0) > 0.0 and tr12["frag"] > 0 and np.isfinite(tr12["total"])
    c13 = copy.deepcopy(c12)
    c13.train.frag.enabled = False
    trn13 = Trainer(c13, dm10, InkLinguaNet(c13.model, dm10.codec.num_classes, NUM_DESC, dm10.n_clusters, c13.data.img_height), LLMEngine(c13.llm, dev12), dev12)
    trn13.global_step = 2 * trn13.steps_per_epoch
    tr13 = trn13._train_epoch(0)
    assert tr13["frag"] == 0.0 and "frag_trust" not in tr13
    shutil.rmtree(tmp10, ignore_errors=True)
    tr_ = object.__new__(Trainer)
    tr_.tc, tr_.steps_per_epoch, tr_.total_steps = Config().train, 10, 1000
    tr_.cycles = [{"start": 0, "end": 1000, "peak": tr_.tc.lr, "warm": tr_.tc.warmup_steps, "lo": 0.0}]
    tr_.global_step = 1000
    lr_end = tr_._lr(999)
    tr_.total_steps = 5000
    tr_._extend_schedule(1000)
    assert tr_._lr(1000) < 2 * lr_end + 1e-6 and tr_._lr(4999) < 0.01 * tr_.tc.lr
    b_ = Config()
    lo_ = Config.make_variant("labeled_only", b_)
    assert not lo_.train.pseudo.enabled and not lo_.train.frag.enabled and lo_.data.calib_fraction == 0.0 and lo_.train.mixline_prob == b_.train.mixline_prob
    assert Config.make_variant("legacy_pl", b_).train.pseudo.enabled and not Config.make_variant("no_frag", b_).train.frag.enabled
    assert Config.make_variant("no_strata", b_).train.frag.strata == 1 and Config.make_variant("ctc_only", b_).model.film is False
    for kind in Config.ABLATIONS:
        Config.ablation(kind)
    assert all(v in Config.ABLATIONS or v in ("labeled_only", "full", "full_supervised") for p in SUITE_PRESETS.values() for v in p)
    assert Config.from_dict(json.loads(json.dumps(Config().to_dict()))) == Config()
    cm_ = Config.multi(("iam", "khatt"))
    assert Config.from_dict(json.loads(json.dumps(cm_.to_dict()))).data.sources == ("iam", "khatt")
    cc_ = Config()
    cc_.data.custom_sources = (DatasetSpec(key="x", hf_dataset="a/b", rtl=True),)
    assert Config.from_dict(json.loads(json.dumps(cc_.to_dict()))).data.custom_sources[0].rtl is True
    tmp3 = tempfile.mkdtemp()
    rs3, paths3 = np.random.RandomState(3), {"labeled_only": {}, "full": {}, "no_frag": {}}
    for s_ in (1, 2):
        base_e = rs3.binomial(40, 0.10, 300)
        for v_, e_ in (("labeled_only", base_e), ("full", np.maximum(base_e - rs3.binomial(1, 0.25, 300), 0)), ("no_frag", np.maximum(base_e - rs3.binomial(1, 0.1, 300), 0))):
            d_ = os.path.join(tmp3, f"{v_}_{s_}", "01_recognition")
            os.makedirs(d_)
            with open(os.path.join(d_, "predictions.csv"), "w", newline="", encoding="utf-8") as f_:
                w_ = csv.writer(f_)
                w_.writerow(["line_index", "edits", "ref_len", "CER_posthoc"])
                for i_, ee in enumerate(e_):
                    w_.writerow([i_, int(ee), 40, f"{ee / 40:.4f}"])
            paths3[v_][s_] = os.path.join(d_, "predictions.csv")
    sm_ = summarise_suite(paths3, "labeled_only", 200)
    pv_ = sm_["paired_vs_baseline"]["full"]
    assert pv_["mean_delta"] < 0 and pv_["seeds_better"] == 2 and sm_["contribution_vs_full"]["no_frag"]["mean_delta"] > 0 and "| full |" in suite_markdown(sm_)
    shutil.rmtree(tmp3, ignore_errors=True)
    # explainability helpers
    np.random.RandomState(0)
    cc2 = np.zeros((60, 30))
    for k in range(60):
        cc2[k, k % 30] = 40
    assert code_char_stats(cc2, 20)["component_code_purity"] == 1.0
    shared = np.zeros((3, 40))
    shared[:, [5, 20]] = 0.4
    for r, f in enumerate((8, 15, 30)):
        shared[r, f] = 0.2
    res, sink = contrast_rollout(shared)
    assert 0.8 < sink < 0.9 and [int(np.argmax(res[r])) for r in range(3)] == [8, 15, 30]
    ok("protocol split, MixLine, FRAG-RC in the training loop, lr schedule on resume, suite aggregation, config round trips, explainability helpers")
    # ---- v11 experiment plan: budget rule, plan, statistics, block caps, decontamination, parquet reader ----------------------------
    from .datasets_registry import DATASETS
    from .experiments import PlanConfig, build_runs, run_config
    from .paper import holm, pooled_paired, sign_test
    from .schedule import apply_budget, budget, complexity_factor
    bi = budget(6482, complexity_factor([DATASETS["iam"]], [6482]), 16)
    d0 = Config()
    assert (bi["epochs"], bi["patience"], bi["min_epochs"], bi["frag_warmup"], bi["frag_refresh"]) == (d0.train.epochs, d0.train.patience, d0.train.min_epochs,
                                                                                                    d0.train.frag.warmup_epochs, d0.train.frag.refresh_every)
    assert budget(2328, 1.15, 16)["epochs"] == 150 and budget(60000, 1.15, 16)["epochs"] < 50 and complexity_factor([DATASETS["casia"]], [1]) == 1.8
    tmpp = tempfile.mkdtemp()
    pl = PlanConfig(work_dir=tmpp, cache_dir=tmpp)
    rr = build_runs(pl)
    ids_ = [r["id"] for r in rr]
    assert len(ids_) == len(set(ids_)) and ids_[0] == "iam/labeled_only_lf030_s1337" and sum(r["profile"] == "full" for r in rr) == len(pl.corpora) + 1
    assert all(r["stage"] != "newseye_de" for r in rr) and any(r["kind"] == "adapt" and r["deps"] == ["iam/full_lf030_s1337"] for r in rr)
    stages_in_order = list(dict.fromkeys(r["stage"] for r in rr if r["kind"] == "run"))
    assert stages_in_order == list(pl.corpora) + ["joint"], stages_in_order                       # corpus after corpus
    cv = run_config(pl, next(r for r in rr if r["variant"] == "no_vq"))
    apply_budget(cv, bi)
    assert cv.train.vq_start_epoch == 10 ** 6 and cv.llm.backend == "causal" and cv.data.source == "hf"
    assert all(run_config(pl, r).llm.backend == "causal" for r in rr[:5])
    shutil.rmtree(tmpp, ignore_errors=True)
    rs_ = np.random.RandomState(0)
    prs = []
    for _ in range(3):
        eb_ = rs_.poisson(4, 600).astype(float)
        prs.append((np.maximum(eb_ - rs_.binomial(1, .3, 600), 0), eb_, np.full(600, 40.0), np.arange(600) // 20))
    tp = pooled_paired(prs, 800)
    assert tp["delta"] < 0 and tp["hi"] < 0 and tp["p"] < 0.01 and pooled_paired([(b, b, l, g) for _, b, l, g in prs], 400)["p"] > 0.5
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]) and abs(sign_test([-1] * 9 + [1])["p"] - 0.0215) < 1e-3
    sb = HandwritingDataModule._select(1000, 205, 7, 20)
    assert len(sb) == 205 and all(((sb // 20) == b).sum() in (20, 5) for b in set(sb // 20)) and np.array_equal(sb, HandwritingDataModule._select(1000, 205, 7, 20))
    tmpd = tempfile.mkdtemp()
    tx = [f"word{i} text" for i in range(40)]
    for deco in (False, True):
        cd_ = smoke_config(tmpd)
        cd_.data.labeled_fraction, cd_.data.decontaminate = 0.5, deco
        dmd = HandwritingDataModule(cd_)
        dmd._load_sources = lambda: [{"train": ListSource(tx, 1), "val": ListSource(tx[20:28], 9), "test": ListSource(tx[:8], 1)}]   # type: ignore
        if not deco:
            try:
                dmd.prepare()
                raise AssertionError("train/test image overlap must be an error without decontamination")
            except RuntimeError:
                pass
        else:
            dmd.prepare()
            assert len(dmd.train_ds) == 32 and dmd.provenance["decontamination"]["removed_from_train"] == 8
    import io as _io
    import pyarrow as pa
    import pyarrow.parquet as pq
    from .data import ParquetSource
    rws = []
    for k in range(25):
        bb = _io.BytesIO()
        Image.fromarray((np.random.RandomState(k).rand(20, 40 + k) * 255).astype(np.uint8)).save(bb, format="PNG")
        rws.append({"image": {"bytes": bb.getvalue(), "path": None}, "text": f"line {k}"})
    pq.write_table(pa.Table.from_pylist(rws), os.path.join(tmpd, "train.parquet"), row_group_size=7)
    ps_ = ParquetSource([os.path.join(tmpd, "train.parquet")])
    assert len(ps_) == 25 and ps_.text(13) == "line 13" and ps_.image(13).size == (53, 20) and ps_.image(24).size == (64, 20)
    # memory-mapped line cache: built once (streamed to disk), re-read without decoding, subsets / concatenations copy no pixels,
    # pickling (what a data-loader worker receives) carries no pixel data
    import pickle as _pk
    from .data import InkStore, _ink_concat, _ink_subset, build_line_cache, cached_line_cache
    lcd, ckey_ = os.path.join(tmpd, "lines"), {"corpus": "t", "split": "train", "ink": True, "v": 2}
    sel_ = np.arange(25)
    c_a = cached_line_cache(lcd, ckey_, lambda p: build_line_cache(ps_, sel_, 32, 400, 16, 4, "t", True, p))
    c_b = cached_line_cache(lcd, ckey_, lambda p: (_ for _ in ()).throw(AssertionError("a cached entry must not be rebuilt")))
    ref_ = build_line_cache(ps_, sel_, 32, 400, 16, 4, "t", True, None)["ink"]
    assert isinstance(c_b["ink"], InkStore) and all(np.array_equal(c_b["ink"][i], ref_[i]) for i in range(25))
    assert c_a["digest"] == c_b["digest"] and np.array_equal(c_a["width"], c_b["width"])
    mix_ = _ink_subset(_ink_concat([c_b["ink"], InkStore.from_arrays([ref_[3]])]), np.array([25, 7]))
    assert np.array_equal(mix_[0], ref_[3]) and np.array_equal(mix_[1], ref_[7]) and len(_pk.dumps(c_b["ink"])) < 4096
    # one DataLoader iterator per epoch (tqdm.auto's asyncio class used to start a second one, i.e. twice the workers)
    from . import trainer as _tr
    _n_it = {"n": 0}

    class _CountIt:
        def __iter__(self):
            _n_it["n"] += 1
            return iter(range(3))

        def __len__(self):
            return 3
    assert list(_tr.tqdm(_CountIt(), disable=True)) == [0, 1, 2] and _n_it["n"] == 1
    shutil.rmtree(tmpd, ignore_errors=True)
    ok("experiment plan: budget rule (IAM = v10 defaults), corpus-major plan, pooled paired bootstrap / Holm / sign test, block caps, decontamination, parquet reader, memory-mapped line cache, one loader iterator")
    _test_exact_resume()
    ok("exact step-level resume: a run stopped in the middle of an epoch (SIGTERM) and restarted ends with the same weights, EMA and "
       "history as an uninterrupted run; ink-band crop of mostly blank line images")
    LOG.info("%d self-tests passed in %.1fs", len(done), time.time() - t0)


def _test_exact_resume() -> None:
    import signal as _signal
    from .llm import LLMEngine
    from .model import InkLinguaNet
    from .trainer import Trainer
    from .utils import restore_logging, set_seed, setup_logging, torch_load
    # ink band: a 4 % text band in a blank page is found, a tight line crop is left alone
    page = np.full((1000, 800), 250.0, np.float32)
    page[100:140, 50:750] = 20.0
    page[700:703, 300:304] = 20.0                                              # a speck far from the text
    from .data import ink_band
    band = ink_band(page)
    assert band is not None and band[0] <= 100 and 140 <= band[1] < 200, band
    assert ink_band(page[90:150]) is None
    tmp = tempfile.mkdtemp(prefix="inklingua_resume_")
    try:
        def make(name):
            c = smoke_config(tmp)
            c.run.name = name
            c.data.synthetic_sizes = (160, 24, 24)
            c.train.epochs, c.train.min_epochs, c.train.patience = 3, 0, 0
            c.train.frag.warmup_epochs, c.train.frag.refresh_every, c.train.frag.min_chars = 1, 1, 30
            c.llm.backend = "dummy"
            c.make_dirs()
            setup_logging(c)
            set_seed(c.run.seed)
            d = HandwritingDataModule(c).prepare()
            m = InkLinguaNet(c.model, d.codec.num_classes, NUM_DESC, d.n_clusters, c.data.img_height)
            return c, Trainer(c, d, m, LLMEngine(c.llm, torch.device("cpu")), torch.device("cpu"))
        try:
            cA, tA = make("A")
            tA.fit()
            A = torch_load(os.path.join(cA.ckpt_dir, "last.pt"))
            cB, tB = make("B")
            orig, n = Trainer._train_step, {"k": 0}

            def stepper(self, *a, **k):
                orig(self, *a, **k)
                n["k"] += 1
                if n["k"] == 13:                                                   # epoch 2, step 3 of 10
                    self._stop_signal = int(_signal.SIGTERM)
            Trainer._train_step = stepper
            try:
                tB.fit()
            finally:
                Trainer._train_step = orig
            part = torch_load(os.path.join(cB.ckpt_dir, "last.pt"))["partial"]
            assert tB.interrupted and (part["epoch"], part["step"]) == (1, 3), part
            cB, tB = make("B")
            tB.fit()
            B = torch_load(os.path.join(cB.ckpt_dir, "last.pt"))
        finally:
            for _ in range(3):
                restore_logging()
        assert A["global_step"] == B["global_step"] and "partial" not in B
        assert all(torch.equal(A["model"][k], B["model"][k]) for k in A["model"]), "resumed weights differ"
        assert all(torch.equal(A["ema"][k], B["ema"][k]) for k in A["ema"]), "resumed EMA differs"
        assert A["history"]["train_total"] == B["history"]["train_total"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_smoke(work_dir: str = None, keep: bool = False) -> Dict[str, Any]:
    """The synthetic end-to-end smoke: train (FRAG-RC refresh + MixLine + val_sel), resume, evaluate (val_gate, WSD, interpretability), report.
    Returns the metrics; the directory is deleted unless `keep`."""
    from .main import main
    tmp = work_dir or tempfile.mkdtemp(prefix="inklingua_smoke_")
    c = smoke_config(tmp)
    c.train.epochs, c.train.frag.warmup_epochs, c.train.frag.refresh_every = 12, 4, 4
    c.train.aux_start_epoch, c.train.aux_ramp_epochs, c.train.vq_start_epoch, c.train.vq_anneal_epochs = 2, 2, 2, 2
    c.data.synthetic_sizes = (480, 64, 48)
    c.infer.posthoc, c.infer.wsd, c.infer.n_vis, c.infer.n_translate = True, True, 1, 1
    c.infer.pl_diag_samples = 20
    try:
        m = main(c)
        assert m is not None and m["SYNTHETIC_SMOKE_TEST_NOT_FOR_PUBLICATION"] is True and m["val_sel_lines"] > 0 and m["val_gate_lines"] > 0
        hist = json.load(open(os.path.join(c.curve_dir, "history.json")))
        assert "val_sel_cer" in hist and "val_gate_cer" in hist and "train_frag" in hist and any(k.startswith("frag_theta_s") for k in hist), "FRAG-RC never calibrated"
        wsd = (m.get("posthoc") or {}).get("wsd")
        assert wsd is not None and "adopted" in wsd
        for rel in ("00_summary/metrics.json", "01_recognition/predictions.csv", "01_recognition/alignments.jsonl", "04_posthoc/posthoc.json", "12_report/index.html",
                    "10_explanations/layer_probes.csv", "02_uncertainty/uncertainty_signals.csv", "03_error_structure/groups.csv", "07_robustness/longline_stress.json"):
            assert os.path.exists(os.path.join(c.infer_dir, rel)), rel
        assert os.path.exists(os.path.join(c.run_dir, "protocol", "protocol.json"))
        # resume: a second call with more epochs continues from the checkpoint (warm restart) instead of starting over
        c.train.epochs, c.run.do_infer = 14, False
        main(c)
        hist2 = json.load(open(os.path.join(c.curve_dir, "history.json")))
        assert len(hist2["epoch"]) == 14 and hist2["epoch"][:12] == hist["epoch"][:12]
        LOG.info("smoke test passed (%s)", tmp)
        return m
    finally:
        restore_logging()
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
