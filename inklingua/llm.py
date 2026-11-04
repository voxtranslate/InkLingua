"""The pretrained LLM: uncertainty-aware translation, LM scoring for the constrained N-best rescoring, calibrated [?] marks."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import difflib
import numpy as np
import torch

from .config import LLMConfig
from .riskcontrol import auroc
from .text import chrf, detokenize
from .utils import LOG

LANG_NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "it": "Italian", "pt": "Portuguese", "ru": "Russian",
              "zh": "Chinese", "ar": "Arabic", "hi": "Hindi", "ja": "Japanese", "nl": "Dutch", "yo": "Yoruba", "ha": "Hausa", "ln": "Lingala",
              "sw": "Swahili", "wo": "Wolof", "bm": "Bambara", "ff": "Fula", "am": "Amharic", "ig": "Igbo", "zu": "Zulu", "tr": "Turkish",
              "pl": "Polish", "ro": "Romanian", "cs": "Czech", "sv": "Swedish", "vi": "Vietnamese", "id": "Indonesian", "fa": "Persian",
              "ur": "Urdu", "ko": "Korean", "el": "Greek", "uk": "Ukrainian", "ca": "Catalan", "la": "Latin", "nb": "Norwegian", "no": "Norwegian",
              "da": "Danish", "et": "Estonian", "he": "Hebrew", "th": "Thai", "bn": "Bengali"}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code, code)


FEWSHOT = {
    "fr": ("Le livre est sur la table près de la fenêtre.", "Nous nous retrouverons à la gare demain matin."),
    "es": ("El libro está sobre la mesa junto a la ventana.", "Nos encontraremos en la estación mañana por la mañana."),
    "de": ("Das Buch liegt auf dem Tisch neben dem Fenster.", "Wir treffen uns morgen früh am Bahnhof."),
    "it": ("Il libro è sul tavolo vicino alla finestra.", "Ci incontreremo alla stazione domani mattina."),
    "pt": ("O livro está sobre a mesa perto da janela.", "Encontrar-nos-emos na estação amanhã de manhã."),
    "hi": ("किताब खिड़की के पास मेज़ पर है।", "हम कल सुबह स्टेशन पर मिलेंगे।"),
}


@dataclass
class Hypothesis:
    """One recognition hypothesis (LOGICAL order, detokenised by the caller when the corpus needs it) = the 'pseudo-label
    information' handed to the LLM."""
    text: str
    char_conf: List[float] = field(default_factory=list)
    logprob: float = 0.0
    post: float = 1.0


def _word_conf(vals: Sequence[float], agg: str) -> float:
    return float(np.min(vals)) if agg == "min" else float(np.mean(vals))


def mark_uncertain(text: str, char_conf: Sequence[float], thresh: float, agg: str = "mean") -> str:
    """Append [?] to words whose character confidence (mean or minimum over the word) is below `thresh`."""
    if len(char_conf) != len(text):
        return text
    words, cur, curc = [], [], []

    def flush():
        if cur:
            words.append("".join(cur) + ("[?]" if _word_conf(curc, agg) < thresh else ""))
            cur.clear()
            curc.clear()

    for ch, c in zip(text, char_conf):
        if ch == " ":
            flush()
        else:
            cur.append(ch)
            curc.append(c)
    flush()
    return " ".join(words)


def word_marking_data(pred: str, gt: str, char_conf: Sequence[float], agg: str) -> Tuple[List[float], List[bool]]:
    """(confidence, is_wrong) of every predicted word of a line."""
    if len(char_conf) != len(pred):
        return [], []
    scores, cur = [], []
    for ch, c in zip(pred, char_conf):
        if ch == " ":
            if cur:
                scores.append(_word_conf(cur, agg))
                cur = []
        else:
            cur.append(c)
    if cur:
        scores.append(_word_conf(cur, agg))
    pw, gw = pred.split(), gt.split()
    right = [False] * len(pw)
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, gw, pw, autojunk=False).get_opcodes():
        if op == "equal":
            for j in range(j1, j2):
                right[j] = True
    return (scores, [not x for x in right]) if len(scores) == len(pw) else ([], [])


def calibrate_word_mark(preds: Sequence[str], gts: Sequence[str], confs: Sequence[Sequence[float]], precision: float = 0.5,
                        min_share: float = 0.02) -> Dict[str, Any]:
    """[CAL-MARK] threshold / aggregation of the [?] marks: on held-out words the score (mean or min character confidence, whichever
    ranks the wrong words better) is thresholded at the LARGEST value for which the flagged words are wrong with probability >= `precision`."""
    best: Optional[Dict[str, Any]] = None
    for agg in ("mean", "min"):
        sc, wr = [], []
        for p, g, c in zip(preds, gts, confs):
            a, b = word_marking_data(p, g, c, agg)
            sc += a
            wr += b
        s_, w_ = np.asarray(sc, np.float64), np.asarray(wr, bool)
        if len(s_) < 50 or w_.sum() < 20 or (~w_).sum() < 20:
            continue
        au = auroc(1.0 - s_, w_)
        if best is None or au > best["auroc"]:
            best = {"agg": agg, "auroc": au, "s": s_, "w": w_}
    if best is None:
        return {"threshold": None, "reason": "too few validation words"}
    s_, w_ = best["s"], best["w"]
    thr, stats = None, {}
    for t in np.unique(np.quantile(s_, np.linspace(0.01, 0.9, 90))):
        m = s_ < t
        if m.mean() < min_share:
            continue
        prec = float(w_[m].mean())
        if prec >= precision:
            thr, stats = float(t), {"precision": prec, "recall": float(w_[m].sum() / w_.sum()), "share_marked": float(m.mean())}
    out = {"threshold": thr, "agg": best["agg"], "auroc": float(best["auroc"]), "word_error_rate": float(w_.mean()), "n_words": int(len(s_)),
           "target_precision": precision}
    out.update(stats)
    return out


def mbr_select(cands: List[str], weights: Sequence[float]) -> int:
    """Minimum-Bayes-risk choice with chrF utility, weighted by the recogniser posterior."""
    if len(cands) == 1:
        return 0
    w = np.asarray(weights, np.float64)
    w = w / max(w.sum(), 1e-9)
    util = [sum(w[j] * chrf(ci, cj) for j, cj in enumerate(cands)) for ci in cands]
    return int(np.argmax(util))


def fuse_scores(ctc_lp: np.ndarray, lm_lp: np.ndarray, ntok: np.ndarray, beta: float, rho: float = 0.0) -> np.ndarray:
    """Shallow fusion over the SAME N-best list:  score = log P_ctc(h) + beta * (log P_lm(h) + rho * #tokens(h)).  Only hypotheses of
    the recogniser can win, so the LLM cannot invent text."""
    return np.asarray(ctc_lp, np.float64) + beta * (np.asarray(lm_lp, np.float64) + rho * np.asarray(ntok, np.float64))


class LLMEngine:
    """Lazy wrapper around a pretrained model.  backend 'causal': instruction-tuned decoder LLM (prompted; also gives LM scores);
    'seq2seq': Marian / NLLB-style NMT model (N-best translation + MBR); 'dummy': no model, for offline plumbing tests."""

    def __init__(self, cfg: LLMConfig, device: torch.device):
        self.cfg, self.device = cfg, device
        self.tok = self.model = None
        self._failed = False

    def available(self) -> bool:
        if self.cfg.backend == "dummy":
            return True
        if self.model is None and not self._failed:
            self._load()
        return self.model is not None

    def can_score(self) -> bool:
        return self.cfg.backend == "causal" and self.available()

    def _load(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
            c = self.cfg
            if c.backend == "causal":
                name, cls = c.model_name, AutoModelForCausalLM
            else:
                name = c.seq2seq_models.get(f"{c.source_language}-{c.target_language}") or (
                    c.seq2seq_models[c.target_language] if c.source_language == "en" else None)
                if name is None:
                    raise KeyError(f"no seq2seq model configured for {c.source_language}->{c.target_language}: add the key "
                                   f"'{c.source_language}-{c.target_language}' to cfg.llm.seq2seq_models, or use backend='causal'")
                cls = AutoModelForSeq2SeqLM
            dtype = torch.float16 if self.device.type == "cuda" else torch.float32
            LOG.info("Loading pretrained %s model '%s' (auto-download on first use) ...", c.backend, name)
            self.tok = AutoTokenizer.from_pretrained(name)
            try:
                model = cls.from_pretrained(name, dtype=dtype)
            except TypeError:
                model = cls.from_pretrained(name, torch_dtype=dtype)
            self.model = model.to(self.device).eval()
            if self.tok.pad_token is None:
                self.tok.pad_token = self.tok.eos_token
        except Exception as e:
            self._failed = True
            LOG.warning("LLM could not be loaded (%s). Translation / LLM rescoring will be skipped.", e)

    @torch.no_grad()
    def score_texts(self, texts: List[str], language: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Sum of token log-probs and #tokens of every text under the causal LM, conditioned on the written language (default:
        cfg.source_language; a multi-corpus run passes each line's own language).  Cached per (language, text), batched by length."""
        assert self.can_score()
        lang = language or self.cfg.source_language
        cache = self.__dict__.setdefault("_lm_cache", {}).setdefault(lang, {})
        todo = sorted({t for t in texts if t not in cache}, key=len)
        prefix = f"Transcription of a line of {lang_name(lang)} handwriting:\n"
        pre = self.tok(prefix, add_special_tokens=False).input_ids
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        for i in range(0, len(todo), self.cfg.score_batch_size):
            chunk = todo[i:i + self.cfg.score_batch_size]
            seqs = [pre + self.tok(t, add_special_tokens=False).input_ids for t in chunk]
            L = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), L), pad, dtype=torch.long)
            att = torch.zeros((len(seqs), L), dtype=torch.long)
            for k, s in enumerate(seqs):
                ids[k, :len(s)], att[k, :len(s)] = torch.tensor(s), 1
            ids, att = ids.to(self.device), att.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=att).logits.float()
            tok_lp = logits[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) - logits[:, :-1].logsumexp(-1)
            for k, s in enumerate(seqs):
                a, b = len(pre) - 1, len(s) - 1
                cache[chunk[k]] = (float(tok_lp[k, a:b].sum()), b - a)
        return (np.array([cache[t][0] for t in texts], np.float64), np.array([cache[t][1] for t in texts], np.float64))

    def for_target(self, lang: str) -> "LLMEngine":
        if lang == self.cfg.target_language:
            return self
        sib = self.__dict__.setdefault("_siblings", {})
        if lang not in sib:
            sib[lang] = LLMEngine(dataclasses.replace(self.cfg, target_language=lang), self.device)
        return sib[lang]

    def _messages(self, hyps: List[Hypothesis], uncertain: bool, c: Optional[LLMConfig] = None, detok: bool = False
                  ) -> Tuple[List[Dict[str, str]], str]:
        c = c or self.cfg
        lang, src = lang_name(c.target_language), lang_name(c.source_language)
        top = hyps[0]
        marked = mark_uncertain(top.text, top.char_conf, c.word_conf_thresh, c.word_conf_agg) if uncertain else top.text
        fix = detokenize if detok else (lambda s: s)
        marked, clean_top = fix(marked), fix(top.text)
        alts: List[str] = []
        if uncertain:
            for h in hyps[1:]:
                a = fix(h.text)
                if a != clean_top and a not in alts:
                    alts.append(a)
        user = f"Transcription: {marked}\n"
        if alts:
            user += "Other possible readings of the same line:\n" + "\n".join(f"- {a}" for a in alts) + "\n"
        user += f"Translate into {lang}. Output only the translation."
        sys_msg = ("You are an expert translator. The input was produced by automatic handwriting recognition and may contain recognition errors."
                   + (" Words followed by [?] are uncertain; use the alternative readings and the context to decide what was most likely written, "
                      "silently correct obvious errors, and never output [?]." if uncertain else "")
                   + f" Translate the intended {src} sentence into {lang}.")
        msgs = [{"role": "system", "content": sys_msg}]
        if c.target_language in FEWSHOT and c.source_language == "en":
            t1, t2 = FEWSHOT[c.target_language]
            if uncertain:
                ex1 = ("Transcription: The book is on the tabel[?] near the window.\nOther possible readings of the same line:\n"
                       f"- The book is on the table near the window.\nTranslate into {lang}. Output only the translation.")
            else:
                ex1 = f"Transcription: The book is on the table near the window.\nTranslate into {lang}. Output only the translation."
            ex2 = f"Transcription: We will meet at the station tomorrow morning.\nTranslate into {lang}. Output only the translation."
            msgs += [{"role": "user", "content": ex1}, {"role": "assistant", "content": t1},
                     {"role": "user", "content": ex2}, {"role": "assistant", "content": t2}]
        msgs.append({"role": "user", "content": user})
        return msgs, marked

    @staticmethod
    def _clean(out: str) -> str:
        out = out.strip().replace("[?]", "")
        lines = [l.strip() for l in out.split("\n") if l.strip()]
        out = lines[0] if lines else ""
        return out.strip().strip('"“”').strip()

    @torch.no_grad()
    def translate(self, hyps: List[Hypothesis], uncertain: Optional[bool] = None, target_language: Optional[str] = None,
                  detok: bool = False, source_language: Optional[str] = None) -> Dict[str, Any]:
        """hyps[0] is the best hypothesis (logical order).  uncertain=False -> plain top-1 translation.  detok=True re-attaches IAM
        tokenisation before prompting (only for corpora whose spec says so)."""
        c = self.cfg
        if source_language and source_language != c.source_language:
            if c.backend == "seq2seq":
                eng = self.__dict__.setdefault("_by_source", {}).setdefault(source_language, LLMEngine(dataclasses.replace(c, source_language=source_language), self.device))
                return eng.translate(hyps, uncertain, target_language, detok)
            c = dataclasses.replace(c, source_language=source_language)
        if target_language is not None and target_language != c.target_language:
            if c.backend == "seq2seq":
                return self.for_target(target_language).translate(hyps, uncertain, detok=detok)
            c = dataclasses.replace(c, target_language=target_language)
        uncertain = c.use_uncertainty_prompt if uncertain is None else uncertain
        if not self.available():
            return {"translation": "", "marked_text": "", "candidates": []}
        fix = detokenize if detok else (lambda s: s)
        if c.backend == "dummy":
            return {"translation": f"[{c.target_language}] {fix(hyps[0].text)}", "marked_text": fix(hyps[0].text), "candidates": []}
        if c.backend == "causal":
            msgs, marked = self._messages(hyps, uncertain, c, detok)
            prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**inp, max_new_tokens=c.max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id)
            txt = self._clean(self.tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True))
            return {"translation": txt, "marked_text": marked, "candidates": [txt]}
        use = hyps[:c.nbest] if uncertain else hyps[:1]
        srcs = [fix(h.text) for h in use]
        inp = self.tok(srcs, return_tensors="pt", padding=True, truncation=True, max_length=128).to(self.device)
        out = self.model.generate(**inp, num_beams=4, max_new_tokens=c.max_new_tokens)
        cands = [t.strip() for t in self.tok.batch_decode(out, skip_special_tokens=True)]
        best = mbr_select(cands, [h.post for h in use])
        return {"translation": cands[best], "marked_text": srcs[0], "candidates": cands}
