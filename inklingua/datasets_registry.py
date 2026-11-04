"""
Registry of FREE line-level handwriting corpora that share the Hugging Face layout of Teklia/IAM-line (an `image` column and a
`text` column, parquet files, train / validation / test splits).  Every entry was checked against the Hub API on 2026-11-04
(commit hash, columns, split sizes, licence tag); the pinned revision keeps a run reproducible.

Why: IAM has 6,482 training lines, all English, all Latin script.  The registry makes it a one-line change to train on more
data (`Config.multi(("iam", "rimes"))`), on another language, or on another script (Arabic: KHATT, Han: CASIA-HWDB2), and
every per-domain table of the evaluation then shows what each corpus contributes.

Licences: the Hub tags are MIT for the Teklia conversions; the UNDERLYING corpora have their own terms (IAM: non-commercial
research; CASIA: free for academic research; KHATT: free research use; RIMES: research) -- check them before publishing.
Non-Teklia corpora (not in the registry) that are worth knowing: Riksarkivet (Swedish, cc-by), aarhus-city-archives
historical-danish-handwriting (cc-by-4.0), Rahvusarhiiv/et_handwriting_recognition (Estonian, cc0), fhswf/german_handwriting
(afl-3.0), SSamDav/portuguese-handwriting-16th-19th (cc-by-4.0); their columns differ, so they need a custom DatasetSpec.
"""
from __future__ import annotations

from typing import Dict

from .config import IAM_LINE_REVISION, IAM_LINE_SIZES, DatasetSpec

DATASETS: Dict[str, DatasetSpec] = {
    "iam": DatasetSpec(
        key="iam", hf_dataset="Teklia/IAM-line", hf_revision=IAM_LINE_REVISION, language="en", script="Latin", detokenize=True,
        expected_sizes=IAM_LINE_SIZES, licence="Hub: MIT; IAM itself: non-commercial research",
        note="Modern English, 657 writers (no writer id in this conversion); the Aachen writer-independent split."),
    "rimes": DatasetSpec(
        key="rimes", hf_dataset="Teklia/RIMES-2011-line", hf_revision="ba3e6b5573094208b30a134e32d9b65dab18e74e", language="fr",
        script="Latin", expected_sizes=(10188, 1138, 778), licence="Hub: MIT; RIMES: research",
        note="Modern French letters (RIMES 2011 line level)."),
    "belfort": DatasetSpec(
        key="belfort", era="historical", hf_dataset="Teklia/Belfort-line", hf_revision="1e1caa665d48aa6756313cc0f72416f628de34c3", language="fr",
        script="Latin", expected_sizes=(25800, 3102, 3819), licence="Hub: MIT",
        note="Historical French municipal registers (Belfort), 19th-20th c."),
    "esposalles": DatasetSpec(
        key="esposalles", era="historical", hf_dataset="Teklia/Esposalles-line", hf_revision="5f8e7ceccb126528ebae15cb81c475403d176ee4", language="ca",
        script="Latin", expected_sizes=(2328, 742, 757), licence="Hub: MIT", note="17th-century Catalan marriage licences."),
    "himanis": DatasetSpec(
        key="himanis", era="historical", hf_dataset="Teklia/Himanis-line", hf_revision="3793c910e6c087ebc1d575ab7384d8d60ad7eafd", language="la",
        script="Latin", expected_sizes=(18504, 2367, 2241), licence="Hub: MIT", note="Medieval Latin / French chancery registers."),
    "home_alcar": DatasetSpec(
        key="home_alcar", era="historical", hf_dataset="Teklia/HOME-Alcar-line", hf_revision="edb8b023ebd6c0c92ad61adcdd550cfee6e34a18", language="la",
        script="Latin", expected_sizes=(59969, 7905, 6932), licence="Hub: MIT", note="Medieval Latin cartularies (HOME-Alcar)."),
    "newseye_de": DatasetSpec(
        key="newseye_de", era="historical", hf_dataset="Teklia/NewsEye-Austrian-line", hf_revision="8d6b94c016b7cb455ad77c7ac67f80036ebfb752",
        language="de", script="Latin", test_split=None, expected_sizes=(51588, 4379), licence="Hub: MIT",
        note="Austrian German; NO test split on the Hub -> contributes train / validation lines only."),
    "norhand1": DatasetSpec(
        key="norhand1", era="historical", hf_dataset="Teklia/NorHand-v1-line", hf_revision="2725c76fe2bd2c0f100c641621e21f4887cbb26c", language="nb",
        script="Latin", expected_sizes=(19653, 2286, 1793), licence="Hub: MIT", note="Norwegian letters and diaries, 19th-20th c."),
    "norhand3": DatasetSpec(
        key="norhand3", era="historical", hf_dataset="Teklia/NorHand-v3-line", hf_revision="113280c8fbb7f6df7923ee15a7ba3c97710a8f1c", language="nb",
        script="Latin", expected_sizes=(222381, 22679, 1562), licence="Hub: MIT",
        note="Large Norwegian corpus (222k train lines): use max_lines to cap it."),
    "popp": DatasetSpec(
        key="popp", era="historical", hf_dataset="Teklia/POPP-line", hf_revision="7ab15c7589c164fc13cb42adccf68e61c05b2945", language="fr",
        script="Latin", expected_sizes=(3835, 480, 479), licence="Hub: MIT", note="Paris census tables (POPP), 1926-1946."),
    "casia": DatasetSpec(
        key="casia", hf_dataset="Teklia/CASIA-HWDB2-line", hf_revision="2974a15ceab0657047a28d2a71b76bd7544d4663", language="zh",
        script="Han", expected_sizes=(33401, 8318, 10441), licence="Hub: MIT; CASIA: free for academic research",
        note="Chinese (CASIA-HWDB2.0-2.2 lines); thousands of output units -> expect a much larger alphabet."),
    "khatt": DatasetSpec(
        key="khatt", hf_dataset="Teklia/KHATT_v1.0", hf_revision="7da941f6705364c085107c75af4f75e5d421fa4e", language="ar",
        script="Arabic", rtl=True, expected_sizes=(4672, 963, 1038), licence="no Hub tag; KHATT: free research use",
        note="Modern Arabic (KHATT v1.0); text in LOGICAL order -> rtl=True converts it to visual order for the recogniser."),
    # the synthetic corpus of Config.smoke(): generated fake handwriting, plumbing tests only
    "synthetic": DatasetSpec(key="synthetic", hf_dataset="", language="en", script="Latin", note="SYNTHETIC smoke-test data."),
}


def get_spec(key: str, custom=()) -> DatasetSpec:
    for s in custom:
        if s.key == key:
            return s
    if key not in DATASETS:
        raise KeyError(f"unknown dataset '{key}'; registry keys: {sorted(DATASETS)}; or add a DatasetSpec to cfg.data.custom_sources")
    return DATASETS[key]
