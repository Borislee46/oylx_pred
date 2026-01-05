import hashlib
import string
from typing import Any

import pandas as pd
import streamlit as st
from numba import jit

from src.pages.prediction.result_modifier.config import (
    CROSS_MAJOR_PENALTY_FACTOR,
    CROSS_MAJOR_SIMILARITY_MIN,
    MIN_SIMILARITY_THRESHOLD,
)


def has_streamlit_runtime() -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(exists and exists())


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty_df"
    try:
        meta_str = f"{df.shape}-{tuple(map(str, df.columns))}-{tuple(map(str, df.dtypes))}"
        n = len(df)
        if n <= 8:
            idxs = list(range(n))
        else:
            idxs = [0, n - 1, n // 2, n // 4, (3 * n) // 4, n // 3, (2 * n) // 3, 1]
            idxs = [i for i in idxs if 0 <= i < n]
            idxs = list(dict.fromkeys(idxs))[:8]

        sample = df.iloc[idxs]
        values_hash = pd.util.hash_pandas_object(sample, index=True).to_numpy().tobytes()
        sample_digest = hashlib.md5(values_hash).hexdigest()
        return hashlib.md5(f"{meta_str}-{sample_digest}".encode()).hexdigest()
    except (IndexError, KeyError, TypeError, ValueError, AttributeError):
        return "fallback_hash"


INVALID_TOKENS = {
    "",
    "无",
    "暂无",
    "没有",
    "na",
    "n/a",
    "none",
    "null",
    "-",
    "--",
    "—",
    "/",
    "／",
    ".",
}

PUNCTUATION_CHARS = string.punctuation + "·—-_/／\\|~`'\"，。；：、"


def is_effectively_empty(text: Any) -> bool:
    if not text:
        return True
    t = str(text).strip()
    if not t:
        return True
    tl = t.lower()
    if tl in INVALID_TOKENS:
        return True
    return not tl.strip(PUNCTUATION_CHARS)


def get_probability(case: dict[str, Any], default: float = 0.0) -> float:
    v = case.get("probability", default)
    val = float(v) if v is not None else default
    return max(0.0, min(1.0, val))


@jit(nopython=True, cache=True)
def clip_probability(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def cross_major_penalty_factor(similarity: Any) -> float:
    s = float(similarity)

    if s >= MIN_SIMILARITY_THRESHOLD:
        return 1.0
    if s <= CROSS_MAJOR_SIMILARITY_MIN:
        return CROSS_MAJOR_PENALTY_FACTOR

    span = MIN_SIMILARITY_THRESHOLD - CROSS_MAJOR_SIMILARITY_MIN
    if span <= 0:
        return CROSS_MAJOR_PENALTY_FACTOR

    t = (s - CROSS_MAJOR_SIMILARITY_MIN) / span
    return CROSS_MAJOR_PENALTY_FACTOR + (1.0 - CROSS_MAJOR_PENALTY_FACTOR) * t


def apply_cross_major_penalty_if_needed(
    result: dict[str, Any],
    probability: float,
    admitted_combinations: set[tuple[Any, Any]] | None = None,
    check_admitted_field: bool = True,
) -> float:
    similarity = result.get("similarity", 1.0)
    sim = float(similarity)

    if sim >= MIN_SIMILARITY_THRESHOLD:
        return probability

    key = (result.get("university"), result.get("major"))
    has_admitted_case = bool(admitted_combinations and key in admitted_combinations)
    if check_admitted_field and result.get("admitted") == 1:
        has_admitted_case = True

    if has_admitted_case:
        return probability

    return probability * cross_major_penalty_factor(sim)


def generate_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def deduplicate_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from src.pages.prediction.result_modifier.types import case_key

    seen = set()
    deduped = []
    for r in results:
        k = case_key(r)
        if k and k not in seen:
            seen.add(k)
            deduped.append(r)
    return deduped


def has_any_experience(experience_details: dict[str, str] | None) -> bool:
    if not experience_details:
        return False
    keys = ("research_details", "award_details", "internship_details", "paper_details")
    return any(not is_effectively_empty(experience_details.get(k, "")) for k in keys)
