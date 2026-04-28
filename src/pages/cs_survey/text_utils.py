from __future__ import annotations

import re

import pandas as pd

_PLACEHOLDER_TEXTS = {"", "/", "暂无", "无", "nan", "none"}
_PURE_NUMBER = re.compile(r"^\d+(?:\.\d+)?$")


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_meaningful_text(value: object) -> bool:
    text = normalize_text(value)
    if not text:
        return False
    lowered = text.lower()
    if lowered in _PLACEHOLDER_TEXTS or text in _PLACEHOLDER_TEXTS:
        return False
    if _PURE_NUMBER.fullmatch(text):
        return False
    return True


def meaningful_text_mask(series: pd.Series) -> pd.Series:
    return series.apply(is_meaningful_text)
