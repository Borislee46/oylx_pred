import hashlib
import string
from typing import Any

import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "empty_df"
    try:
        meta_str = f"{df.shape}-{tuple(map(str, df.columns))}-{tuple(map(str, df.dtypes))}"
        sample = pd.concat([df.head(1), df.tail(1)])
        sample_digest = hashlib.md5(sample.to_numpy().tobytes()).hexdigest()
        return hashlib.md5(f"{meta_str}-{sample_digest}".encode()).hexdigest()
    except Exception:
        return "fallback_hash"


FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

EXPERIENCE_ANALYSIS_MESSAGES = [
    "正在智能分析您的背景经历",
    "深度解析{field}经历中",
    "正在评估您的{field}背景",
    "智能识别{field}相关信息",
    "分析{field}内容与专业匹配度",
]

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


def is_effectively_empty(text: str | None) -> bool:
    if text is None:
        return True
    t = str(text).strip()
    if not t:
        return True
    tl = t.lower()
    if tl in INVALID_TOKENS:
        return True
    stripped = tl.strip(PUNCTUATION_CHARS)
    return len(stripped) == 0


def clip_probability(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def has_valid_experience_details(experience_details: dict[str, str] | None) -> bool:
    if not experience_details:
        return False
    keys = ("research_details", "award_details", "internship_details", "paper_details")
    for k in keys:
        if not is_effectively_empty(experience_details.get(k, "")):
            return True
    return False
