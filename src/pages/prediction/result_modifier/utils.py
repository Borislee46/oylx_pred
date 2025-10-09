import hashlib
import re
import string
from typing import Any

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


def is_effectively_empty(text: str | None) -> bool:
    if text is None:
        return True
    t = str(text).strip()
    if not t:
        return True
    tl = t.lower()
    if tl in INVALID_TOKENS:
        return True
    punct = string.punctuation + "·—-_/／\\|~`'\"，。；：、"
    stripped = tl.strip(punct)
    return len(stripped) == 0


def clip_probability(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def generate_content_hash(content: str) -> str:
    try:
        return hashlib.md5(content.encode("utf-8")).hexdigest()
    except Exception:
        return f"fallback_{hash(content)}"


def has_valid_experience_details(experience_details: dict[str, str] | None) -> bool:
    if not experience_details:
        return False
    keys = ("research_details", "award_details", "internship_details", "paper_details")
    for k in keys:
        if not is_effectively_empty(experience_details.get(k, "")):
            return True
    return False


def has_meaningful_experience_text(experience_details: dict[str, str] | None) -> bool:
    if not has_valid_experience_details(experience_details):
        return False

    if experience_details is None:
        return False

    try:
        keys = ("research_details", "award_details", "internship_details", "paper_details")
        merged = " ".join(str(experience_details.get(k, "")) for k in keys)
        merged = merged.strip().lower()
        cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", merged)
        return len(cleaned) >= 6
    except Exception:
        return True
