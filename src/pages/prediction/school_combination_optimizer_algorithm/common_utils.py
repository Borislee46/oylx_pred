from typing import Any


def clip_probability(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return default


def normalize_school_name(name: str | None) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    s = s.replace("\u00a0", "").replace("\u3000", "")
    s = "".join(ch for ch in s if not ch.isspace())
    s = s.replace("（", "(").replace("）", ")")
    return s
