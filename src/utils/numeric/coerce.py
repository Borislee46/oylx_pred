import math
from typing import Any

from src.utils.numeric.scalars import clip_probability


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def clip_probability_coerce(value: Any, default: float = 0.0) -> float:
    f = safe_float(value, default=default)
    if not math.isfinite(f):
        return default
    return clip_probability(f)


def float_or_none(v: Any) -> float | None:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def prob_to_pct(p: float | None, *, cap: int = 100) -> int:
    if p is None:
        return 0
    return min(cap, round(float(p) * 100.0))


def prob_round(value: Any, ndigits: int = 4, *, default: float = 0.0) -> float:
    return round(clip_probability_coerce(value, default=default), ndigits)
