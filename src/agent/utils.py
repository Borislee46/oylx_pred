from __future__ import annotations

from typing import Any


def truncate_text(x: Any, limit: int) -> str:
    s = str(x or "").strip()
    limit = int(limit) if limit is not None else 0
    if limit > 0 and len(s) > limit:
        return s[:limit]
    return s


def to_str_singleline(x: Any) -> str:
    return str(x or "").strip().replace("\n", " ")


def to_float(x: Any, default: float = 0.0) -> float:
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return float(default)
        try:
            return float(s)
        except ValueError:
            return float(default)
    return float(default)


def parse_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n", ""}:
            return False
    return bool(default)
