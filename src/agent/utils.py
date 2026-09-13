from typing import Any


def truncate_text(x: Any, limit: int) -> str:
    s = str(x or "").strip()
    limit = int(limit) if limit is not None else 0
    if limit > 0 and len(s) > limit:
        return s[:limit]
    return s


def to_str_singleline(x: Any) -> str:
    return str(x or "").strip().replace("\n", " ")


def parse_bool(x: Any, default: bool = False) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(x)
    if isinstance(x, str):
        raw = x.strip()
        if not raw:
            return False
        s = raw.lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
        if raw in {"是", "对", "正确"}:
            return True
        if raw in {"否", "错", "错误"}:
            return False
    return bool(default)
