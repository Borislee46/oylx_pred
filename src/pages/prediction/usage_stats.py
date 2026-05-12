"""Lightweight usage counter for (university, major) prediction combos.

Writes to cache/usage_stats.json — a simple {"university|major": count} dict.
Used to feed hot-path config for the next release.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

_CACHE_DIR = Path(__file__).parent.parent.parent.parent / "cache"
_STATS_PATH = _CACHE_DIR / "usage_stats.json"
_LOCK = threading.Lock()
_MAX_ENTRIES = 2000  # cap to prevent unbounded growth


def _load() -> dict[str, int]:
    try:
        if _STATS_PATH.exists():
            return json.loads(_STATS_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save(data: dict[str, int]) -> None:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    tmp.replace(_STATS_PATH)


def increment(results: list[dict]) -> None:
    """Increment usage counts for each (university, major) in results."""
    if not results:
        return
    with _LOCK:
        data = _load()
        for r in results:
            uni = str(r.get("university", "")).strip()
            major = str(r.get("major", "")).strip()
            if not uni or not major:
                continue
            key = f"{uni}|{major}"
            data[key] = data.get(key, 0) + 1
        # Prune if oversized: keep top _MAX_ENTRIES by count
        if len(data) > _MAX_ENTRIES:
            data = dict(sorted(data.items(), key=lambda x: -x[1])[:_MAX_ENTRIES])
        _save(data)


def get_top(n: int = 30) -> list[tuple[str, str, int]]:
    """Return top N (university, major, count) tuples."""
    data = _load()
    sorted_items = sorted(data.items(), key=lambda x: -x[1])[:n]
    result: list[tuple[str, str, int]] = []
    for key, cnt in sorted_items:
        parts = key.split("|", 1)
        if len(parts) == 2:
            result.append((parts[0], parts[1], cnt))
    return result


def get_stats_json() -> str:
    """Return full stats as a compact JSON string (for download)."""
    data = _load()
    return json.dumps(data, ensure_ascii=False, indent=2)
