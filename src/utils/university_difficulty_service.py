from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_university_difficulty_order(
    config_path: Path,
    default_order: tuple[str, ...],
) -> list[str]:
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f) or {}
        order = cfg.get("difficulty_order", None)
        if isinstance(order, list) and all(isinstance(x, str) and x.strip() for x in order):
            return [x.strip() for x in order]
    except (FileNotFoundError, OSError, json.JSONDecodeError, TypeError, ValueError):
        pass

    return list(default_order)
