from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from src.utils.numeric import clip_probability
from src.utils.schools.level_service import (
    SCHOOL_LEVEL_PRIORITY,
    get_school_level_service,
)


def calculate_prestige_score(schools: list[dict[str, Any]]) -> float:
    if not schools:
        return 0.0
    priorities = [get_priority_cached(school.get("university", "")) for school in schools]
    return clip_probability((12 - np.mean(priorities)) / 11)


@lru_cache(maxsize=5000)
def get_priority_cached(university: str) -> int:
    svc = get_school_level_service()
    info = svc.get_school_info(university or "")
    return int(info.get("priority", SCHOOL_LEVEL_PRIORITY.get("未知", 12)))


@lru_cache(maxsize=5000)
def _prestige(university: str) -> float:
    return calculate_prestige_score([{"university": university}])


def school_prestige(s: dict[str, Any]) -> float:
    return _prestige(s.get("university", ""))
