from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

_logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_project_root() -> Path:
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[3]

_get_project_root = get_project_root

@lru_cache(maxsize=1)
def _load_rules() -> dict:
    path = _get_project_root() / "config" / "prediction_rules.json"
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        _logger.warning(
            "Failed to load prediction_rules.json from %s, using hardcoded fallbacks",
            path,
            exc_info=True,
        )
    return {}


_rules = _load_rules()

_DEFAULT_DIFFICULTY_ORDER: tuple[str, ...] = (
    "新加坡国立大学",
    "新加坡南洋理工大学",
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "新加坡管理大学",
    "马来亚大学",
    "香港理工大学",
    "香港城市大学",
    "马来西亚理科大学",
    "马来西亚博特拉大学",
    "香港浸会大学",
    "马来西亚国立大学",
    "澳门大学",
    "香港中文大学 (深圳校区)",
    "澳门科技大学",
    "澳门城市大学",
    "澳门理工大学",
    "香港教育大学",
    "香港岭南大学",
    "香港都会大学",
    "香港恒生大学",
    "香港珠海学院",
)

UNIVERSITY_DIFFICULTY_ORDER: tuple[str, ...] = tuple(
    _rules.get("UNIVERSITY_DIFFICULTY_ORDER", _DEFAULT_DIFFICULTY_ORDER)
)

_DEFAULT_DISPLAY_ORDER: list[str] = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港理工大学",
    "香港城市大学",
    "香港中文大学 (深圳校区)",
    "香港浸会大学",
    "香港岭南大学",
    "香港教育大学",
    "香港都会大学",
    "香港恒生大学",
    "香港珠海学院",
    "新加坡南洋理工大学",
    "新加坡国立大学",
    "新加坡管理大学",
    "澳门大学",
    "澳门科技大学",
    "澳门理工大学",
    "澳门城市大学",
    "马来亚大学",
    "马来西亚博特拉大学",
    "马来西亚理科大学",
    "马来西亚国立大学",
]

UNIVERSITY_DISPLAY_ORDER: list[str] = _rules.get("UNIVERSITY_DISPLAY_ORDER", _DEFAULT_DISPLAY_ORDER)

_DEFAULT_COUNTRY_UNIVERSITY_MAP: dict[str, list[str]] = {
    "中国香港": [
        "香港大学",
        "香港中文大学",
        "香港科技大学",
        "香港理工大学",
        "香港城市大学",
        "香港中文大学 (深圳校区)",
        "香港浸会大学",
        "香港岭南大学",
        "香港教育大学",
        "香港都会大学",
        "香港恒生大学",
        "香港珠海学院",
    ],
    "新加坡": ["新加坡南洋理工大学", "新加坡国立大学", "新加坡管理大学"],
    "中国澳门": ["澳门大学", "澳门科技大学", "澳门理工大学", "澳门城市大学"],
    "马来西亚": [
        "马来亚大学",
        "马来西亚博特拉大学",
        "马来西亚理科大学",
        "马来西亚国立大学",
    ],
}

TARGET_COUNTRY_UNIVERSITY_MAP: dict[str, list[str]] = _rules.get(
    "TARGET_COUNTRY_UNIVERSITY_MAP", _DEFAULT_COUNTRY_UNIVERSITY_MAP
)

def get_target_countries() -> list[str]:
    return list(TARGET_COUNTRY_UNIVERSITY_MAP.keys())


def get_all_target_universities() -> list[str]:
    return [uni for schools in TARGET_COUNTRY_UNIVERSITY_MAP.values() for uni in schools]


def get_university_difficulty_order() -> tuple[str, ...]:
    return UNIVERSITY_DIFFICULTY_ORDER


def get_university_display_order() -> list[str]:
    return list(UNIVERSITY_DISPLAY_ORDER)
