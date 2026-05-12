import json
from pathlib import Path


def _get_project_root() -> Path:
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[4]


PROJECT_ROOT = _get_project_root()
PREDICTION_RULES_PATH = PROJECT_ROOT / "config" / "prediction_rules.json"


def _load_display_order() -> list[str]:
    if PREDICTION_RULES_PATH.exists():
        try:
            with open(PREDICTION_RULES_PATH, encoding="utf-8") as f:
                rules = json.load(f)
                return rules.get("UNIVERSITY_DISPLAY_ORDER", [])
        except Exception:
            pass
    return []


UNIVERSITY_SORT_ORDER = _load_display_order() or [
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

UNIVERSITY_ORDER_MAP = {name: i for i, name in enumerate(UNIVERSITY_SORT_ORDER)}

TOP_SIM_RESULT_UI_CONFIG = {
    "推荐院校": "small",
    "推荐专业": "medium",
    "±%": "small",
    "录取概率": "small",
    "推荐专业详情": "small",
}

TOP_CROSS_RESULT_UI_CONFIG = {
    "推荐院校": "small",
    "推荐专业": "medium",
    "±%": "small",
    "录取概率": "small",
    "推荐专业详情": "small",
}
