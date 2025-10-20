from dataclasses import dataclass
from typing import Iterable, List

SCHOOL_CATEGORY_THRESHOLDS = {
    "safety": 0.75,
    "target_lower": 0.55,
}

BALANCE_RATIOS = {
    "safety": 0.3,
    "target": 0.4,
    "reach": 0.3,
}

BALANCE_RATIOS_HIGH_BG = {
    "safety": 0.2,
    "target": 0.4,
    "reach": 0.4,
}

ADAPTIVE_THRESHOLD_PERCENTILES = {
    "reach_percentile_val": 10,
    "safety_percentile_val": 70,
}

ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG = {
    "reach_percentile_val": 20,
    "safety_percentile_val": 60,
}

SAFETY_SCHOOL_THRESHOLD = 0.7
MIN_SAFETY_SCHOOL_COUNT_DEFAULT = 3
MIN_SAFETY_SCHOOL_COUNT_HIGH_BG = 1

MONTE_CARLO_DEFAULTS = {
    "n_simulations": 5000,
    "min_simulations": 1000,
    "max_simulations": 10000,
    "convergence_threshold": 0.01,
    "batch_size": 500,
}

TOP3_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]

TOP5_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港理工大学",
    "香港城市大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]

TOP8_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港城市大学",
    "香港理工大学",
    "香港浸会大学",
    "香港教育大学",
    "香港岭南大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]

MACAU_UNIVERSITIES = {
    "澳门大学",
    "澳门科技大学",
    "澳门城市大学",
    "澳门理工大学",
}

GLOBAL_MIN_SIMILARITY = 0.90
PRESTIGE_WEIGHT = 3.0

OBJECTIVE_WEIGHTS = {
    "rejection_probability": 0.9,
    "diversity": 5.0,
    "balance_score": 1.0,
    "major_similarity": 2.0,
    "new_major_ratio": 0.5,
}

TOP_BG_LEVELS_SET = {"985", "211", "1-50", "51-100"}

MIN_TOP3_COUNT_FOR_HIGH_BG = 2
MIN_TOP5_COUNT_FOR_HIGH_BG = 3

PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2 = 5
PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8 = 6
PRIORITY_THRESHOLD_TOP_BG_DEFAULT = 7

PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0 = 5
PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT = 7

CONSTRAINT_FLEXIBILITY: dict[str, bool | float | int] = {}


@dataclass(frozen=True)
class PlanConfig:
    name: str
    min_schools: int
    max_schools: int


DEFAULT_PLAN_CONFIGS: List[PlanConfig] = [
    PlanConfig(name="申请策略1", min_schools=6, max_schools=6),
    PlanConfig(name="申请策略2", min_schools=9, max_schools=9),
    PlanConfig(name="申请策略3", min_schools=10, max_schools=10),
]


def get_plan_configs(overrides: Iterable[PlanConfig] | None = None) -> List[PlanConfig]:
    if overrides:
        return list(overrides)
    return DEFAULT_PLAN_CONFIGS.copy()


CROSS_FACULTY_RULES: dict[str, set[str]] = {
    "文学院": {"文学院", "社会科学院", "教育学院", "商学院", "艺术学院"},
    "社会科学院": {
        "社会科学院",
        "文学院",
        "商学院",
        "教育学院",
        "艺术学院",
        "建筑学院",
    },
    "法学院": {"法学院"},
    "教育学院": {"教育学院", "文学院", "社会科学院"},
    "商学院": {"商学院", "社会科学院", "文学院"},
    "理学院": {
        "理学院",
        "工程学院",
        "商学院",
        "经济金融学院",
        "科学学院",
        "计算机科学院",
    },
    "工程学院": {
        "工程学院",
        "理学院",
        "商学院",
        "计算机科学院",
        "建筑学院",
        "设计学院",
        "科学学院",
    },
    "计算机科学院": {"计算机科学院", "工程学院", "理学院", "商学院"},
    "艺术学院": {"艺术学院", "社会科学院", "文学院", "设计学院", "建筑学院"},
    "医学院": {"医学院"},
    "建筑学院": {"建筑学院", "工程学院", "设计学院", "艺术学院"},
    "设计学院": {"设计学院", "艺术学院", "建筑学院", "社会科学院"},
}


def get_allowed_target_faculties(background_faculty: str | None) -> set[str]:
    if not background_faculty:
        return set()
    return CROSS_FACULTY_RULES.get(background_faculty, set())
