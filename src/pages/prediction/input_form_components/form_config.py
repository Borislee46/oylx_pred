from src.utils.schools.config_loader import (
    TARGET_COUNTRY_UNIVERSITY_MAP,
    UNIVERSITY_DISPLAY_ORDER,
)

GPA_SCALES: dict[str, dict] = {
    "4.0": {"max": 4.0, "step": 0.1, "format": "%.2f"},
    "4.3": {"max": 4.3, "step": 0.1, "format": "%.2f"},
    "4.5": {"max": 4.5, "step": 0.1, "format": "%.2f"},
    "5.0": {"max": 5.0, "step": 0.1, "format": "%.2f"},
    "10": {"max": 10.0, "step": 0.5, "format": "%.1f"},
    "20": {"max": 20.0, "step": 0.5, "format": "%.1f"},
    "100": {"max": 100.0, "step": 1.0, "format": "%.0f"},
}

DEFAULT_GPA_SCALE = "4.0"

LANGUAGE_TYPES = ["雅思", "托福"]

LANGUAGE_SCORE_RANGES: dict[str, dict] = {
    "雅思": {"min": 0.0, "max": 9.0, "step": 0.5, "format": "%.1f"},
    "托福": {"min": 0, "max": 120, "step": 1, "format": "%d"},
}

UNIVERSITY_SORT_ORDER: list[str] = list(UNIVERSITY_DISPLAY_ORDER)

TARGET_COUNTRIES: list[str] = list(TARGET_COUNTRY_UNIVERSITY_MAP.keys())

GPA_WARNING_THRESHOLDS: dict[str, float] = {
    "4.0": 2.0,
    "4.3": 2.15,
    "4.5": 2.25,
    "5.0": 2.5,
    "10": 5.0,
    "20": 10.0,
    "100": 50.0,
}

LANGUAGE_WARNING_THRESHOLDS: dict[str, float] = {"雅思": 5.5, "托福": 72}

DEFAULT_LANGUAGE_SCORES: dict[str, float] = {
    "雅思": 6.5,
    "托福": 90,
}

STANDARDIZED_TEST_TYPES: list[str] = ["GRE", "GMAT"]

GRE_SCORE_RANGE: dict = {"min": 260, "max": 340, "step": 1, "format": "%d"}
GMAT_SCORE_RANGE: dict = {"min": 200, "max": 800, "step": 10, "format": "%d"}

GRE_BONUS_THRESHOLD: float = 311
GRE_SIGMOID_MIDPOINT: float = 325
GRE_SIGMOID_STEEPNESS: float = 0.5
GRE_MAX_BONUS: float = 0.5

GMAT_BONUS_THRESHOLD: float = 611
GMAT_SIGMOID_MIDPOINT: float = 700
GMAT_SIGMOID_STEEPNESS: float = 0.05
GMAT_MAX_BONUS: float = 0.5
