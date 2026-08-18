from __future__ import annotations

import json
from pathlib import Path

from src.adjustment.config import (
    GPA_MINIMUM,
    GPA_PENALTY_MAX_COEFFICIENT,
    GPA_PENALTY_QUADRATIC_COEFFICIENT,
    GPA_PENALTY_SEVERE_THRESHOLD,
    LANGUAGE_MINIMUM,
    LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_1_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER,
    LANGUAGE_PENALTY_LEVEL_2_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_3_THRESHOLD,
    LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER,
    LANGUAGE_PENALTY_SEVERE_THRESHOLD,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_scalar
from src.utils.schools.constants import SCHOOL_LEVEL_SCORES

_logger = setup_logger("page3", "prediction")

MAX_GPA = 4.0
MAX_ACADEMIC_ITEMS = 3
MAX_PRACTICAL_ITEMS = 5

_COHORT_JSON = Path(__file__).resolve().parents[4] / "config" / "cohort_statistics.json"
_cohort = {}
try:
    if _COHORT_JSON.exists():
        _cohort = json.loads(_COHORT_JSON.read_text(encoding="utf-8"))
        _logger.info("Loaded cohort statistics from %s", _COHORT_JSON)
    else:
        _logger.warning("cohort_statistics.json not found, using hardcoded defaults")
except Exception:
    _logger.warning(
        "Failed to load cohort_statistics.json, using hardcoded defaults", exc_info=True
    )

GPA_MEAN: float = _cohort.get("gpa_mean", 3.0)
GPA_STD: float = _cohort.get("gpa_std", 0.5)
LANGUAGE_MEAN: float = _cohort.get("language_mean", 0.75)
LANGUAGE_STD: float = _cohort.get("language_std", 0.15)

RADAR_LABELS = ["学术绩点", "语言能力", "科研论文", "实习获奖", "学校水平"]


def calculate_gpa_score(
    gpa: float,
    gpa_mean: float = GPA_MEAN,
    gpa_std: float = GPA_STD,
) -> float:
    if gpa <= 0:
        return 0.0
    if gpa < GPA_MINIMUM:
        return max(0.0, (1.0 - GPA_PENALTY_SEVERE_THRESHOLD) * 100)

    base = (gpa / MAX_GPA) * 100
    if gpa >= gpa_mean:
        return clip_scalar(base, 0.0, 100.0)

    gap = (gpa_mean - gpa) / gpa_std if gpa_std > 0 else 0.0
    penalty = min(GPA_PENALTY_MAX_COEFFICIENT, GPA_PENALTY_QUADRATIC_COEFFICIENT * gap * gap)
    return clip_scalar(base * (1.0 - penalty), 0.0, 100.0)


def calculate_language_score(
    language_score: float,
    lang_mean: float = LANGUAGE_MEAN,
    lang_std: float = LANGUAGE_STD,
) -> float:
    if language_score <= 0:
        return 0.0
    if language_score < LANGUAGE_MINIMUM:
        return max(0.0, (1.0 - LANGUAGE_PENALTY_SEVERE_THRESHOLD) * 100)

    base = language_score * 100
    excellent = lang_mean + 0.5 * lang_std
    pass_line = lang_mean - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * lang_std

    if language_score >= excellent:
        return clip_scalar(base, 0.0, 100.0)

    if language_score >= pass_line:
        denom = excellent - pass_line
        gap_ratio = (excellent - language_score) / denom if denom > 0 else 0.0
        reduction = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD * gap_ratio * 0.15
        return clip_scalar(base * (1.0 - reduction), 0.0, 100.0)

    # Below pass_line: stepped penalties
    below1 = pass_line - LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER * lang_std
    if language_score < below1:
        penalty = LANGUAGE_PENALTY_LEVEL_1_THRESHOLD
    elif language_score < pass_line - LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER * lang_std:
        penalty = LANGUAGE_PENALTY_LEVEL_2_THRESHOLD
    else:
        penalty = LANGUAGE_PENALTY_LEVEL_3_THRESHOLD

    return clip_scalar(base * (1.0 - penalty), 0.0, 100.0)


def calculate_academic_score(research_count: int, paper_count: int) -> float:
    weighted = research_count * 0.6 + paper_count * 0.4
    return min(weighted / MAX_ACADEMIC_ITEMS * 100, 100.0)


def calculate_practical_score(internship_count: int, award_count: int) -> float:
    weighted = internship_count * 0.5 + award_count * 0.5
    return min(weighted / MAX_PRACTICAL_ITEMS * 100, 100.0)


def calculate_school_score(school_level: str | None) -> float:
    score = SCHOOL_LEVEL_SCORES.get(school_level, 0.50)
    return score * 100


def normalise_language(raw_score: float, lang_type: str) -> float:
    if raw_score <= 0:
        return 0.0
    lang_max = 120.0 if lang_type in ("托福", "TOEFL") else 9.0
    return min(raw_score / lang_max, 1.0)


def compute_radar_values(input_data: dict) -> tuple[list[float], list[str]]:
    # GPA
    try:
        gpa = float(input_data.get("gpa", 0) or 0)
    except (TypeError, ValueError):
        gpa = 0.0
    if gpa != gpa:  # NaN 防护
        gpa = 0.0

    lang_score = float(input_data.get("language_score", 0) or 0)
    if lang_score != lang_score:  # NaN 防护
        lang_score = 0.0
    if lang_score <= 0:
        try:
            raw_lang = float(input_data.get("language_score_raw", 0) or 0)
        except (TypeError, ValueError):
            raw_lang = 0.0
        if raw_lang != raw_lang:  # NaN 防护
            raw_lang = 0.0
        lang_type = str(input_data.get("language_type", ""))
        lang_score = normalise_language(raw_lang, lang_type)

    research_n = int(input_data.get("research_count", 0) or 0)
    paper_n = int(input_data.get("paper_count", 0) or 0)

    intern_n = int(input_data.get("internship_count", 0) or 0)
    award_n = int(input_data.get("award_count", 0) or 0)

    bg_uni = str(input_data.get("background_university", ""))
    school_level = None
    if bg_uni:
        try:
            from src.utils.schools.level_service import SchoolLevelService

            info = SchoolLevelService().get_school_info(bg_uni)
            school_level = info.get("school_level", None)
        except Exception:
            _logger.warning(
                "compute_radar_values: SchoolLevelService lookup failed for %s",
                bg_uni,
                exc_info=True,
            )

    values = [
        calculate_gpa_score(gpa),
        calculate_language_score(lang_score),
        calculate_academic_score(research_n, paper_n),
        calculate_practical_score(intern_n, award_n),
        calculate_school_score(school_level),
    ]
    return values, list(RADAR_LABELS)
