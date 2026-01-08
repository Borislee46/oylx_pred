import math
from typing import Any

import numba
import pandas as pd

from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
    COMPREHENSIVE_SCORE_BOOST_THRESHOLD,
    DEFAULT_UNIVERSITY_DIFFICULTY_ORDER,
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
    PROBABILITY_ADJUSTMENT_THRESHOLD,
    PROBABILITY_EXTREME_STD_MULTIPLIER,
    PROBABILITY_MIN_VALUE,
    SELECTION_SCORE_BOOST_FACTOR,
    UNIVERSITY_DIFFICULTY_CONFIG_PATH,
)
from src.pages.prediction.result_modifier.streamlit_cache import cache_data
from src.pages.prediction.result_modifier.utils import (
    apply_cross_major_penalty_if_needed,
    clip_probability,
    compute_dataframe_hash,
)
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service
from src.utils.university_difficulty_service import get_university_difficulty_order

logger = setup_logger("page3", "prediction")


@numba.njit(cache=True)
def _fast_sigmoid(x: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


@numba.njit(cache=True)
def _compute_gpa_penalty(
    gpa: float,
    gpa_minimum: float,
    gpa_mean: float,
    gpa_std: float,
    severe_threshold: float,
    max_coeff: float,
    quad_coeff: float,
) -> float:
    if gpa < gpa_minimum:
        return severe_threshold
    if gpa >= gpa_mean:
        return 0.0
    gpa_gap = (gpa_mean - gpa) / gpa_std
    return min(max_coeff, quad_coeff * (gpa_gap**2))


@numba.njit(cache=True)
def _compute_language_penalty(
    score: float,
    minimum: float,
    pass_line: float,
    std: float,
    severe_threshold: float,
    l1_mult: float,
    l1_thresh: float,
    l2_mult: float,
    l2_thresh: float,
    l3_thresh: float,
) -> float:
    if score < minimum:
        return severe_threshold
    if score >= pass_line:
        return 0.0

    if score < (pass_line - l1_mult * std):
        return l1_thresh
    elif score < (pass_line - l2_mult * std):
        return l2_thresh
    else:
        return l3_thresh


@numba.njit(cache=True)
def _compute_adjusted_probability(
    prob: float,
    gpa_penalty: float,
    lang_penalty: float,
    min_val: float,
    adj_thresh: float,
    gpa: float,
    gpa_mean: float,
    gpa_std: float,
    lang_score: float,
    lang_pass_line: float,
    lang_std: float,
    extreme_mult: float,
) -> float:
    adjusted = prob
    if gpa_penalty > 0:
        adjusted *= 1.0 - gpa_penalty
    if lang_penalty > 0:
        adjusted *= 1.0 - lang_penalty

    if adjusted < adj_thresh:
        is_extreme_gpa = gpa < (gpa_mean - extreme_mult * gpa_std)
        is_extreme_lang = lang_score < (lang_pass_line - extreme_mult * lang_std)
        if is_extreme_gpa or is_extreme_lang:
            return min_val

    return max(min_val, min(adjusted, 1.0))


@cache_data(show_spinner=False)
def _calculate_cases_statistics(_cases_df: pd.DataFrame, hash_key: str) -> dict[str, float]:
    stats = {
        "gpa_mean": 0.0,
        "gpa_std": 1e-6,
        "language_mean": 0.0,
        "language_std": 1e-6,
        "language_pass_line": 0.0,
    }

    if _cases_df is None or _cases_df.empty:
        return stats

    try:
        if "gpa" in _cases_df.columns:
            gpa_series = pd.to_numeric(_cases_df["gpa"], errors="coerce").dropna()
            if not gpa_series.empty:
                stats["gpa_mean"] = float(gpa_series.mean())
                stats["gpa_std"] = max(1e-6, float(gpa_series.std()))

        lang_cols = [c for c in ["toefl", "ielts"] if c in _cases_df.columns]
        if lang_cols:
            temp_df = _cases_df[lang_cols].apply(pd.to_numeric, errors="coerce")

            if "toefl" in temp_df.columns:
                temp_df["toefl"] = temp_df["toefl"] / 120.0
            if "ielts" in temp_df.columns:
                temp_df["ielts"] = temp_df["ielts"] / 9.0

            norm_scores = temp_df.max(axis=1).dropna()

            if not norm_scores.empty:
                stats["language_mean"] = float(norm_scores.mean())
                stats["language_std"] = max(1e-6, float(norm_scores.std()))

        pass_line = (
            stats["language_mean"] - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * stats["language_std"]
        )
        stats["language_pass_line"] = max(LANGUAGE_MINIMUM, float(pass_line))

    except Exception as e:
        logger.warning(f"计算统计信息失败: {e}")

    return stats


class ProbabilityAdjuster:
    def __init__(self, cases_df: pd.DataFrame, data_hash: str | int | None = None):
        self._data_hash = (
            str(data_hash) if data_hash is not None else compute_dataframe_hash(cases_df)
        )
        self.stats = _calculate_cases_statistics(cases_df, self._data_hash)

        self.gpa_mean = self.stats["gpa_mean"]
        self.gpa_std = self.stats["gpa_std"]
        self.language_mean = self.stats["language_mean"]
        self.language_std = self.stats["language_std"]
        self.language_pass_line = self.stats["language_pass_line"]

        self.gpa_minimum = GPA_MINIMUM
        self.language_minimum = LANGUAGE_MINIMUM

        self.difficulty_order = get_university_difficulty_order(
            UNIVERSITY_DIFFICULTY_CONFIG_PATH,
            DEFAULT_UNIVERSITY_DIFFICULTY_ORDER,
        )
        self.difficulty_map = {uni: i for i, uni in enumerate(self.difficulty_order)}
        self.max_difficulty_index = len(self.difficulty_order)

    def get_university_difficulty_score(self, university_name: str) -> float:
        return self._get_university_difficulty_score(university_name)

    def get_comprehensive_score(
        self, gpa: float, language_score: float, background_university: str | None
    ) -> float:
        return self._calculate_comprehensive_score(gpa, language_score, background_university)

    def _get_university_difficulty_score(self, university_name: str) -> float:
        if not university_name:
            return 0.0
        name = university_name.strip()
        if name in self.difficulty_map:
            rank = self.difficulty_map[name]
            return 1.0 - (rank / max(1, self.max_difficulty_index))
        return 0.0

    @staticmethod
    def _sigmoid_score(x: float, k: float, x0: float) -> float:
        return _fast_sigmoid(x, k, x0)

    def _gpa_to_score(self, gpa: float) -> float:
        return _fast_sigmoid(gpa, 3.0, 3.3)

    def _language_to_score(self, language_score: float) -> float:
        return _fast_sigmoid(language_score, 15.0, 0.72)

    def _calculate_comprehensive_score(
        self, gpa: float, language_score: float, background_university: str | None
    ) -> float:
        gpa_score = _fast_sigmoid(gpa, 3.0, 3.3)
        lang_score = _fast_sigmoid(language_score, 15.0, 0.72)

        service = get_school_level_service()
        school_score = service.get_school_score(background_university)

        total_score = 0.4 * gpa_score + 0.3 * lang_score + 0.3 * school_score
        return total_score

    def calculate_selection_score(
        self,
        similarity: float,
        target_university: str,
        gpa: float,
        language_score: float,
        background_university: str | None,
    ) -> float:
        comp_score = self.get_comprehensive_score(gpa, language_score, background_university)
        target_diff_score = self.get_university_difficulty_score(target_university)

        boost = 0.0
        if comp_score > COMPREHENSIVE_SCORE_BOOST_THRESHOLD and target_diff_score > 0.0:
            boost = SELECTION_SCORE_BOOST_FACTOR * comp_score * target_diff_score

        return similarity * (1.0 + boost)

    def _calculate_gpa_penalty(self, gpa: float) -> float:
        return _compute_gpa_penalty(
            gpa,
            self.gpa_minimum,
            self.gpa_mean,
            self.gpa_std,
            GPA_PENALTY_SEVERE_THRESHOLD,
            GPA_PENALTY_MAX_COEFFICIENT,
            GPA_PENALTY_QUADRATIC_COEFFICIENT,
        )

    def _calculate_language_penalty(self, language_score: float) -> float:
        return _compute_language_penalty(
            language_score,
            self.language_minimum,
            self.language_pass_line,
            self.language_std,
            LANGUAGE_PENALTY_SEVERE_THRESHOLD,
            LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER,
            LANGUAGE_PENALTY_LEVEL_1_THRESHOLD,
            LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER,
            LANGUAGE_PENALTY_LEVEL_2_THRESHOLD,
            LANGUAGE_PENALTY_LEVEL_3_THRESHOLD,
        )

    def get_penalties(self, gpa: float, language_score: float) -> dict[str, float]:
        return {
            "gpa": self._calculate_gpa_penalty(gpa),
            "language": self._calculate_language_penalty(language_score),
        }

    def adjust_probability(
        self,
        probability: float,
        gpa: float | None,
        language_score: float | None,
        background_university_name: str | None = None,
    ) -> float:
        if gpa is None or language_score is None:
            return probability

        if gpa < self.gpa_minimum and language_score < self.language_minimum:
            return PROBABILITY_MIN_VALUE

        gpa_penalty = self._calculate_gpa_penalty(gpa)
        language_penalty = self._calculate_language_penalty(language_score)

        return _compute_adjusted_probability(
            probability,
            gpa_penalty,
            language_penalty,
            PROBABILITY_MIN_VALUE,
            PROBABILITY_ADJUSTMENT_THRESHOLD,
            gpa,
            self.gpa_mean,
            self.gpa_std,
            language_score,
            self.language_pass_line,
            self.language_std,
            PROBABILITY_EXTREME_STD_MULTIPLIER,
        )


def penalize_cross_major_without_cases(
    user_specified_results: list[dict[str, Any]],
    background_major: str,
    cases_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    if not user_specified_results or not background_major or cases_df is None or cases_df.empty:
        return user_specified_results

    bg_major_clean = str(background_major).strip()
    admitted_combinations = get_admitted_combinations_from_dataframe(cases_df, bg_major_clean)

    adjusted_results = []
    for result in user_specified_results:
        result_copy = result.copy()

        original_prob = result_copy.get("probability", 0.0)
        if original_prob is not None:
            adjusted_prob = apply_cross_major_penalty_if_needed(
                result=result,
                probability=clip_probability(original_prob),
                admitted_combinations=admitted_combinations,
                check_admitted_field=False,
            )
            result_copy["probability"] = clip_probability(adjusted_prob)

        adjusted_results.append(result_copy)

    return adjusted_results
