import math
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from src.pages.prediction.prediction_utils import normalize_language_score
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
    COMPREHENSIVE_SCORE_BOOST_THRESHOLD,
    CROSS_MAJOR_PENALTY_FACTOR,
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
    MIN_SIMILARITY_THRESHOLD,
    PROBABILITY_ADJUSTMENT_THRESHOLD,
    PROBABILITY_EXTREME_STD_MULTIPLIER,
    PROBABILITY_MIN_VALUE,
    SELECTION_SCORE_BOOST_FACTOR,
    get_university_difficulty_order,
)
from src.pages.prediction.result_modifier.utils import clip_probability, compute_dataframe_hash
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service

logger = setup_logger("page3", "prediction")


@st.cache_data(show_spinner=False)
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
            gpa_series = pd.to_numeric(_cases_df["gpa"], errors="coerce")
            stats["gpa_mean"] = float(np.nan_to_num(gpa_series.mean(), nan=0.0))
            stats["gpa_std"] = float(np.nan_to_num(gpa_series.std(), nan=0.0))
            if stats["gpa_std"] == 0:
                stats["gpa_std"] = 1e-6

        has_toefl = "toefl" in _cases_df.columns
        has_ielts = "ielts" in _cases_df.columns

        norm_scores = None

        if has_toefl or has_ielts:
            scores_list = []
            if has_toefl:
                valid_toefl = _cases_df["toefl"].dropna()
                if not valid_toefl.empty:
                    norm_toefl = valid_toefl.apply(lambda x: normalize_language_score(x, "托福"))
                    scores_list.append(norm_toefl)

            if has_ielts:
                valid_ielts = _cases_df["ielts"].dropna()
                if not valid_ielts.empty:
                    norm_ielts = valid_ielts.apply(lambda x: normalize_language_score(x, "雅思"))
                    scores_list.append(norm_ielts)

            if scores_list:
                temp_df = pd.DataFrame(index=_cases_df.index)
                if has_toefl:
                    temp_df["toefl_norm"] = _cases_df["toefl"].apply(
                        lambda x: normalize_language_score(x, "托福")
                    )
                else:
                    temp_df["toefl_norm"] = np.nan

                if has_ielts:
                    temp_df["ielts_norm"] = _cases_df["ielts"].apply(
                        lambda x: normalize_language_score(x, "雅思")
                    )
                else:
                    temp_df["ielts_norm"] = np.nan

                final_scores = temp_df["toefl_norm"].fillna(temp_df["ielts_norm"]).fillna(0.0)
                norm_scores = final_scores

        if norm_scores is not None:
            stats["language_mean"] = float(np.nan_to_num(norm_scores.mean(), nan=0.0))
            stats["language_std"] = float(np.nan_to_num(norm_scores.std(), nan=0.0))

        if stats["language_std"] == 0:
            stats["language_std"] = 1e-6

        stats["language_pass_line"] = (
            stats["language_mean"] - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * stats["language_std"]
        )

    except Exception as e:
        logger.warning(f"计算统计信息失败: {e}")

    return stats


class ProbabilityAdjuster:
    def __init__(self, cases_df: pd.DataFrame):
        self._data_hash = compute_dataframe_hash(cases_df)
        self.stats = _calculate_cases_statistics(cases_df, self._data_hash)

        self.gpa_mean = self.stats["gpa_mean"]
        self.gpa_std = self.stats["gpa_std"]
        self.language_mean = self.stats["language_mean"]
        self.language_std = self.stats["language_std"]
        self.language_pass_line = self.stats["language_pass_line"]

        self.gpa_minimum = GPA_MINIMUM
        self.language_minimum = LANGUAGE_MINIMUM

        self.difficulty_order = get_university_difficulty_order()
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
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))

    def _gpa_to_score(self, gpa: float) -> float:
        return self._sigmoid_score(gpa, k=3.0, x0=3.3)

    def _language_to_score(self, language_score: float) -> float:
        return self._sigmoid_score(language_score, k=15.0, x0=0.72)

    def _calculate_comprehensive_score(
        self, gpa: float, language_score: float, background_university: str | None
    ) -> float:
        gpa_score = self._gpa_to_score(gpa)
        lang_score = self._language_to_score(language_score)

        school_score = 0.5
        if background_university:
            service = get_school_level_service()
            priority = service.get_school_priority(background_university)
            school_score = max(0.0, min(1.0, 1.0 - (priority - 1) / 10.0))

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
        if gpa < self.gpa_minimum:
            return GPA_PENALTY_SEVERE_THRESHOLD
        if gpa >= self.gpa_mean:
            return 0.0
        gpa_gap = (self.gpa_mean - gpa) / self.gpa_std
        return min(GPA_PENALTY_MAX_COEFFICIENT, GPA_PENALTY_QUADRATIC_COEFFICIENT * gpa_gap**2)

    def _calculate_language_penalty(self, language_score: float) -> float:
        if language_score < self.language_minimum:
            return LANGUAGE_PENALTY_SEVERE_THRESHOLD
        if language_score >= self.language_pass_line:
            return 0.0
        if language_score < (
            self.language_pass_line - LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER * self.language_std
        ):
            return LANGUAGE_PENALTY_LEVEL_1_THRESHOLD
        elif language_score < (
            self.language_pass_line - LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER * self.language_std
        ):
            return LANGUAGE_PENALTY_LEVEL_2_THRESHOLD
        else:
            return LANGUAGE_PENALTY_LEVEL_3_THRESHOLD

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

        adjusted_probability = probability

        gpa_penalty = self._calculate_gpa_penalty(gpa)
        if gpa_penalty > 0:
            adjusted_probability *= 1 - gpa_penalty

        language_penalty = self._calculate_language_penalty(language_score)
        if language_penalty > 0:
            adjusted_probability *= 1 - language_penalty

        if adjusted_probability < PROBABILITY_ADJUSTMENT_THRESHOLD and (
            gpa < self.gpa_mean - PROBABILITY_EXTREME_STD_MULTIPLIER * self.gpa_std
            or language_score
            < self.language_pass_line - PROBABILITY_EXTREME_STD_MULTIPLIER * self.language_std
        ):
            return PROBABILITY_MIN_VALUE

        return max(PROBABILITY_MIN_VALUE, min(adjusted_probability, 1.0))


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

        is_cross_major = result.get("similarity", 1.0) < MIN_SIMILARITY_THRESHOLD
        has_admitted_case = (
            result.get("university"),
            result.get("major"),
        ) in admitted_combinations

        if is_cross_major and not has_admitted_case:
            original_prob = result_copy.get("probability", 0.0)
            if original_prob is not None:
                adjusted_prob = clip_probability(original_prob) * CROSS_MAJOR_PENALTY_FACTOR
                result_copy["probability"] = clip_probability(adjusted_prob)

        adjusted_results.append(result_copy)

    return adjusted_results
