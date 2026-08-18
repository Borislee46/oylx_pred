import numba
import pandas as pd

from src.adjustment.config import (
    COMPREHENSIVE_SCORE_BOOST_THRESHOLD,
    COMPREHENSIVE_SCORE_WEIGHTS,
    DEFAULT_UNIVERSITY_DIFFICULTY_ORDER,
    GPA_MINIMUM,
    GPA_PENALTY_MAX_COEFFICIENT,
    GPA_PENALTY_QUADRATIC_COEFFICIENT,
    GPA_PENALTY_SEVERE_THRESHOLD,
    LANGUAGE_MINIMUM,
    LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER,
    LANGUAGE_PENALTY_SEVERE_THRESHOLD,
    LANGUAGE_PENALTY_SIGMOID_STEEPNESS,
    SELECTION_SCORE_BOOST_FACTOR,
    UNIVERSITY_DIFFICULTY_CONFIG_PATH,
)
from src.adjustment.utils import (
    cache_data,
    compute_dataframe_hash,
)
from src.utils.logger import setup_logger
from src.utils.numeric import sigmoid_k
from src.utils.schools.difficulty import get_university_difficulty_order
from src.utils.schools.level_service import get_school_level_service

logger = setup_logger("page3", "prediction")


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
def _compute_language_penalty_sigmoid(
    score: float,
    minimum: float,
    pass_line: float,
    severe_threshold: float,
    k: float,
) -> float:
    if score >= pass_line:
        return 0.0
    if score < minimum:
        return severe_threshold

    t = (pass_line - score) / (pass_line - minimum)
    penalty = severe_threshold * sigmoid_k(t, k, 0.5)
    return penalty


@cache_data(
    show_spinner=False,
    # hash_key 是权威缓存键；hash_funcs 用采样指纹替代 streamlit 对整帧的
    # O(n) 内容哈希（80k 行实测 ~11ms → ~1ms），并保留对采样内容的二次校验。
    hash_funcs={pd.DataFrame: compute_dataframe_hash},
)
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

    logger.debug(
        "案例统计计算完成 | gpa_mean=%.2f gpa_std=%.4f lang_mean=%.4f lang_std=%.4f lang_pass=%.4f",
        stats["gpa_mean"],
        stats["gpa_std"],
        stats["language_mean"],
        stats["language_std"],
        stats["language_pass_line"],
    )
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

    def _calculate_comprehensive_score(
        self, gpa: float, language_score: float, background_university: str | None
    ) -> float:
        gpa_score = sigmoid_k(gpa, 3.0, 3.3)
        lang_score = sigmoid_k(language_score, 15.0, 0.72)

        service = get_school_level_service()
        school_score = service.get_school_score(background_university)

        w_gpa, w_lang, w_school = COMPREHENSIVE_SCORE_WEIGHTS
        total_score = w_gpa * gpa_score + w_lang * lang_score + w_school * school_score
        return total_score

    def calculate_selection_score(
        self,
        similarity: float,
        target_university: str,
        gpa: float | None = None,
        language_score: float | None = None,
        background_university: str | None = None,
        *,
        comp_score: float | None = None,
    ) -> float:
        if comp_score is None:
            if gpa is None or language_score is None:
                return similarity
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
        return _compute_language_penalty_sigmoid(
            language_score,
            self.language_minimum,
            self.language_pass_line,
            LANGUAGE_PENALTY_SEVERE_THRESHOLD,
            LANGUAGE_PENALTY_SIGMOID_STEEPNESS,
        )

    def get_penalties(self, gpa: float, language_score: float) -> dict[str, float]:
        return {
            "gpa": self._calculate_gpa_penalty(gpa),
            "language": self._calculate_language_penalty(language_score),
        }
