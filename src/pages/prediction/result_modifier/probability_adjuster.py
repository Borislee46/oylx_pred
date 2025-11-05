import threading
from typing import Any

import numpy as np
import pandas as pd

from src.pages.prediction.prediction_utils import normalize_language_score
from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
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
    PROBABILITY_ADJUSTER_CACHE_SIZE,
    PROBABILITY_ADJUSTMENT_THRESHOLD,
    PROBABILITY_EXTREME_STD_MULTIPLIER,
    PROBABILITY_MIN_VALUE,
)
from src.pages.prediction.result_modifier.utils import (
    clip_probability,
    generate_content_hash,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class ProbabilityAdjuster:
    """概率调整器，基于历史案例统计对GPA和语言分数进行保守惩罚"""

    _stats_cache: dict[str, dict[str, Any]] = {}
    _cache_lock = threading.Lock()  # 保护缓存的线程锁

    def __init__(self, cases_df: pd.DataFrame):
        """
        初始化概率调整器

        Args:
            cases_df: 历史案例数据框
        """
        cache_key = self._generate_data_hash(cases_df)

        with self._cache_lock:
            if cache_key in self._stats_cache:
                self._load_cached_statistics(cache_key)
                self.cases_df = None
            else:
                self.cases_df = cases_df.copy()
                self._calculate_statistics()
                self._cache_statistics(cache_key)
                self.cases_df = None

        self.gpa_minimum = GPA_MINIMUM
        self.language_minimum = LANGUAGE_MINIMUM

    def _generate_data_hash(self, cases_df: pd.DataFrame) -> str:
        """
        生成数据框的哈希键用于缓存

        Args:
            cases_df: 数据框

        Returns:
            数据框的哈希键
        """
        try:
            data_summary = {
                "shape": cases_df.shape,
                "columns": sorted(cases_df.columns.tolist()),
                "gpa_stats": (
                    dict(sorted(cases_df["gpa"].describe().to_dict().items()))
                    if "gpa" in cases_df.columns
                    else {}
                ),
            }

            if "toefl" in cases_df.columns:
                data_summary["toefl_stats"] = dict(
                    sorted(cases_df["toefl"].describe().to_dict().items())
                )
            if "ielts" in cases_df.columns:
                data_summary["ielts_stats"] = dict(
                    sorted(cases_df["ielts"].describe().to_dict().items())
                )

            data_str = str(sorted(data_summary.items()))
            return generate_content_hash(data_str)
        except (KeyError, AttributeError, ValueError) as e:
            logger.warning(f"生成数据哈希失败，使用fallback: {str(e)}")
            return f"fallback_{hash(str(cases_df.shape))}"
        except Exception as e:
            logger.error(f"生成数据哈希时发生未知错误: {str(e)}", exc_info=True)
            return f"fallback_{hash(str(cases_df.shape))}"

    def _load_cached_statistics(self, cache_key: str):
        """从缓存加载统计信息"""
        cached_stats = self._stats_cache[cache_key]
        self.gpa_mean = cached_stats["gpa_mean"]
        self.gpa_std = cached_stats["gpa_std"]
        self.language_mean = cached_stats["language_mean"]
        self.language_std = cached_stats["language_std"]
        self.language_pass_line = cached_stats["language_pass_line"]

    def _cache_statistics(self, cache_key: str):
        """缓存统计信息，如果超过大小限制则删除最旧的条目"""
        self._stats_cache[cache_key] = {
            "gpa_mean": self.gpa_mean,
            "gpa_std": self.gpa_std,
            "language_mean": self.language_mean,
            "language_std": self.language_std,
            "language_pass_line": self.language_pass_line,
        }

        if len(self._stats_cache) > PROBABILITY_ADJUSTER_CACHE_SIZE:
            oldest_key = next(iter(self._stats_cache))
            del self._stats_cache[oldest_key]

    def _calculate_statistics(self):
        """计算GPA和语言分数的统计信息"""
        try:
            self.gpa_mean = float(np.nan_to_num(self.cases_df["gpa"].mean(), nan=0.0))
            self.gpa_std = float(np.nan_to_num(self.cases_df["gpa"].std(), nan=0.0))
            if self.gpa_std == 0:
                self.gpa_std = 1e-6
        except (KeyError, AttributeError) as e:
            logger.warning(f"计算GPA统计信息失败: {str(e)}")
            self.gpa_mean = 0.0
            self.gpa_std = 1e-6

        try:
            normalized_toefl = (
                (self.cases_df["toefl"].apply(lambda x: normalize_language_score(x, "托福")))
                if "toefl" in self.cases_df.columns
                else pd.Series(0, index=self.cases_df.index)
            )
            normalized_ielts = (
                (self.cases_df["ielts"].apply(lambda x: normalize_language_score(x, "雅思")))
                if "ielts" in self.cases_df.columns
                else pd.Series(0, index=self.cases_df.index)
            )

            self.cases_df["normalized_language_score"] = np.where(
                self.cases_df["toefl"].notna(), normalized_toefl, normalized_ielts
            )

        except (KeyError, AttributeError, ValueError) as e:
            logger.warning(f"计算语言分数统计信息失败: {str(e)}")
            self.cases_df["normalized_language_score"] = 0.0
        except Exception as e:
            logger.error(f"计算语言分数统计信息时发生未知错误: {str(e)}", exc_info=True)
            self.cases_df["normalized_language_score"] = 0.0

        self.language_mean = float(
            np.nan_to_num(self.cases_df["normalized_language_score"].mean(), nan=0.0)
        )
        self.language_std = float(
            np.nan_to_num(self.cases_df["normalized_language_score"].std(), nan=0.0)
        )
        if self.language_std == 0:
            self.language_std = 1e-6
        self.language_pass_line = (
            self.language_mean - LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER * self.language_std
        )

    def _calculate_gpa_penalty(self, gpa: float) -> float:
        """
        计算GPA惩罚系数

        Args:
            gpa: GPA分数

        Returns:
            惩罚系数 (0.0-1.0)
        """
        if gpa < self.gpa_minimum:
            return GPA_PENALTY_SEVERE_THRESHOLD
        if gpa >= self.gpa_mean:
            return 0.0
        gpa_gap = (self.gpa_mean - gpa) / self.gpa_std
        return min(GPA_PENALTY_MAX_COEFFICIENT, GPA_PENALTY_QUADRATIC_COEFFICIENT * gpa_gap**2)

    def _calculate_language_penalty(self, language_score: float) -> float:
        """
        计算语言分数惩罚系数

        Args:
            language_score: 归一化后的语言分数

        Returns:
            惩罚系数 (0.0-1.0)
        """
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
        """
        调整概率值，基于GPA和语言分数进行惩罚

        Args:
            probability: 原始概率值
            gpa: GPA分数
            language_score: 归一化后的语言分数
            background_university_name: 背景大学名称（未使用，保留接口兼容性）

        Returns:
            调整后的概率值
        """
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

    @classmethod
    def clear_stats_cache(cls):
        """清空统计信息缓存"""
        with cls._cache_lock:
            cls._stats_cache.clear()

    @classmethod
    def get_cache_stats(cls) -> dict[str, Any]:
        """获取缓存统计信息"""
        with cls._cache_lock:
            return {
                "cache_size": len(cls._stats_cache),
                "cache_limit": PROBABILITY_ADJUSTER_CACHE_SIZE,
                "cached_keys": list(cls._stats_cache.keys()),
            }


def penalize_cross_major_without_cases(
    user_specified_results: list[dict[str, Any]],
    background_major: str,
    cases_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    对没有历史成功案例的跨专业申请进行惩罚

    Args:
        user_specified_results: 用户指定的结果列表
        background_major: 背景专业
        cases_df: 历史案例数据框

    Returns:
        调整后的结果列表
    """
    if not user_specified_results or not background_major or cases_df is None or cases_df.empty:
        return user_specified_results

    try:
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
    except (KeyError, AttributeError, ValueError) as e:
        logger.warning(f"跨专业惩罚处理失败: {str(e)}")
        return user_specified_results
    except Exception as e:
        logger.error(f"跨专业惩罚处理时发生未知错误: {str(e)}", exc_info=True)
        return user_specified_results
