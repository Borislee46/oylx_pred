# =============================================================================
# 概率调整器 (Probability Adjuster)
# ─────────────────────────────────────────────────────────────────────────────
# XGBoost 输出的概率经过校准，但仍有两个问题需要后处理：
#
# 问题1：模型对极端低分没有"地板"意识。
#   纯数据驱动下，GPA=2.1 的学生申 Tier-1 学校可能给 15% 概率，
#   但行业共识是这种组合几乎不可能。需要基于分布的业务惩罚修正。
#
# 问题2：模型只知道"录取/不录"，但不知道"录取了什么专业"。
#   同校跨专业录取模式完全不同，相似专业和跨专业需要差异化处理。
#   （跨专业惩罚在 adjustment_pipeline.py 中实现）
#
# 设计原则：
#   - 所有惩罚基于训练数据的分布（mean/std），不是绝对阈值
#   - 惩罚力度呈非线性（quadratic），轻微偏差影响小，严重偏差快速放大
#   - 极端情况有硬地板（probability → 0.001），不给虚假希望
#   - 用 numba jit 编译所有核心计算以保证性能（每个预测轮次调用数百次）
# =============================================================================

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
    LANGUAGE_PENALTY_LEVEL_3_5_THRESHOLD,
    LANGUAGE_PENALTY_LEVEL_3_MULTIPLIER,
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


# ─────────────────────────────────────────────────────────────────────────────
# 为什么用 Sigmoid 做分数映射？
# ─────────────────────────────────────────────────────────────────────────────
# GPA (2.0-4.0) 和语言成绩 (0.6-1.0 normalized) 是线性增加的，
# 但它们对录取概率的贡献是非线性的：
#   GPA 3.7→3.9 与 GPA 2.5→2.7 的增量含金量不同。
# Sigmoid 将原始分数映射到 [0,1] 区间，自动编码边际递减效应：
#   高分段的提升越来越小，低分段每提升一点都有较大收益。
# k 控制陡峭度，x0 控制中心点（"及格线"附近的值）。
# ─────────────────────────────────────────────────────────────────────────────
@numba.njit(cache=True)
def _fast_sigmoid(x: float, k: float, x0: float) -> float:
    return 1.0 / (1.0 + math.exp(-k * (x - x0)))


# ─────────────────────────────────────────────────────────────────────────────
# GPA 惩罚：二次函数 (Quadratic Penalty)
# ─────────────────────────────────────────────────────────────────────────────
# penalty = min(max_coeff, quad_coeff * z_score²)
# 其中 z_score = (mean - gpa) / std  （只在 gpa < mean 时惩罚）
#
# 为什么用二次函数？
# 1. 轻微低于均值（z < 1）：惩罚几乎不可见（0.15 × 1² = 15%），
#    因为数据本身存在方差，低0.5个标准差并不意味着什么。
# 2. 明显低于均值（z > 1.5）：惩罚快速上升（0.15 × 2.25 = 34%），
#    信号增强 — 确实比较低。
# 3. 严重低于均值（z > 2.3）：达到 max_coeff 上限（80%），
#    几乎确定被拒，不给虚假期望。
#
# 为什么不是线性惩罚？
#   线性会让 2.5→2.7 和 3.5→3.7 的增量惩罚相同，
#   但实际上低分段多加 0.1 分和接近均值时多加 0.1 分意义完全不同。
#   二次函数让"接近均值"的同学几乎不受罚，"远低于均值"的被快速放大。
#
# 为什么是 z_score 而非绝对阈值？
#   录取标准因学校/专业/年份而异。Z-score 基于当前案例分布，
#   自动适配不同申请季、不同竞争环境的现象 — 这是 data-driven 的思路。
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 语言惩罚：分段阶梯 (Tiered Penalty)
# ─────────────────────────────────────────────────────────────────────────────
# 与 GPA 不同，语言成绩有明确的"门槛效应"：
#   - 达到 pass_line（均值-0.5std，约0.8分归一化 ≈ IELTS 7.2）基本不受影响
#   - 低于 pass_line 分段惩罚：L1(85%)、L2(70%)、L3(40%)
#   - 低于 minimum（低于~0.6 ≈ IELTS 5.4）直接严重惩罚（95%）
#
# 为什么用阶梯而非连续函数？
# 录取审核中语言成绩有实际的门槛：
#   - 官网最低要求通常 IELTS 6.0-6.5（归一化 0.67-0.72）
#   - "有竞争力"通常 IELTS 7.0+（归一化 0.78+）
#   在门槛之上和之下语义完全不同，不是连续的渐变。
#   阶梯函数比二次函数更贴近实际审核行为。
#
# 为什么 pass_line = mean - 0.5*std？
#   这是"竞争力线"的统计近似 — 取均值下移半个标准差，
#   保证大部分录取案例在此之上，而又不会太松。
#   纯均值（mean）太严（50%学生在此之下），
#   纯最小值太松。0.5个标准差是经验平衡点。
# ─────────────────────────────────────────────────────────────────────────────
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
    l3_mult: float,
    l3_thresh: float,
    l3_5_thresh: float,
) -> float:
    if score < minimum:
        return severe_threshold
    if score >= pass_line:
        return 0.0

    if score < (pass_line - l1_mult * std):
        return l1_thresh
    elif score < (pass_line - l2_mult * std):
        return l2_thresh
    elif score < (pass_line - l3_mult * std):
        return l3_thresh
    else:
        return l3_5_thresh


# ─────────────────────────────────────────────────────────────────────────────
# 组合惩罚与极端值处理
# ─────────────────────────────────────────────────────────────────────────────
# 惩罚顺序：prob → ×(1-gpa_penalty) → ×(1-lang_penalty) → clip
# 两者相互独立叠加（相乘），因为 GPA 弱和语言弱是两件独立的事。
#
# 极端值检测（is_extreme_gpa/lang）：
#   当 GPA 或语言低于 mean - 2*std 时，直接设 prob = MIN_VALUE (0.001)。
#   2σ 原则：正态分布下约 2.5% 的情况。这些是统计异常值，
#   训练数据中几乎不存在这类案例，模型预测置信度极低，
#   后处理中直接"地板化"比保留不可靠的概率更安全。
#
# 为什么不是乘完就完了？
#   adjusted < threshold 的 check 确保即使惩罚不大，
#   非常低的概率（<1%）也会被极端值逻辑重新评估。
#   但普通低分（非极端）不会被错误降到 MIN_VALUE。
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# 案例库统计计算
# ─────────────────────────────────────────────────────────────────────────────
# 从历史录取数据中计算 GPA 和语言成绩的分布参数。
# 这些统计量是概率调整惩罚计算的基准：
#   - mean: 判断学生相对于历史平均水平的位置
#   - std: 标准化偏差大小
#
# std 的 floor = 1e-6：防止除零报错。如果某批次数据 GPA 完全相同，
# z_score 计算 ∝ (mean-gpa)/std 会除零。1e-6 相当于约 0 方差时
# z_score 极大 → 一律按 max_coeff 惩罚，行为合理且安全。
# ─────────────────────────────────────────────────────────────────────────────
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

    # ─────────────────────────────────────────────────────────────────────────
    # 综合评分 (Comprehensive Score)：0.4 GPA + 0.3 语言 + 0.3 院校
    # ─────────────────────────────────────────────────────────────────────────
    # 为什么是这个权重？
    # - GPA (40%): 录取预测中最重要的单一信号。3年学业表现 > 一次考试。
    # - 语言 (30%): 次重要的学术能力信号，但精确度不如 GPA。
    # - 本科院校 (30%): C9/985/211/双非的等级信号，反映学术训练基础。
    #
    # 3个维度均通过 sigmoid 归一化到 [0,1] 再加权，保证量纲统一。
    # Sigmoid 参数：
    #   GPA: k=3.0, x0=3.3 — 3.3 (B+) 后开始饱和，3.7+ 接近满分
    #   语言: k=15.0, x0=0.72 — 约 IELTS 6.5/TOEFL 86 附近线性区，高分饱和
    # k 值选择依据：让中位学生落在 sigmoid 的线性区（最有区分度），
    # 极高/极低分饱和（区分度低，因为极少发生）。
    # ─────────────────────────────────────────────────────────────────────────
    def _calculate_comprehensive_score(
        self, gpa: float, language_score: float, background_university: str | None
    ) -> float:
        gpa_score = _fast_sigmoid(gpa, 3.0, 3.3)
        lang_score = _fast_sigmoid(language_score, 15.0, 0.72)

        service = get_school_level_service()
        school_score = service.get_school_score(background_university)

        total_score = 0.4 * gpa_score + 0.3 * lang_score + 0.3 * school_score
        return total_score

    # ─────────────────────────────────────────────────────────────────────────
    # 选择分数 (Selection Score)：相似度 × 竞争力 boost
    # ─────────────────────────────────────────────────────────────────────────
    # 基础相似度只衡量"专业名称有多接近"，不评估"学生有没有实力申这所学校"。
    # 选择分数在相似度基础上叠加竞争力：
    #   comp_score > 0.6 且目标学校有难度分数时 → boost = 0.3 × comp × diff
    #
    # 效果：实力强的学生（comp_score 高）申难度高的学校（diff 高），
    # 该组合被放大；实力弱的学生不受影响（不触发 boost）。
    # 这相当于在推荐排序中给"值得挑战"的组合加权。
    # ─────────────────────────────────────────────────────────────────────────
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
            LANGUAGE_PENALTY_LEVEL_3_MULTIPLIER,
            LANGUAGE_PENALTY_LEVEL_3_THRESHOLD,
            LANGUAGE_PENALTY_LEVEL_3_5_THRESHOLD,
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
