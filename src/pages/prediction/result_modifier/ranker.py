from typing import Any

import pandas as pd

from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
    CROSS_MAJOR_SIMILARITY_MIN,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    TOP_N_RECOMMENDATIONS,
    UNIVERSITY_COUNT_THRESHOLD,
)
from src.pages.prediction.result_modifier.utils import clip_probability


def get_similar_major_recommendations(
    results_with_similarity: list[dict[str, Any]], num_target_universities: int
) -> list[dict[str, Any]]:
    """
    获取相似专业的推荐结果

    Args:
        results_with_similarity: 包含相似度分数的结果列表
        num_target_universities: 目标大学数量

    Returns:
        排序后的推荐结果列表，按概率降序排列
    """
    if not results_with_similarity:
        return []

    current_threshold = MIN_SIMILARITY_THRESHOLD
    if num_target_universities > 0 and num_target_universities <= UNIVERSITY_COUNT_THRESHOLD:
        current_threshold = HIGHER_SIMILARITY_THRESHOLD

    filtered_by_similarity = [
        res for res in results_with_similarity if res.get("similarity", 0.0) >= current_threshold
    ]

    if not filtered_by_similarity:
        return []

    if len(filtered_by_similarity) <= TOP_N_RECOMMENDATIONS:
        filtered_by_similarity.sort(key=lambda x: x.get("similarity", 0.0), reverse=True)
        top_candidates = filtered_by_similarity
    else:
        import heapq

        top_candidates = heapq.nlargest(
            TOP_N_RECOMMENDATIONS,
            filtered_by_similarity,
            key=lambda x: x.get("similarity", 0.0),
        )

    for c in top_candidates:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    top_candidates.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    return top_candidates


def get_cross_major_recommendations(
    results_with_similarity: list[dict[str, Any]],
    background_major: str,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    获取跨专业推荐结果（仅包含有历史成功案例的跨专业申请）

    Args:
        results_with_similarity: 包含相似度分数的结果列表
        background_major: 背景专业
        cases_df: 历史案例数据框
        user_specified_combinations: 用户指定的组合列表（未使用，保留接口兼容性）

    Returns:
        排序后的跨专业推荐结果列表，按概率降序排列
    """
    if not background_major or not results_with_similarity:
        return []

    bg_major_clean = str(background_major).strip()
    admitted_combinations = get_admitted_combinations_from_dataframe(cases_df, bg_major_clean)

    if admitted_combinations:
        # 修正逻辑：筛选跨专业（相似度低于阈值）但有历史成功案例的申请
        # 相似度应该在合理范围内（大于最小值但小于阈值）
        admitted_results = [
            res
            for res in results_with_similarity
            if (res.get("university"), res.get("major")) in admitted_combinations
            and res.get("similarity", 1.0) < MIN_SIMILARITY_THRESHOLD
            and res.get("similarity", 0.0) >= CROSS_MAJOR_SIMILARITY_MIN
        ]

        if admitted_results:
            # 先按相似度升序排序（选择最不相似但有成功案例的）
            admitted_results.sort(key=lambda x: x.get("similarity", 1.0))
            top_least_similar = admitted_results[:TOP_N_RECOMMENDATIONS]
            # 再按概率降序排序
            top_least_similar.sort(key=lambda x: x.get("probability", 0), reverse=True)
            return top_least_similar
    return []
