from typing import Any

from src.pages.prediction.result_modifier.config import (
    PROFESSIONAL_MAJORS,
    PROFESSIONAL_REDUCTION_FACTOR,
    PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR,
)
from src.pages.prediction.result_modifier.utils import clip_probability

# 预计算专业关键词的小写版本，避免重复转换
_PROFESSIONAL_MAJORS_LOWER = [m.lower() for m in PROFESSIONAL_MAJORS]


def adjust_for_professional_majors(
    results: list[dict[str, Any]],
    internship_count: int,
    user_specified_majors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    针对职业型专业进行调整：如果用户没有实习经历，则降低职业型专业的概率

    Args:
        results: 预测结果列表
        internship_count: 实习数量
        user_specified_majors: 用户指定的专业列表

    Returns:
        调整后的结果列表
    """
    if not results:
        return []

    if internship_count > 0:
        return results

    adjusted_results = []
    # 预计算用户指定专业的小写版本
    user_majors_lower = [m.lower() for m in user_specified_majors] if user_specified_majors else []

    for result in results:
        target_major = result.get("major")
        if not target_major:
            adjusted_results.append(result)
            continue

        target_major_lower = target_major.lower()
        # 使用预计算的小写专业关键词列表
        is_professional = any(
            prof_major in target_major_lower for prof_major in _PROFESSIONAL_MAJORS_LOWER
        )

        if not is_professional:
            adjusted_results.append(result)
            continue

        is_user_specified = any(
            spec_major in target_major_lower for spec_major in user_majors_lower
        )

        result_copy = result.copy()
        p = float(result_copy.get("probability", 0.0) or 0.0)
        factor = (
            PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR
            if is_user_specified
            else PROFESSIONAL_REDUCTION_FACTOR
        )
        p = clip_probability(p * factor)
        result_copy["probability"] = p
        adjusted_results.append(result_copy)

    return adjusted_results
