from typing import Any

from src.pages.prediction.prediction_utils import get_cached_major_similarity
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    GLOBAL_MIN_SIMILARITY,
    MACAU_UNIVERSITIES,
)
from src.pages.prediction.school_combination_optimizer_algorithm.filters import (
    deduplicate_majors,
    deduplicate_universities_by_similarity,
)
from src.pages.prediction.school_combination_optimizer_algorithm.optimizer.context import (
    OptimizationContext,
)
from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
    calibrate_cross_major_probabilities,
    normalize_major_name,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def apply_all_filters(
    schools_data: list[dict[str, Any]],
    context: OptimizationContext,
    bg_target_similarity_cache: dict[str, Any],
) -> list[dict[str, Any]]:
    logger.info(
        f"_apply_all_filters开始: 输入学校数量={len(schools_data)}, "
        f"background_major={context.background_major}"
    )

    schools_data = [
        ({**s, "major_norm": normalize_major_name(s.get("major", ""))} if s else s)
        for s in schools_data
    ]
    logger.info(f"添加major_norm后，学校数量={len(schools_data)}")

    def similarity_filter(schools):
        result = deduplicate_universities_by_similarity(
            schools, context.background_major, None, MACAU_UNIVERSITIES
        )
        logger.info(f"similarity_filter: {len(schools)} -> {len(result)}")
        return result

    def similarity_filter_only(schools):
        cache = bg_target_similarity_cache
        logger.info(f"similarity_filter_only使用缓存大小: {len(cache)}")

        school_similarities = [
            (s, get_cached_major_similarity(s.get("major", ""), context.background_major, cache))
            for s in schools
        ]
        similarities = [sim for _, sim in school_similarities]
        filtered = [s for s, sim in school_similarities if sim >= GLOBAL_MIN_SIMILARITY]

        if similarities:
            logger.info(
                f"similarity_filter_only: {len(schools)} -> {len(filtered)}, "
                f"相似度范围=[{min(similarities):.3f}, {max(similarities):.3f}], "
                f"平均值={sum(similarities) / len(similarities):.3f}, "
                f"阈值={GLOBAL_MIN_SIMILARITY}"
            )
            sample_majors = [(s.get("major", ""), sim) for s, sim in school_similarities[:5]]
            if sample_majors:
                sample_info = ", ".join([f"{m[:30]}:{sim:.3f}" for m, sim in sample_majors])
                logger.info(f"前5个专业相似度示例: {sample_info}")
            if len(filtered) == 0 and len(schools) > 0:
                logger.warning(
                    f"所有学校都被过滤掉！最高相似度={max(similarities):.3f} < 阈值{GLOBAL_MIN_SIMILARITY}, "
                    f"background_major={context.background_major}"
                )
        return filtered

    def probability_calibration(schools):
        result = calibrate_cross_major_probabilities(schools, context.background_faculty)
        logger.info(f"probability_calibration: {len(schools)} -> {len(result)}")
        return result

    filters = [
        ("deduplicate_majors", deduplicate_majors, False),
        ("similarity_filter", similarity_filter, True),
        ("similarity_filter_only", similarity_filter_only, True),
        ("probability_calibration", probability_calibration, True),
    ]

    filtered_data = schools_data
    for filter_name, filter_func, continue_on_empty in filters:
        original_len = len(filtered_data)
        logger.info(f"应用过滤器 {filter_name}: 输入数量={original_len}")

        try:
            filtered_data = filter_func(filtered_data)
        except (KeyError, TypeError, ValueError) as e:
            logger.warning(f"过滤器 {filter_name} 出现错误: {type(e).__name__}: {e}")
            if not continue_on_empty:
                logger.error(f"过滤器 {filter_name} 出现关键错误，停止后续过滤器")
                break
            filtered_data = filtered_data
        except Exception as e:
            logger.error(
                f"过滤器 {filter_name} 出现未知错误: {type(e).__name__}: {e}", exc_info=True
            )
            if not continue_on_empty:
                logger.error(f"过滤器 {filter_name} 出现未知错误，停止后续过滤器")
                break
            filtered_data = filtered_data

        new_len = len(filtered_data)
        logger.info(f"过滤器 {filter_name} 完成: {original_len} -> {new_len}")

        if new_len == 0 and not continue_on_empty:
            logger.warning(f"过滤器 {filter_name} 将所有数据过滤为空，停止后续过滤器")
            break

    logger.info(f"_apply_all_filters完成: 最终学校数量={len(filtered_data)}")
    return filtered_data


def apply_post_filters(
    schools: list[dict[str, Any]],
    context: OptimizationContext,
    problem: Any,
) -> list[dict[str, Any]]:
    if not context.background_faculty or not context.adaptive_thresholds:
        return schools

    from src.pages.prediction.school_combination_optimizer_algorithm.utils import (
        clip_probability,
    )

    reach_threshold = context.adaptive_thresholds.get("target_lower", 0.0)

    def should_include(school: dict[str, Any]) -> bool:
        is_reach = clip_probability(school.get("probability", 1.0)) < reach_threshold
        if not is_reach:
            return True

        uni = school.get("university", "")
        major = school.get("major", "")
        cache_key = f"{uni}|{major}"
        target_faculty = problem.major_category_cache.get(cache_key)
        return target_faculty == context.background_faculty

    return [school for school in schools if should_include(school)]
