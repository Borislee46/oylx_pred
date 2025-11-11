from functools import lru_cache

import pandas as pd

from src.pages.prediction.prediction_types import PredictionInput
from src.pages.prediction.prediction_utils import (
    get_cached_major_similarities_batch,
    get_school_major_details,
    get_valid_school_major_set,
    has_school_major_details,
)
from src.pages.prediction.result_modifier.config import (
    AGENT_MIN_BALANCE_DIFF,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    UNIVERSITY_COUNT_THRESHOLD,
    USER_SPECIFIED_LARGE_RANGE_TOP_N,
    USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD,
    USER_SPECIFIED_MEDIUM_RANGE_TOP_N,
    USER_SPECIFIED_SMALL_RANGE_THRESHOLD,
)
from src.pages.prediction.result_modifier.ranker import (
    adjust_similarity_results_with_agent,
    get_cross_major_recommendations,
    get_similar_major_recommendations,
)
from src.pages.prediction.result_modifier.similarity_adjuster import (
    adjust_similarity_score,
)
from src.utils.logger import setup_logger

boundary_processor_logger = setup_logger("page3", "prediction")


@lru_cache(maxsize=1)
def _get_cached_details_df():
    return get_school_major_details(None, None, return_df=True)


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    target_universities = input_data.get("target_universities", []) or []
    target_majors = input_data.get("target_majors", []) or []

    universities_to_consider = target_universities or all_universities_target
    majors_to_consider = target_majors or all_majors_target

    estimated_total = len(universities_to_consider) * len(majors_to_consider)

    if estimated_total <= 100:
        valid_combinations = [
            (univ, major)
            for univ in universities_to_consider
            for major in majors_to_consider
            if has_school_major_details(univ, major)
        ]
    else:
        valid_set = get_valid_school_major_set()
        if not valid_set:
            return [], {"combination_count": 0}

        valid_universities = set()
        valid_majors = set()
        for key in valid_set:
            parts = key.split("|", 1)
            if len(parts) == 2:
                valid_universities.add(parts[0])
                valid_majors.add(parts[1])

        universities_filtered = [u for u in universities_to_consider if u in valid_universities]
        majors_filtered = [m for m in majors_to_consider if m in valid_majors]

        if not universities_filtered or not majors_filtered:
            return [], {"combination_count": 0}

        valid_combinations = [
            (univ, major)
            for univ in universities_filtered
            for major in majors_filtered
            if f"{univ}|{major}" in valid_set
        ]

    combination_count = len(valid_combinations)

    meta = {"combination_count": combination_count}

    return valid_combinations, meta


def _filter_part_time_majors(results: list) -> list:
    if not results:
        return results

    return [
        result
        for result in results
        if not (
            "part" in result.get("major", "").lower() and "time" in result.get("major", "").lower()
        )
    ]


def _attach_chinese_names_batch(results: list, details_df_full: pd.DataFrame | None = None) -> list:
    if not results:
        return results

    if details_df_full is None:
        details_df_full = _get_cached_details_df()

    if (
        details_df_full is None
        or not isinstance(details_df_full, pd.DataFrame)
        or details_df_full.empty
    ):
        for res in results:
            res["chinese_name"] = ""
            res["faculty"] = ""
        return results

    query_data = [
        {"学校": r["university"], "专业英文名称": r["major"]}
        for r in results
        if isinstance(r, dict)
    ]
    if not query_data:
        for res in results:
            res["chinese_name"] = ""
            res["faculty"] = ""
        return results

    query_df = pd.DataFrame(query_data).drop_duplicates()

    merged_df = pd.merge(query_df, details_df_full, on=["学校", "专业英文名称"], how="left")

    cn_map = pd.Series(
        merged_df.get("专业中文名称", "").fillna("").values,
        index=pd.MultiIndex.from_frame(merged_df[["学校", "专业英文名称"]]),
    ).to_dict()

    faculty_map = pd.Series(
        merged_df.get("专业大类", "").fillna("").values,
        index=pd.MultiIndex.from_frame(merged_df[["学校", "专业英文名称"]]),
    ).to_dict()

    for res in results:
        key = (res.get("university"), res.get("major"))
        res["chinese_name"] = cn_map.get(key, "")
        res["faculty"] = faculty_map.get(key, "")

    return results


def _calculate_and_attach_similarities(
    valid_results: list, background_major: str, bg_target_similarity_cache: dict
) -> list:
    if not valid_results:
        return valid_results

    bg_major_clean = str(background_major).strip() if background_major else ""

    if not bg_major_clean:
        for res in valid_results:
            res["similarity"] = 0.0
        return valid_results

    similarity_pairs = []
    valid_indices = []

    for i, result in enumerate(valid_results):
        target_major = str(result.get("major", "")).strip()
        if target_major:
            similarity_pairs.append((target_major, bg_major_clean))
            valid_indices.append(i)

    if similarity_pairs:
        batch_similarities = get_cached_major_similarities_batch(
            similarity_pairs, cache=bg_target_similarity_cache
        )

        for idx, similarity in zip(valid_indices, batch_similarities):
            result = valid_results[idx]
            target_major = result.get("major", "")

            adjusted_similarity = adjust_similarity_score(
                background_major=bg_major_clean,
                target_major=target_major,
                similarity=similarity,
            )
            result["similarity"] = adjusted_similarity

    for result in valid_results:
        if "similarity" not in result:
            result["similarity"] = 0.0

    return valid_results


def _get_user_specified_results(
    results: list, user_specified_combinations: list[tuple[str, str]] | None
) -> list:
    if not user_specified_combinations or not results:
        return []

    specified_set = set(user_specified_combinations)
    specified_results = [
        res for res in results if (res.get("university"), res.get("major")) in specified_set
    ]

    if not specified_results:
        return []

    combination_count = len(user_specified_combinations)

    if combination_count <= USER_SPECIFIED_SMALL_RANGE_THRESHOLD:
        return specified_results

    if combination_count <= USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD:
        specified_results.sort(key=lambda x: x.get("probability", 0), reverse=True)
        return specified_results[:USER_SPECIFIED_MEDIUM_RANGE_TOP_N]

    filtered = [
        res for res in specified_results if res.get("similarity", 0.0) >= MIN_SIMILARITY_THRESHOLD
    ]
    filtered.sort(key=lambda x: x.get("probability", 0), reverse=True)
    return filtered[:USER_SPECIFIED_LARGE_RANGE_TOP_N]


def process_prediction_results(
    results: list,
    background_major: str,
    bg_target_similarity_cache: dict,
    num_target_universities: int,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
    background_faculty: str | None = None,
):
    if not results:
        return [], [], []

    filtered_results = _filter_part_time_majors(results)
    if not filtered_results:
        return [], [], []

    results_with_similarity = _calculate_and_attach_similarities(
        filtered_results, background_major, bg_target_similarity_cache
    )

    details_df_full = _get_cached_details_df()
    results_with_similarity = _attach_chinese_names_batch(
        results_with_similarity, details_df_full=details_df_full
    )

    final_user_specified_results = _get_user_specified_results(
        results_with_similarity, user_specified_combinations
    )

    if background_faculty:
        from src.pages.prediction.school_combination_optimizer_algorithm.filters import (
            filter_schools_by_faculty_rules,
        )

        results_for_recommendations = filter_schools_by_faculty_rules(
            results_with_similarity, background_faculty
        )
    else:
        results_for_recommendations = results_with_similarity

    top_similarity_results = get_similar_major_recommendations(
        results_for_recommendations, num_target_universities
    )

    top_cross_major_results = get_cross_major_recommendations(
        results_for_recommendations, background_major, cases_df, user_specified_combinations
    )

    balance_diff = len(top_cross_major_results) - len(top_similarity_results)

    boundary_processor_logger.info(
        f"[边界处理] 结果统计 - 相似专业: {len(top_similarity_results)}, "
        f"跨专业: {len(top_cross_major_results)}, 平衡差: {balance_diff}, "
        f"阈值: {AGENT_MIN_BALANCE_DIFF}"
    )

    if abs(balance_diff) >= AGENT_MIN_BALANCE_DIFF:
        current_threshold = MIN_SIMILARITY_THRESHOLD
        if num_target_universities > 0 and num_target_universities <= UNIVERSITY_COUNT_THRESHOLD:
            current_threshold = HIGHER_SIMILARITY_THRESHOLD

        boundary_processor_logger.info(
            f"[边界处理] 触发条件满足 - 平衡差绝对值: {abs(balance_diff)}, "
            f"当前相似度阈值: {current_threshold}, 背景专业: {background_major}, "
            f"目标院校数量: {num_target_universities}"
        )

        agent = None
        if cases_df is not None and background_major:
            from src.agent.boundary_case_agent import BoundaryCaseAgent

            agent = BoundaryCaseAgent(cases_df=cases_df)
            boundary_processor_logger.info("[边界处理] Agent 实例创建成功")

        if agent:
            original_sim_count = len(top_similarity_results)
            original_cross_count = len(top_cross_major_results)

            top_similarity_results = adjust_similarity_results_with_agent(
                top_similarity_results,
                results_for_recommendations,
                balance_diff,
                background_major,
                current_threshold,
                agent,
            )

            sim_set = {(r.get("university"), r.get("major")) for r in top_similarity_results}
            top_cross_major_results = [
                r
                for r in top_cross_major_results
                if (r.get("university"), r.get("major")) not in sim_set
            ]

            adjusted_sim_count = len(top_similarity_results)
            adjusted_cross_count = len(top_cross_major_results)

            removed_from_cross = original_cross_count - adjusted_cross_count

            boundary_processor_logger.info(
                f"[边界处理] 调整完成 - 相似专业: {original_sim_count} -> {adjusted_sim_count} "
                f"(变化: {adjusted_sim_count - original_sim_count:+d}), "
                f"跨专业: {original_cross_count} -> {adjusted_cross_count} "
                f"(移除重复: {removed_from_cross})"
            )

            if removed_from_cross > 0:
                boundary_processor_logger.info(
                    f"[边界处理] 从跨专业结果中移除了 {removed_from_cross} 个重复案例"
                )
        else:
            boundary_processor_logger.warning(
                "[边界处理] Agent 未创建，跳过边界处理 - "
                f"cases_df: {cases_df is not None}, background_major: {bool(background_major)}"
            )
    else:
        boundary_processor_logger.debug(
            f"[边界处理] 未触发 - 平衡差绝对值 {abs(balance_diff)} < 阈值 {AGENT_MIN_BALANCE_DIFF}"
        )

    return top_similarity_results, top_cross_major_results, final_user_specified_results
