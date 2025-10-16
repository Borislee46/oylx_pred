from typing import Any

import pandas as pd

from src.pages.prediction.prediction_types import PredictionInput
from src.pages.prediction.prediction_utils import (
    get_cached_major_similarities_batch,
    get_school_major_details,
    get_valid_school_major_set,
    has_school_major_details,
)
from src.pages.prediction.result_modifier.ranker import (
    get_cross_major_recommendations,
    get_similar_major_recommendations,
)
from src.pages.prediction.result_modifier.similarity_adjuster import (
    adjust_similarity_score,
)


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
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
        prefix_patterns = [f"{univ}|" for univ in universities_to_consider]
        valid_combinations = [
            (univ, major)
            for univ, prefix in zip(universities_to_consider, prefix_patterns)
            for major in majors_to_consider
            if prefix + major in valid_set
        ]

    combination_count = len(valid_combinations)
    if combination_count == 0:
        message = "根据您的选择，没有找到有效的学校和专业组合进行分析"
    elif combination_count > 100:
        message = f"将分析 {combination_count} 个学校和专业组合，这可能需要一些时间"
    else:
        message = f"将分析 {combination_count} 个学校和专业组合"

    meta = {
        "combination_count": combination_count,
        "combination_message": message,
    }

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
        details_df_full = get_school_major_details(None, None, return_df=True)

    if details_df_full is None or details_df_full.empty:
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

        for idx, similarity in zip(valid_indices, batch_similarities, strict=False):
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
    return [res for res in results if (res.get("university"), res.get("major")) in specified_set]


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

    details_df_full = get_school_major_details(None, None, return_df=True)
    results_with_similarity = _attach_chinese_names_batch(
        results_with_similarity, details_df_full=details_df_full
    )

    if background_faculty:
        from src.pages.prediction.school_combination_optimizer_algorithm.filters import (
            filter_schools_by_faculty_rules,
        )

        results_with_similarity = filter_schools_by_faculty_rules(
            results_with_similarity, background_faculty
        )

    top_similarity_results = get_similar_major_recommendations(
        results_with_similarity, num_target_universities
    )

    top_cross_major_results = get_cross_major_recommendations(
        results_with_similarity, background_major, cases_df, user_specified_combinations
    )

    final_user_specified_results = _get_user_specified_results(
        results_with_similarity, user_specified_combinations
    )

    return top_similarity_results, top_cross_major_results, final_user_specified_results
