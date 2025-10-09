from typing import Any

import pandas as pd

from src.pages.prediction.prediction_types import PredictionInput
from src.pages.prediction.prediction_utils import (
    get_cached_major_similarities_batch,
    get_valid_school_major_set,
    has_school_major_details,
)
from src.pages.prediction.result_modifier.ranker import (
    _get_cross_major_recommendations,
    _get_similar_major_recommendations,
)
from src.pages.prediction.result_modifier.similarity_adjuster import adjust_similarity_score


def generate_prediction_combinations(
    input_data: PredictionInput, all_universities_target: list[str], all_majors_target: list[str]
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    target_universities = input_data.get("target_universities", []) or []
    target_majors = input_data.get("target_majors", []) or []
    universities_to_consider = (
        target_universities if target_universities else all_universities_target
    )
    majors_to_consider = target_majors if target_majors else all_majors_target

    valid_combinations = []
    estimated_total = len(universities_to_consider) * len(majors_to_consider)
    if estimated_total <= 100:
        for univ in universities_to_consider:
            for major in majors_to_consider:
                if has_school_major_details(univ, major):
                    valid_combinations.append((univ, major))
    else:
        valid_set = get_valid_school_major_set()
        for univ in universities_to_consider:
            prefix = f"{univ}|"
            for major in majors_to_consider:
                key = prefix + major
                if key in valid_set:
                    valid_combinations.append((univ, major))

    combination_count = len(valid_combinations)
    if combination_count > 0:
        if combination_count > 100:
            message = f"将分析 {combination_count} 个学校和专业组合，这可能需要一些时间"
        else:
            message = f"将分析 {combination_count} 个学校和专业组合"
    else:
        message = "根据您的选择，没有找到有效的学校和专业组合进行分析"

    meta = {
        "combination_count": combination_count,
        "combination_message": message,
    }

    return valid_combinations, meta


def _calculate_and_attach_similarities(
    valid_results: list, background_major: str, bg_target_similarity_cache: dict
) -> list:
    if not background_major or not valid_results:
        for res in valid_results:
            res["similarity"] = 0.0
        return valid_results

    bg_major_clean = str(background_major).strip()
    similarity_pairs = []
    valid_indices_for_sim = []

    for i, result in enumerate(valid_results):
        target_major_clean = str(result.get("major", "")).strip()
        if target_major_clean:
            similarity_pairs.append((target_major_clean, bg_major_clean))
            valid_indices_for_sim.append(i)

    if similarity_pairs:
        try:
            batch_similarities = get_cached_major_similarities_batch(
                similarity_pairs, cache=bg_target_similarity_cache
            )
            for original_idx, similarity in zip(
                valid_indices_for_sim, batch_similarities, strict=False
            ):
                result = valid_results[original_idx]
                target_major = result.get("major", "")

                adjusted_similarity = adjust_similarity_score(
                    background_major=bg_major_clean,
                    target_major=target_major,
                    similarity=similarity,
                )
                result["similarity"] = adjusted_similarity
        except Exception:
            pass

    for i, result in enumerate(valid_results):
        if "similarity" not in result:
            result["similarity"] = 0.0

    return valid_results


def process_prediction_results(
    results: list,
    background_major: str,
    bg_target_similarity_cache: dict,
    num_target_universities: int,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
):
    if not results:
        return [], [], []

    results_with_similarity = _calculate_and_attach_similarities(
        results, background_major, bg_target_similarity_cache
    )

    top_similarity_results = _get_similar_major_recommendations(
        results_with_similarity, num_target_universities
    )

    top_cross_major_results = _get_cross_major_recommendations(
        results_with_similarity, background_major, cases_df, user_specified_combinations
    )

    final_user_specified_results = []
    if user_specified_combinations and results_with_similarity:
        specified_set = set(user_specified_combinations)
        final_user_specified_results = [
            res
            for res in results_with_similarity
            if (res.get("university"), res.get("major")) in specified_set
        ]

    return top_similarity_results, top_cross_major_results, final_user_specified_results
