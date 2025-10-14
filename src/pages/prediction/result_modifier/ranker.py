import pandas as pd

from src.pages.prediction.result_modifier.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.pages.prediction.result_modifier.config import (
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    TOP_N_RECOMMENDATIONS,
    UNIVERSITY_COUNT_THRESHOLD,
)
from src.pages.prediction.result_modifier.utils import clip_probability


def get_similar_major_recommendations(
    results_with_similarity: list, num_target_universities: int
) -> list:
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
            TOP_N_RECOMMENDATIONS, filtered_by_similarity, key=lambda x: x.get("similarity", 0.0)
        )

    for c in top_candidates:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    top_candidates.sort(key=lambda x: x.get("probability", 0.0), reverse=True)

    return top_candidates


def get_cross_major_recommendations(
    results_with_similarity: list,
    background_major: str,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
) -> list:
    if not background_major or not results_with_similarity:
        return []

    try:
        bg_major_clean = str(background_major).strip()
        admitted_combinations = get_admitted_combinations_from_dataframe(cases_df, bg_major_clean)

        if admitted_combinations:
            admitted_results = [
                res
                for res in results_with_similarity
                if (res.get("university"), res.get("major")) in admitted_combinations
                and res.get("similarity", 1.0) < MIN_SIMILARITY_THRESHOLD
                and res.get("similarity", 0.0) > 0.8
            ]

            if admitted_results:
                admitted_results.sort(key=lambda x: x.get("similarity", 1.0))
                top_least_similar = admitted_results[:TOP_N_RECOMMENDATIONS]
                top_least_similar.sort(key=lambda x: x.get("probability", 0), reverse=True)
                return top_least_similar
        return []

    except Exception:
        return []
