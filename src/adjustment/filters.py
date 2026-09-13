import heapq
from typing import Any

import pandas as pd

from src.adjustment.admission_cache import (
    get_admitted_combinations_from_dataframe,
)
from src.adjustment.config import (
    AGENT_MIN_SAFE_RELAX_THRESHOLD,
    CROSS_MAJOR_SIMILARITY_MIN,
    FUZZY_BIAS_THRESHOLD_HIGH,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    TOP_N_RECOMMENDATIONS,
    UNIVERSITY_COUNT_THRESHOLD,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability

logger = setup_logger("page3", "prediction")


def _make_selection_sort_key(
    probability_adjuster: Any | None,
    gpa: float | None,
    language_score: float | None,
    background_university: str | None,
):
    comp_score = None
    if probability_adjuster is not None and gpa is not None and language_score is not None:
        comp_score = probability_adjuster.get_comprehensive_score(
            gpa, language_score, background_university
        )

    def get_sort_key(res: dict[str, Any]) -> float:
        similarity = float(res.get("similarity", 0.0) or 0.0)
        if comp_score is None:
            return similarity
        return probability_adjuster.calculate_selection_score(
            similarity,
            str(res.get("university", "")),
            comp_score=comp_score,
        )

    return get_sort_key


def get_similar_major_recommendations(
    results_with_similarity: list[dict[str, Any]],
    num_target_universities: int,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    background_university: str | None = None,
    background_major: str | None = None,
    cases_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if not results_with_similarity:
        return []

    use_higher_threshold = 0 < num_target_universities <= UNIVERSITY_COUNT_THRESHOLD
    current_threshold = (
        HIGHER_SIMILARITY_THRESHOLD if use_higher_threshold else MIN_SIMILARITY_THRESHOLD
    )

    filtered_by_similarity = [
        res for res in results_with_similarity if res.get("similarity", 0.0) >= current_threshold
    ]

    if not filtered_by_similarity and use_higher_threshold:
        filtered_by_similarity = [
            res
            for res in results_with_similarity
            if res.get("similarity", 0.0) >= MIN_SIMILARITY_THRESHOLD
        ]

    if not filtered_by_similarity:
        floor_threshold = max(AGENT_MIN_SAFE_RELAX_THRESHOLD, CROSS_MAJOR_SIMILARITY_MIN)
        filtered_by_similarity = [
            res for res in results_with_similarity if res.get("similarity", 0.0) >= floor_threshold
        ]
        if not filtered_by_similarity:
            logger.warning(
                "相似专业推荐: 无候选通过最低阈值 | threshold=%.2f n_total=%d",
                floor_threshold,
                len(results_with_similarity),
            )
            return []

    logger.debug(
        "相似专业推荐: 过滤后 | threshold=%.2f n_filtered=%d use_higher=%s",
        current_threshold,
        len(filtered_by_similarity),
        use_higher_threshold,
    )

    get_sort_key = _make_selection_sort_key(
        probability_adjuster, gpa, language_score, background_university
    )

    IDENTITY_MIN_SLOT_RATIO = 0.4
    target_count = TOP_N_RECOMMENDATIONS

    def is_strong_match(res: dict[str, Any]) -> bool:
        return res.get("_strong_match_score", 0) > FUZZY_BIAS_THRESHOLD_HIGH

    strong_matches = [r for r in filtered_by_similarity if is_strong_match(r)]
    others = [r for r in filtered_by_similarity if not is_strong_match(r)]

    strong_matches.sort(key=get_sort_key, reverse=True)
    others.sort(key=get_sort_key, reverse=True)

    min_identity_slots = int(target_count * IDENTITY_MIN_SLOT_RATIO)

    selected = []
    identity_to_take = min(len(strong_matches), min_identity_slots)
    selected.extend(strong_matches[:identity_to_take])

    remaining_count = target_count - len(selected)
    competing_pool = strong_matches[identity_to_take:] + others
    competing_pool.sort(key=get_sort_key, reverse=True)

    selected.extend(competing_pool[:remaining_count])

    top_candidates = selected

    if len(top_candidates) < TOP_N_RECOMMENDATIONS:
        floor_threshold = max(AGENT_MIN_SAFE_RELAX_THRESHOLD, CROSS_MAJOR_SIMILARITY_MIN)
        expanded = [
            res for res in results_with_similarity if res.get("similarity", 0.0) >= floor_threshold
        ]
        if expanded:
            top_candidates = heapq.nlargest(
                TOP_N_RECOMMENDATIONS,
                expanded,
                key=get_sort_key,
            )

    for c in top_candidates:
        if isinstance(c, dict) and "probability" in c:
            c["probability"] = clip_probability(c.get("probability", 0.0))

    top_candidates.sort(key=lambda x: x.get("probability", 0.0), reverse=True)
    logger.info(
        "相似专业推荐完成 | n_final=%d strong=%d others=%d",
        len(top_candidates),
        len(strong_matches),
        len(others),
    )
    return top_candidates


def get_cross_major_recommendations(
    results_with_similarity: list[dict[str, Any]],
    background_major: str,
    cases_df: pd.DataFrame | None = None,
    background_faculty: str | None = None,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    background_university: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if not background_major or not results_with_similarity:
        return []

    bg_major_clean = str(background_major).strip()
    admitted_combos = (
        admitted_combinations
        if admitted_combinations is not None
        else get_admitted_combinations_from_dataframe(cases_df, bg_major_clean)
    )

    if not admitted_combos:
        logger.debug(
            "跨专业推荐: 无历史录取组合 | bg_major=%s",
            bg_major_clean,
        )
        return []

    faculty_filter = _create_faculty_filter(background_faculty)

    admitted_results = [
        res
        for res in results_with_similarity
        if (res.get("university"), res.get("major")) in admitted_combos
        and CROSS_MAJOR_SIMILARITY_MIN <= res.get("similarity", 0.0) < MIN_SIMILARITY_THRESHOLD
        and faculty_filter(res)
    ]

    if admitted_results:
        get_sort_key = _make_selection_sort_key(
            probability_adjuster, gpa, language_score, background_university
        )

        top_candidates = heapq.nlargest(
            TOP_N_RECOMMENDATIONS,
            admitted_results,
            key=get_sort_key,
        )
        for res in top_candidates:
            res["admitted"] = 1
        top_candidates.sort(key=get_sort_key, reverse=True)
        logger.info(
            "跨专业推荐完成 | n_admitted=%d n_final=%d",
            len(admitted_results),
            len(top_candidates),
        )
        return top_candidates
    logger.debug("跨专业推荐: 无通过筛选的录取组合")
    return []


def _create_faculty_filter(background_faculty: str | None):
    if not background_faculty:
        return lambda res: True

    from src.adjustment.faculty_filters import get_allowed_target_faculties

    allowed_faculties = get_allowed_target_faculties(background_faculty)
    bg_faculty_clean = background_faculty.strip()

    if allowed_faculties:
        return lambda res: _check_faculty(res, allowed_faculties, bg_faculty_clean)
    else:
        return lambda res: _check_faculty_simple(res, bg_faculty_clean)


def _check_faculty(res, allowed_faculties, bg_faculty_clean):
    faculty = res.get("faculty", "")
    return not faculty or (faculty in allowed_faculties and faculty != bg_faculty_clean)


def _check_faculty_simple(res, bg_faculty_clean):
    faculty = res.get("faculty", "")
    return not faculty or faculty != bg_faculty_clean
