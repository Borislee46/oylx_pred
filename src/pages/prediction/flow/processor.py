import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import (
    denormalize_language_score,
    get_cached_major_similarities_batch,
    has_school_major_details,
)
from src.pages.prediction.result_modifier.config import (
    AGENT_MIN_BALANCE_DIFF_MIN,
    AGENT_MIN_BALANCE_DIFF_RATIO,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    UNIVERSITY_COUNT_THRESHOLD,
    USER_SPECIFIED_LARGE_RANGE_TOP_N,
    USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD,
)
from src.pages.prediction.result_modifier.filters import (
    get_cross_major_recommendations,
    get_similar_major_recommendations,
)
from src.pages.prediction.result_modifier.language_penalty import LanguageRequirementPenalty
from src.pages.prediction.result_modifier.ranker import adjust_similarity_results_with_agent
from src.pages.prediction.result_modifier.similarity_adjuster import (
    adjust_similarity_score,
)
from src.utils.logger import setup_logger

boundary_processor_logger = setup_logger("page3", "prediction")


@dataclass
class ProcessingContext:
    background_major: str
    bg_target_similarity_cache: dict
    num_target_universities: int
    cases_df: pd.DataFrame | None = None
    user_specified_combinations: list[tuple[str, str]] | None = None
    background_faculty: str | None = None
    background_major_original: str | None = None
    probability_adjuster: Any | None = None
    gpa: float | None = None
    language_score: float | None = None
    language_type: str | None = None
    background_university: str | None = None


@dataclass
class ProcessingResult:
    similarity_results: list[dict[str, Any]]
    cross_major_results: list[dict[str, Any]]
    user_specified_results: list[dict[str, Any]]


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
        valid_combinations = _generate_combinations_small_scale(
            universities_to_consider, majors_to_consider
        )
    else:
        valid_combinations = _generate_combinations_large_scale(
            universities_to_consider, majors_to_consider
        )

    return valid_combinations, {"combination_count": len(valid_combinations)}


def _generate_combinations_small_scale(
    universities: list[str], majors: list[str]
) -> list[tuple[str, str]]:
    return [
        (univ, major)
        for univ in universities
        for major in majors
        if has_school_major_details(univ, major)
    ]


def _generate_combinations_large_scale(
    universities: list[str], majors: list[str]
) -> list[tuple[str, str]]:
    from src.pages.prediction.core.utils import _data_manager

    valid_universities = _data_manager.valid_universities
    valid_majors = _data_manager.valid_majors
    valid_set = _data_manager.valid_combinations

    if not valid_set:
        return []

    universities_filtered = [u for u in universities if u in valid_universities]
    majors_filtered = [m for m in majors if m in valid_majors]

    if not universities_filtered or not majors_filtered:
        return []

    return [
        (univ, major)
        for univ in universities_filtered
        for major in majors_filtered
        if (univ, major) in valid_set
    ]


def _filter_part_time_majors(results: list) -> list:
    if not results:
        return results

    from src.pages.prediction.core.utils import _data_manager

    details_df = _data_manager.details_df
    study_mode_col = None
    if details_df is not None:
        study_mode_cols = [c for c in details_df.columns if "学习模式" in c or "ѧϰ" in c]
        if study_mode_cols:
            study_mode_col = study_mode_cols[0]

    filtered = []
    for res in results:
        major_name = str(res.get("major", "")).lower()
        if "part" in major_name and "time" in major_name:
            continue

        if study_mode_col:
            univ = str(res.get("university", ""))
            major = str(res.get("major", ""))
            row = _data_manager.get_row(univ, major)
            if row is not None:
                mode_val = str(row.get(study_mode_col, "")).lower()
                is_pt = any(kw in mode_val for kw in ["part-time", "兼读", "pt", "part time"])
                is_ft = any(kw in mode_val for kw in ["full-time", "全日", "ft", "full time"])
                if is_pt and not is_ft:
                    continue

        filtered.append(res)

    return filtered


def _attach_faculty_batch(results: list) -> list:
    if not results:
        return results

    from src.pages.prediction.core.utils import _data_manager

    for res in results:
        univ = str(res.get("university", ""))
        major = str(res.get("major", ""))
        row = _data_manager.get_row(univ, major)
        res["faculty"] = (
            str(row.get("专业大类", ""))
            if row is not None and pd.notna(row.get("专业大类"))
            else ""
        )

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
        if bg_target_similarity_cache is None:
            boundary_processor_logger.warning(
                "bg_target_similarity_cache 为空，将导致相似度计算失效"
            )

        batch_similarities = get_cached_major_similarities_batch(
            similarity_pairs, cache=bg_target_similarity_cache
        )

        for idx, similarity in zip(valid_indices, batch_similarities, strict=True):
            result = valid_results[idx]
            target_major = str(result.get("major", "")).strip()

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
    results: list,
    user_specified_combinations: list[tuple[str, str]] | None,
    allow_degraded: bool = False,
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

    specified_results.sort(key=lambda x: x.get("probability", 0), reverse=True)

    if combination_count <= USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD:
        return specified_results

    return specified_results[:USER_SPECIFIED_LARGE_RANGE_TOP_N]


def _apply_faculty_filter(results: list, background_faculty: str | None) -> list:
    if not background_faculty:
        return results

    from src.pages.prediction.result_modifier.faculty_filters import (
        filter_schools_by_faculty_rules,
    )

    return filter_schools_by_faculty_rules(results, background_faculty)


def _get_similarity_threshold(num_target_universities: int) -> float:
    if num_target_universities > 0 and num_target_universities <= UNIVERSITY_COUNT_THRESHOLD:
        return HIGHER_SIMILARITY_THRESHOLD
    return MIN_SIMILARITY_THRESHOLD


def _apply_agent_balance_adjustment(
    top_similarity_results: list,
    top_cross_major_results: list,
    results_with_similarity: list,
    ctx: ProcessingContext,
    progress_reporter: Any | None = None,
) -> tuple[list, list]:
    balance_diff = len(top_cross_major_results) - len(top_similarity_results)

    max_len = max(len(top_similarity_results), len(top_cross_major_results))
    balance_threshold = max(
        AGENT_MIN_BALANCE_DIFF_MIN, int(math.ceil(AGENT_MIN_BALANCE_DIFF_RATIO * max_len))
    )
    if abs(balance_diff) < balance_threshold:
        return top_similarity_results, top_cross_major_results

    if ctx.cases_df is None or not ctx.background_major:
        return top_similarity_results, top_cross_major_results

    from src.agent.boundary_case_agent import BoundaryCaseAgent

    agent = BoundaryCaseAgent(cases_df=ctx.cases_df)
    current_threshold = _get_similarity_threshold(ctx.num_target_universities)
    agent_background_major = ctx.background_major_original or ctx.background_major

    top_similarity_results = adjust_similarity_results_with_agent(
        top_similarity_results,
        results_with_similarity,
        balance_diff,
        agent_background_major,
        current_threshold,
        agent,
        ctx.background_faculty,
        progress_reporter=progress_reporter,
    )

    sim_set = {(r.get("university"), r.get("major")) for r in top_similarity_results}
    top_cross_major_results = [
        r for r in top_cross_major_results if (r.get("university"), r.get("major")) not in sim_set
    ]

    return top_similarity_results, top_cross_major_results


def _preprocess_results(results: list, ctx: ProcessingContext) -> list:
    filtered_results = _filter_part_time_majors(results)
    if not filtered_results:
        return []

    if ctx.language_score is not None:
        lang_type = ctx.language_type or "雅思"
        raw_lang_score = denormalize_language_score(ctx.language_score, lang_type)
        filtered_results = LanguageRequirementPenalty.apply_penalty_to_results(
            filtered_results, raw_lang_score, lang_type
        )

    results_with_similarity = _calculate_and_attach_similarities(
        filtered_results, ctx.background_major, ctx.bg_target_similarity_cache
    )

    return _attach_faculty_batch(results_with_similarity)


def _generate_recommendations(
    results_with_similarity: list, ctx: ProcessingContext
) -> tuple[list, list]:
    results_for_recommendations = _apply_faculty_filter(
        results_with_similarity, ctx.background_faculty
    )

    top_similarity_results = get_similar_major_recommendations(
        results_for_recommendations,
        ctx.num_target_universities,
        probability_adjuster=ctx.probability_adjuster,
        gpa=ctx.gpa,
        language_score=ctx.language_score,
        background_university=ctx.background_university,
    )

    top_cross_major_results = get_cross_major_recommendations(
        results_for_recommendations,
        ctx.background_major,
        ctx.cases_df,
        ctx.background_faculty,
    )

    return top_similarity_results, top_cross_major_results


def process_prediction_results(
    results: list,
    background_major: str,
    bg_target_similarity_cache: dict,
    num_target_universities: int,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
    background_faculty: str | None = None,
    background_major_original: str | None = None,
    allow_degraded_user_specified: bool = False,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    language_type: str | None = None,
    background_university: str | None = None,
    progress_reporter: Any | None = None,
) -> tuple[list, list, list]:
    if not results:
        return [], [], []

    ctx = ProcessingContext(
        background_major=background_major,
        bg_target_similarity_cache=bg_target_similarity_cache,
        num_target_universities=num_target_universities,
        cases_df=cases_df,
        user_specified_combinations=user_specified_combinations,
        background_faculty=background_faculty,
        background_major_original=background_major_original,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type,
        background_university=background_university,
    )

    results_with_similarity = _preprocess_results(results, ctx)
    if not results_with_similarity:
        return [], [], []

    final_user_specified_results = _get_user_specified_results(
        results_with_similarity, ctx.user_specified_combinations, allow_degraded_user_specified
    )

    top_similarity_results, top_cross_major_results = _generate_recommendations(
        results_with_similarity, ctx
    )

    top_similarity_results, top_cross_major_results = _apply_agent_balance_adjustment(
        top_similarity_results,
        top_cross_major_results,
        results_with_similarity,
        ctx,
        progress_reporter,
    )

    return top_similarity_results, top_cross_major_results, final_user_specified_results
