from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.agent.boundary_case_agent import BoundaryCaseAgent
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import (
    _data_manager,
    denormalize_language_score,
)
from src.pages.prediction.flow.result_processor import SingleResultProcessor
from src.pages.prediction.result_modifier.config import (
    AGENT_MIN_BALANCE_DIFF_MIN,
    AGENT_MIN_BALANCE_DIFF_RATIO,
    AGENT_NO_CHANGE_THRESHOLD,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    UNIVERSITY_COUNT_THRESHOLD,
    USER_SPECIFIED_LARGE_RANGE_TOP_N,
    USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD,
)
from src.pages.prediction.result_modifier.faculty_filters import (
    get_allowed_target_faculties,
)
from src.pages.prediction.result_modifier.filters import (
    get_cross_major_recommendations,
    get_similar_major_recommendations,
)
from src.pages.prediction.result_modifier.ranker import adjust_similarity_results_with_agent
from src.utils.logger import setup_logger

boundary_processor_logger = setup_logger("page3", "prediction")


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[str, float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    target_unis = input_data.get("target_universities") or all_universities_target
    user_specified_majors = input_data.get("target_majors")

    if not user_specified_majors:
        bg_mapped = str(input_data.get("background_major", "")).strip().lower()

        SEMANTIC_THRESHOLD = 0.6
        target_majors = []
        bg_orig = str(background_major_original or "").strip().lower()

        for m in all_majors_target:
            m_str = str(m).strip()
            if not m_str:
                continue
            m_lower = m_str.lower()

            sim_score = 0.0
            if bg_target_similarity_cache is not None and bg_mapped:
                if isinstance(bg_target_similarity_cache, dict):
                    sim_score = bg_target_similarity_cache.get((bg_mapped, m_lower), 0.0)
                elif isinstance(bg_target_similarity_cache, pd.Series):
                    try:
                        sim_score = bg_target_similarity_cache.get((bg_mapped, m_lower), 0.0)
                    except (KeyError, TypeError):
                        pass

            if sim_score >= SEMANTIC_THRESHOLD:
                target_majors.append(m)
                continue

            if fuzz.token_sort_ratio(bg_orig, m_lower) > 90:
                target_majors.append(m)

        if not target_majors:
            target_majors = all_majors_target
    else:
        target_majors = user_specified_majors

    valid_unis = _data_manager.valid_universities
    valid_majors = _data_manager.valid_majors
    valid_set = _data_manager.valid_combinations

    res = [
        (u, m)
        for u in target_unis
        if u in valid_unis
        for m in target_majors
        if m in valid_majors and (u, m) in valid_set
    ]
    return res, {"combination_count": len(res)}


def _get_user_specified_results(
    results: list,
    user_specified_combinations: list[tuple[str, str]] | None,
    allow_degraded: bool = True,
) -> list:
    if not user_specified_combinations or not results:
        return []

    specified_set = set(user_specified_combinations)
    specified_results = [
        res
        for res in results
        if (res.get("university"), res.get("major")) in specified_set
        and (allow_degraded or res.get("_is_in_faculty_scope", True))
    ]

    if not specified_results:
        return []

    specified_results.sort(key=lambda x: x.get("probability", 0), reverse=True)

    if len(user_specified_combinations) <= USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD:
        return specified_results

    return specified_results[:USER_SPECIFIED_LARGE_RANGE_TOP_N]


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
    agent: Any | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
) -> tuple[list, list, list]:
    if not results:
        return [], [], []

    lang_type = language_type or "雅思"
    raw_lang = (
        denormalize_language_score(language_score, lang_type)
        if language_score is not None
        else None
    )

    allowed_faculties = (
        get_allowed_target_faculties(background_faculty)
        if background_faculty and not allow_degraded_user_specified
        else set()
    )

    result_processor = SingleResultProcessor(
        data_manager=_data_manager,
        bg_major=str(background_major or "").strip(),
        bg_major_orig=str(background_major_original or "").strip(),
        bg_orig_lower=str(background_major_original or background_major or "").strip().lower(),
        raw_lang=raw_lang,
        lang_type=lang_type,
        bg_target_similarity_cache=bg_target_similarity_cache,
        allowed_faculties=allowed_faculties,
        background_faculty=background_faculty,
    )

    processed_results = [pr for r in results if (pr := result_processor.process(r))]

    if not processed_results:
        return [], [], []

    user_results = _get_user_specified_results(
        processed_results, user_specified_combinations, allow_degraded=allow_degraded_user_specified
    )
    res_for_rec = [r for r in processed_results if r.get("_is_in_faculty_scope", True)]

    bg_major_param = background_major_original or background_major
    sim_rec = get_similar_major_recommendations(
        res_for_rec,
        num_target_universities,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=bg_major_param,
    )

    cross_rec = get_cross_major_recommendations(
        res_for_rec,
        background_major,
        cases_df,
        background_faculty,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        admitted_combinations=admitted_combinations,
    )

    sim_rec, cross_rec = _apply_agent_balance_adjustment_flat(
        sim_rec,
        cross_rec,
        processed_results,
        bg_major_param,
        background_faculty,
        num_target_universities,
        cases_df,
        progress_reporter,
        is_cross_faculty=allow_degraded_user_specified,
        agent=agent,
    )

    return sim_rec, cross_rec, user_results


def _apply_agent_balance_adjustment_flat(
    sim_rec: list,
    cross_rec: list,
    all_results: list,
    background_major: str,
    background_faculty: str | None,
    num_unis: int,
    cases_df: pd.DataFrame | None,
    reporter: Any | None,
    is_cross_faculty: bool = False,
    agent: Any | None = None,
) -> tuple[list, list]:
    diff = len(cross_rec) - len(sim_rec)
    max_len = max(len(sim_rec), len(cross_rec))
    value = AGENT_MIN_BALANCE_DIFF_RATIO * max_len
    threshold = max(AGENT_MIN_BALANCE_DIFF_MIN, int(-(-value // 1)) if value >= 0 else int(value))

    if abs(diff) < threshold or cases_df is None or not background_major:
        return sim_rec, cross_rec

    if diff < 0 and len(cross_rec) < AGENT_NO_CHANGE_THRESHOLD:
        return sim_rec, cross_rec

    if agent is None:
        agent = BoundaryCaseAgent(cases_df=cases_df)

    limit = (
        HIGHER_SIMILARITY_THRESHOLD
        if 0 < num_unis <= UNIVERSITY_COUNT_THRESHOLD
        else MIN_SIMILARITY_THRESHOLD
    )

    sim_rec = adjust_similarity_results_with_agent(
        sim_rec,
        all_results,
        diff,
        background_major,
        limit,
        agent,
        background_faculty,
        progress_reporter=reporter,
        is_cross_faculty=is_cross_faculty,
    )

    sim_keys = {(r.get("university"), r.get("major")) for r in sim_rec}
    cross_rec = [r for r in cross_rec if (r.get("university"), r.get("major")) not in sim_keys]

    return sim_rec, cross_rec
