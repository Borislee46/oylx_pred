import math
import time
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.adjustment.config import (
    AGENT_MIN_BALANCE_DIFF_MIN,
    AGENT_MIN_BALANCE_DIFF_RATIO,
    AGENT_NO_CHANGE_THRESHOLD,
    COMBINATION_POOL_FUZZY_MIN,
    COMBINATION_POOL_SEMANTIC_MIN,
    ENABLE_AGENT_BALANCE,
    HIGHER_SIMILARITY_THRESHOLD,
    MIN_SIMILARITY_THRESHOLD,
    UNIVERSITY_COUNT_THRESHOLD,
    USER_SPECIFIED_LARGE_RANGE_TOP_N,
    USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD,
)
from src.adjustment.faculty_filters import (
    get_allowed_target_faculties,
)
from src.adjustment.filters import (
    get_cross_major_recommendations,
    get_similar_major_recommendations,
)
from src.adjustment.language_penalty import reset_penalty_tracker
from src.adjustment.ranker import (
    adjust_similarity_results_with_agent,
)
from src.agent.boundary_case_agent import BoundaryCaseAgent
from src.pages.prediction.core.data_manager import _data_manager
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import denormalize_language_score
from src.pages.prediction.flow.result_processor import SingleResultProcessor
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce

processor_logger = setup_logger("page3", "prediction")


def _is_major_match(
    m_lower: str,
    bg_mapped: str,
    bg_orig: str,
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
) -> bool:
    if bg_mapped == m_lower:
        return True
    sim = float((bg_target_similarity_cache or {}).get((bg_mapped, m_lower), 0.0))
    if sim >= COMBINATION_POOL_SEMANTIC_MIN:
        return True
    return fuzz.token_sort_ratio(bg_orig, m_lower) > COMBINATION_POOL_FUZZY_MIN


def _resolve_prediction_target_lists(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[str], list[str]]:
    target_unis = list(input_data.get("target_universities") or all_universities_target)
    user_specified_majors = input_data.get("target_majors")

    if not user_specified_majors:
        bg_mapped = str(input_data.get("background_major", "")).strip().lower()
        target_majors: list[str] = []
        bg_orig = str(background_major_original or "").strip().lower()

        for m in all_majors_target:
            m_str = str(m).strip()
            if not m_str:
                continue
            if _is_major_match(m_str.lower(), bg_mapped, bg_orig, bg_target_similarity_cache):
                target_majors.append(m)

        if not target_majors:
            target_majors = list(all_majors_target)
    else:
        target_majors = list(user_specified_majors)

    return target_unis, target_majors


def _enumerate_valid_combinations(
    target_unis: list[str],
    target_majors: list[str],
) -> list[tuple[str, str]]:
    valid_unis = _data_manager.valid_universities
    valid_majors = _data_manager.valid_majors
    valid_set = _data_manager.valid_combinations
    return [
        (u, m)
        for u in target_unis
        if u in valid_unis
        for m in target_majors
        if m in valid_majors and (u, m) in valid_set
    ]


def _ensure_per_university_coverage(
    combos: list[tuple[str, str]],
    target_unis: list[str],
    target_majors: list[str],
) -> list[tuple[str, str]]:
    covered = {u for u, _ in combos}
    missing = [u for u in target_unis if u not in covered and u in _data_manager.valid_universities]
    if not missing:
        return combos

    valid_set = _data_manager.valid_combinations
    result = list(combos)
    for u in missing:
        candidates = [
            m for m in target_majors if m in _data_manager.valid_majors and (u, m) in valid_set
        ]
        if not candidates:
            candidates = [m for m in _data_manager.valid_majors if (u, m) in valid_set]
        if candidates:
            result.append((u, candidates[0]))
    return result


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    t0 = time.monotonic()
    target_unis, target_majors = _resolve_prediction_target_lists(
        input_data,
        all_universities_target,
        all_majors_target,
        bg_target_similarity_cache,
        background_major_original,
    )
    n_before_filter = len(target_unis) * len(target_majors)
    res = _enumerate_valid_combinations(target_unis, target_majors)
    n_after_filter = len(res)
    res = _ensure_per_university_coverage(res, target_unis, target_majors)
    n_final = len(res)

    elapsed = time.monotonic() - t0
    processor_logger.info(
        "Candidate combinations generated | unis=%d majors=%d cartesian=%d valid=%d final=%d elapsed=%.3fs",
        len(target_unis),
        len(target_majors),
        n_before_filter,
        n_after_filter,
        n_final,
        elapsed,
    )

    return res, {
        "combination_count": n_final,
        "progress_hints": {
            "target_unis": [str(u).strip() for u in target_unis if str(u).strip()],
            "target_majors": [str(m).strip() for m in target_majors if str(m).strip()],
            "user_locked_majors": bool(input_data.get("target_majors")),
        },
    }


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

    specified_results.sort(
        key=lambda x: clip_probability_coerce(x.get("probability")), reverse=True
    )

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
    enable_language_requirement_penalty: bool = True,
) -> tuple[list, list, list]:
    if not results:
        return [], [], []

    t0 = time.monotonic()

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
        enable_language_requirement_penalty=enable_language_requirement_penalty,
    )

    reset_penalty_tracker()
    processed_results = [pr for r in results if (pr := result_processor.process(r))]
    n_dropped = len(results) - len(processed_results)

    if not processed_results:
        processor_logger.warning(
            "No valid results after processing | raw=%d dropped=%d", len(results), n_dropped
        )
        return [], [], []

    user_results = _get_user_specified_results(
        processed_results, user_specified_combinations, allow_degraded=allow_degraded_user_specified
    )
    res_for_rec = [r for r in processed_results if r.get("_is_in_faculty_scope", True)]
    n_out_of_scope = len(processed_results) - len(res_for_rec)

    bg_major_param = background_major_original or background_major

    sim_rec = get_similar_major_recommendations(
        res_for_rec,
        num_target_universities,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
        background_major=bg_major_param,
        cases_df=cases_df,
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

    sim_before, cross_before = len(sim_rec), len(cross_rec)
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

    elapsed = time.monotonic() - t0
    processor_logger.info(
        "Results processed | raw=%d dropped=%d out_of_scope=%d "
        "sim=%d→%d cross=%d→%d user=%d elapsed=%.3fs",
        len(results),
        n_dropped,
        n_out_of_scope,
        sim_before,
        len(sim_rec),
        cross_before,
        len(cross_rec),
        len(user_results),
        elapsed,
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
    if not ENABLE_AGENT_BALANCE:
        return sim_rec, cross_rec

    diff = len(cross_rec) - len(sim_rec)
    max_len = max(len(sim_rec), len(cross_rec))
    value = AGENT_MIN_BALANCE_DIFF_RATIO * max_len
    threshold = max(AGENT_MIN_BALANCE_DIFF_MIN, math.ceil(value))

    if abs(diff) < threshold or cases_df is None or not background_major:
        if abs(diff) >= threshold:
            processor_logger.debug(
                "Agent balance skipped | diff=%d threshold=%d (cases_df=%s bg_major=%s)",
                diff,
                threshold,
                cases_df is not None,
                bool(background_major),
            )
        return sim_rec, cross_rec

    if diff < 0 and len(cross_rec) < AGENT_NO_CHANGE_THRESHOLD:
        processor_logger.debug(
            "Agent balance skipped | cross count insufficient: cross=%d < threshold=%d",
            len(cross_rec),
            AGENT_NO_CHANGE_THRESHOLD,
        )
        return sim_rec, cross_rec

    processor_logger.info(
        "Agent balance triggered | sim=%d cross=%d diff=%d threshold=%d",
        len(sim_rec),
        len(cross_rec),
        diff,
        threshold,
    )

    if agent is None:
        agent = BoundaryCaseAgent(cases_df=cases_df)

    limit = (
        HIGHER_SIMILARITY_THRESHOLD
        if 0 < num_unis <= UNIVERSITY_COUNT_THRESHOLD
        else MIN_SIMILARITY_THRESHOLD
    )

    t0 = time.monotonic()
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
    processor_logger.info(
        "Agent balance completed | elapsed=%.3fs sim_after=%d",
        time.monotonic() - t0,
        len(sim_rec),
    )

    sim_keys = {(r.get("university"), r.get("major")) for r in sim_rec}
    n_before_dedup = len(cross_rec)
    cross_rec = [r for r in cross_rec if (r.get("university"), r.get("major")) not in sim_keys]
    if n_before_dedup != len(cross_rec):
        processor_logger.info(
            "Agent balance deduplication | cross %d→%d", n_before_dedup, len(cross_rec)
        )

    return sim_rec, cross_rec
