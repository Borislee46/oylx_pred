import math
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
    COMBINATION_POOL_FUZZY_MIN,
    COMBINATION_POOL_SEMANTIC_MIN,
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

# ── Hot-path optimisations ──
# Load hot major substrings from config lazily; fallback to hard-coded defaults.
_HOT_MAJOR_SUBSTRINGS: tuple | None = None


def _load_hot_paths():
    global _HOT_MAJOR_SUBSTRINGS
    if _HOT_MAJOR_SUBSTRINGS is not None:
        return
    import json
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "hot_paths.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        _HOT_MAJOR_SUBSTRINGS = tuple(cfg.get("hot_major_substrings", []))
    except Exception:
        _HOT_MAJOR_SUBSTRINGS = (
            "smart manufacturing",
            "accounting and finance analytics",
            "information technology",
        )


def _is_major_match(
    m_lower: str,
    bg_mapped: str,
    bg_orig: str,
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
) -> bool:
    _load_hot_paths()
    if any(kw in m_lower for kw in _HOT_MAJOR_SUBSTRINGS):
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


def _count_fuzz_passing_majors(
    input_data: PredictionInput,
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
    background_major_original: str | None,
) -> int:
    bg_mapped = str(input_data.get("background_major", "")).strip().lower()
    bg_orig = (
        str(background_major_original or input_data.get("background_major") or "").strip().lower()
    )
    if not bg_orig:
        return 0
    n = 0
    seen: set[str] = set()
    for m in all_majors_target:
        m_str = str(m).strip()
        if not m_str:
            continue
        m_lower = m_str.lower()
        if m_lower in seen:
            continue
        if _is_major_match(m_lower, bg_mapped, bg_orig, bg_target_similarity_cache):
            seen.add(m_lower)
            n += 1
    return n


def prediction_progress_scope_meta(
    input_data: PredictionInput,
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
    background_major_original: str,
) -> tuple[int | None, int | None, bool]:
    tu = input_data.get("target_universities") or []
    utrim = [str(u).strip() for u in tu if str(u).strip()]
    n_uni = len(set(utrim)) if utrim else None

    tm = input_data.get("target_majors") or []
    mtrim = [str(m).strip() for m in tm if str(m).strip()]
    if mtrim:
        return n_uni, len(set(mtrim)), True

    fuzz_n = _count_fuzz_passing_majors(
        input_data,
        all_majors_target,
        bg_target_similarity_cache,
        background_major_original,
    )
    if fuzz_n > 0:
        return n_uni, fuzz_n, False
    return n_uni, None, False


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


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    target_unis, target_majors = _resolve_prediction_target_lists(
        input_data,
        all_universities_target,
        all_majors_target,
        bg_target_similarity_cache,
        background_major_original,
    )
    res = _enumerate_valid_combinations(target_unis, target_majors)
    return res, {
        "combination_count": len(res),
        "progress_hints": {
            "target_unis": [str(u).strip() for u in target_unis if str(u).strip()],
            "target_majors": [str(m).strip() for m in target_majors if str(m).strip()],
            "user_locked_majors": bool(input_data.get("target_majors")),
        },
    }


def count_cases_with_similar_background(
    cases_df: pd.DataFrame | None,
    background_major: str,
    background_major_original: str,
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
) -> int:
    if cases_df is None or cases_df.empty or "background_major" not in cases_df.columns:
        return 0

    bg_mapped = str(background_major or "").strip().lower()
    bg_orig = str(background_major_original or background_major or "").strip().lower()
    if not bg_orig:
        return 0

    vc = cases_df["background_major"].astype(str).str.strip().value_counts(dropna=False)
    total = 0
    for maj_val, cnt in vc.items():
        m_lower = str(maj_val).strip().lower()
        if not m_lower:
            continue
        sim_score = 0.0
        if bg_target_similarity_cache is not None and bg_mapped:
            sim_score = float(bg_target_similarity_cache.get((bg_mapped, m_lower), 0.0))
        if sim_score >= COMBINATION_POOL_SEMANTIC_MIN:
            total += int(cnt)
            continue
        if fuzz.token_sort_ratio(bg_orig, m_lower) > COMBINATION_POOL_FUZZY_MIN:
            total += int(cnt)

    if total == 0 and bg_mapped:
        mask = cases_df["background_major"].astype(str).str.strip().str.lower() == bg_mapped
        return int(mask.sum())
    return total


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
    threshold = max(AGENT_MIN_BALANCE_DIFF_MIN, math.ceil(value))

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
