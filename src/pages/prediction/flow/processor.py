from typing import Any

import pandas as pd

from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import (
    denormalize_language_score,
    get_cached_major_similarities_batch,
)
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


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    from src.pages.prediction.core.utils import _data_manager

    target_unis = input_data.get("target_universities") or all_universities_target
    target_majors = input_data.get("target_majors") or all_majors_target

    valid_unis = _data_manager.valid_universities
    valid_majors = _data_manager.valid_majors
    valid_set = _data_manager.valid_combinations

    filtered_unis = [u for u in target_unis if u in valid_unis]
    filtered_majors = [m for m in target_majors if m in valid_majors]

    if not (filtered_unis and filtered_majors and valid_set):
        return [], {"combination_count": 0}

    res = [(u, m) for u in filtered_unis for m in filtered_majors if (u, m) in valid_set]
    return res, {"combination_count": len(res)}


def _filter_results(results: list) -> list:
    if not results:
        return []

    from src.pages.prediction.core.utils import _data_manager

    df = _data_manager.details_df
    if df is None:
        return results

    mode_col = next((c for c in df.columns if "学习模式" in c or "ѧϰ" in c), None)

    res = []
    for r in results:
        major = str(r.get("major", "")).lower()
        if "part" in major and "time" in major:
            continue

        if mode_col:
            row = _data_manager.get_row(r.get("university"), r.get("major"))
            if row is not None:
                m = str(row.get(mode_col, "")).lower()
                if "part" in m and "time" in m:
                    continue
        res.append(r)
    return res


def _attach_metadata(
    results: list,
    background_major: str,
    similarity_cache: dict,
    background_major_original: str | None = None,
) -> list:
    if not results:
        return []

    from src.pages.prediction.core.utils import _data_manager

    bg_major = str(background_major or "").strip()
    bg_major_orig = str(background_major_original or "").strip()

    pairs = [(str(r.get("major", "")).strip(), bg_major) for r in results]
    sims_mapped = (
        get_cached_major_similarities_batch(pairs, cache=similarity_cache)
        if bg_major
        else [0.0] * len(results)
    )
    sims_orig = None
    if bg_major_orig and bg_major_orig != bg_major:
        pairs_orig = [(str(r.get("major", "")).strip(), bg_major_orig) for r in results]
        sims_orig = get_cached_major_similarities_batch(pairs_orig, cache=similarity_cache)

    for i, r in enumerate(results):
        u, m = r.get("university"), r.get("major")
        row = _data_manager.get_row(u, m)
        r["faculty"] = (
            str(row.get("专业大类", ""))
            if row is not None and pd.notna(row.get("专业大类"))
            else ""
        )
        raw_sim = sims_mapped[i]
        if sims_orig is not None and i < len(sims_orig):
            raw_sim = max(raw_sim, sims_orig[i])
        r["similarity"] = (
            adjust_similarity_score(
                background_major=bg_major_orig or bg_major, target_major=str(m), similarity=raw_sim
            )
            if (bg_major_orig or bg_major)
            else 0.0
        )

    return results


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

    results = _filter_results(results)
    if not results:
        return [], [], []

    if language_score is not None:
        lang_type = language_type or "雅思"
        raw_lang = denormalize_language_score(language_score, lang_type)
        results = LanguageRequirementPenalty.apply_penalty_to_results(results, raw_lang, lang_type)

    results = _attach_metadata(
        results, background_major, bg_target_similarity_cache, background_major_original
    )

    user_results = _get_user_specified_results(
        results, user_specified_combinations, allow_degraded_user_specified
    )

    res_for_rec = results
    if background_faculty:
        from src.pages.prediction.result_modifier.faculty_filters import (
            filter_schools_by_faculty_rules,
        )

        res_for_rec = filter_schools_by_faculty_rules(results, background_faculty)

    sim_rec = get_similar_major_recommendations(
        res_for_rec,
        num_target_universities,
        probability_adjuster=probability_adjuster,
        gpa=gpa,
        language_score=language_score,
        background_university=background_university,
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
    )

    sim_rec, cross_rec = _apply_agent_balance_adjustment_flat(
        sim_rec,
        cross_rec,
        results,
        background_major,
        background_major_original,
        background_faculty,
        num_target_universities,
        cases_df,
        progress_reporter,
    )

    return sim_rec, cross_rec, user_results


def _apply_agent_balance_adjustment_flat(
    sim_rec: list,
    cross_rec: list,
    all_results: list,
    background_major: str,
    background_major_original: str | None,
    background_faculty: str | None,
    num_unis: int,
    cases_df: pd.DataFrame | None,
    reporter: Any | None,
) -> tuple[list, list]:
    diff = len(cross_rec) - len(sim_rec)
    max_len = max(len(sim_rec), len(cross_rec))
    value = AGENT_MIN_BALANCE_DIFF_RATIO * max_len
    threshold = max(
        AGENT_MIN_BALANCE_DIFF_MIN,
        int(-(-value // 1)) if value >= 0 else int(value),
    )

    if abs(diff) < threshold or cases_df is None or not background_major:
        return sim_rec, cross_rec

    if diff < 0 and len(cross_rec) < AGENT_NO_CHANGE_THRESHOLD:
        return sim_rec, cross_rec

    from src.agent.boundary_case_agent import BoundaryCaseAgent

    agent = BoundaryCaseAgent(cases_df=cases_df)
    limit = (
        HIGHER_SIMILARITY_THRESHOLD
        if 0 < num_unis <= UNIVERSITY_COUNT_THRESHOLD
        else MIN_SIMILARITY_THRESHOLD
    )
    bg_major = background_major_original or background_major

    sim_rec = adjust_similarity_results_with_agent(
        sim_rec,
        all_results,
        diff,
        bg_major,
        limit,
        agent,
        background_faculty,
        progress_reporter=reporter,
    )

    sim_keys = {(r.get("university"), r.get("major")) for r in sim_rec}
    cross_rec = [r for r in cross_rec if (r.get("university"), r.get("major")) not in sim_keys]

    return sim_rec, cross_rec
