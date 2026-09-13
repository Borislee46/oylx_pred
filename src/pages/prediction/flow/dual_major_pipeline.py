from __future__ import annotations

from typing import Any

from src.adjustment.admission_cache import get_admitted_combinations_from_dataframe
from src.pages.prediction.core.utils import get_background_faculty
from src.pages.prediction.flow.pipeline import run_prediction_pipeline_with_progress
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce
from src.utils.session_manager import PredictionResultModel

dual_major_logger = setup_logger("page3", "prediction")


def _compute_weights(alpha: float, is_dual_degree: bool) -> tuple[float, float]:
    if is_dual_degree:
        return 0.5, 0.5
    w2 = alpha / (1.0 + alpha)
    return 1.0 - w2, w2


def _dedup_result_pool(items: list[dict] | None) -> list[dict]:
    if not items:
        return []
    deduped: dict[tuple[str, str], dict] = {}
    for r in items:
        key = (str(r.get("university", "")), str(r.get("major", "")))
        existing = deduped.get(key)
        if existing is None or clip_probability_coerce(
            r.get("probability")
        ) > clip_probability_coerce(existing.get("probability")):
            deduped[key] = r
    return list(deduped.values())


def merge_dual_major_results(
    results_m1: PredictionResultModel,
    results_m2: PredictionResultModel,
    alpha: float,
    is_dual_degree: bool,
) -> PredictionResultModel:
    m1_unified = results_m1.unified_results or []
    m2_unified = results_m2.unified_results or []

    if not m1_unified:
        return results_m2
    if not m2_unified:
        return results_m1

    w1, w2 = _compute_weights(alpha, is_dual_degree)

    m1_map: dict[tuple, dict] = {
        (str(r.get("university", "")), str(r.get("major", ""))): r for r in m1_unified
    }
    m2_map: dict[tuple, dict] = {
        (str(r.get("university", "")), str(r.get("major", ""))): r for r in m2_unified
    }

    all_keys = set(m1_map.keys()) | set(m2_map.keys())

    merged: list[dict] = []
    for key in all_keys:
        r1 = m1_map.get(key)
        r2 = m2_map.get(key)

        if r1 is not None and r2 is not None:
            prob1 = clip_probability_coerce(r1.get("probability"))
            prob2 = clip_probability_coerce(r2.get("probability"))
            sim1 = float(r1.get("similarity", 0) or 0)
            sim2 = float(r2.get("similarity", 0) or 0)

            merged_prob = w1 * prob1 + w2 * prob2
            effective_sim = max(sim1, alpha * sim2) if not is_dual_degree else max(sim1, sim2)

            merged_r = dict(r1)
            merged_r["probability"] = round(merged_prob, 6)
            merged_r["similarity"] = effective_sim
            merged_r["_dual_major"] = {
                "m1_prob": prob1,
                "m2_prob": prob2,
                "m1_sim": sim1,
                "m2_sim": sim2,
                "effective_sim": effective_sim,
                "w1": w1,
                "w2": w2,
                "selected_major": "M1"
                if (sim1 >= sim2 if is_dual_degree else sim1 >= alpha * sim2)
                else "M2",
            }
        elif r1 is not None:
            merged_r = dict(r1)
            merged_r["_dual_major_missing"] = "M2"
        else:
            merged_r = dict(r2)
            merged_r["_dual_major_missing"] = "M1"

        merged.append(merged_r)

    merged.sort(key=lambda r: clip_probability_coerce(r.get("probability")), reverse=True)

    sim_results = _dedup_result_pool(
        (results_m1.similarity_results or []) + (results_m2.similarity_results or [])
    )
    cross_results = _dedup_result_pool(
        (results_m1.cross_major_results or []) + (results_m2.cross_major_results or [])
    )
    user_results = _dedup_result_pool(
        (results_m1.user_specified_results or []) + (results_m2.user_specified_results or [])
    )

    dual_major_logger.info(
        "Dual-major merge: M1=%d M2=%d merged=%d w1=%.2f w2=%.2f",
        len(m1_unified),
        len(m2_unified),
        len(merged),
        w1,
        w2,
    )

    return PredictionResultModel(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_results or None,
        unified_results=merged,
        meta={
            "dual_major": True,
            "alpha": alpha,
            "is_dual_degree": is_dual_degree,
            "w1": w1,
            "w2": w2,
            "m1_count": len(m1_unified),
            "m2_count": len(m2_unified),
            "merged_count": len(merged),
        },
    )


def run_dual_major_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    *,
    progress_cb=None,
    background_faculty: str | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
    page_state=None,
) -> PredictionResultModel:
    alpha = float(input_data.get("dual_alpha", 0.85))
    is_dual_degree = bool(input_data.get("is_dual_degree", False))
    m1 = input_data.get("background_major")
    m2 = input_data.get("background_major_2")

    if not m2:
        dual_major_logger.warning(
            "Dual-major pipeline called without background_major_2, falling back to single"
        )
        return run_prediction_pipeline_with_progress(
            input_data,
            model_name,
            cases_df_fingerprint,
            loaded_feature_names,
            progress_cb=progress_cb,
            background_faculty=background_faculty,
            admitted_combinations=admitted_combinations,
            page_state=page_state,
        )

    dual_major_logger.info(
        "Dual-major pipeline: M1=%s M2=%s alpha=%.2f is_dual=%s",
        m1,
        m2,
        alpha,
        is_dual_degree,
    )

    input_m1 = dict(input_data)
    input_m2 = dict(input_data)
    input_m2["background_major"] = m2
    input_m2["background_major_original"] = input_data.get("background_major_2_original") or m2
    input_m2.pop("background_major_2", None)
    input_m2.pop("background_major_2_original", None)
    input_m2.pop("is_dual_degree", None)
    input_m2.pop("dual_alpha", None)

    cases_df = page_state.cases_df if page_state else None
    m2_faculty = get_background_faculty(m2, cases_df) if cases_df is not None else None
    m2_admitted = (
        get_admitted_combinations_from_dataframe(cases_df, m2)
        if cases_df is not None and m2
        else None
    )

    result_m1 = run_prediction_pipeline_with_progress(
        input_m1,
        model_name,
        cases_df_fingerprint,
        loaded_feature_names,
        progress_cb=progress_cb,
        background_faculty=background_faculty,
        admitted_combinations=admitted_combinations,
        page_state=page_state,
    )

    result_m2 = run_prediction_pipeline_with_progress(
        input_m2,
        model_name,
        cases_df_fingerprint,
        loaded_feature_names,
        progress_cb=progress_cb,
        background_faculty=m2_faculty,
        admitted_combinations=m2_admitted,
        page_state=page_state,
    )

    return merge_dual_major_results(result_m1, result_m2, alpha, is_dual_degree)
