import hashlib
import json
from typing import Any

import pandas as pd

from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS, DEFAULT_UI_KEYS
from src.pages.prediction.result_display import ResultsDisplay
from src.pages.prediction.result_display.delta_calculator import DeltaCalculator
from src.utils import log_interaction_event
from src.utils.session_manager import SessionManager

PROBABILITY_PRECISION = 6


def _normalize_target_list(v) -> list[str]:
    """Normalize target_universities/target_majors to list[str].

    The stored input_data may hold a comma-separated string (from form logging)
    or a list (from multiselect).  Always return a list.
    """
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [x.strip() for x in v.replace("，", ",").split(",") if x.strip()]
    return []


def _tag_results(
    sim: list[dict] | None,
    cross: list[dict] | None,
    user: list[dict] | None,
) -> list[dict]:
    tagged: list[dict] = []
    for r in sim or []:
        tagged.append({**r, "_source": "相似专业"})
    for r in cross or []:
        tagged.append({**r, "_source": "潜力跨专业"})
    for r in user or []:
        tagged.append({**r, "_source": "指定专业"})
    return tagged


def _compute_results_hash(
    sim_results: list[dict[str, Any]] | None,
    cross_results: list[dict[str, Any]] | None,
    user_specified_results: list[dict[str, Any]] | None,
) -> str:
    def _extract(res):
        return [
            (
                str(r.get("university")),
                str(r.get("major")),
                round(float(r.get("probability", 0)), 4),
            )
            for r in (res or [])
            if isinstance(r, dict) and r.get("university")
        ]

    combined = {
        "s": _extract(sim_results),
        "c": _extract(cross_results),
        "u": _extract(user_specified_results),
    }
    return hashlib.md5(json.dumps(combined, sort_keys=True).encode()).hexdigest()


def display_results_section(
    input_data: dict[str, Any],
    sim_results: list[dict[str, Any]] | None,
    cross_results: list[dict[str, Any]] | None,
    user_specified_results: list[dict[str, Any]] | None,
    cases_df: pd.DataFrame,
    submitted: bool = True,
) -> None:
    if not any([sim_results, cross_results, user_specified_results]):
        return

    all_results = _tag_results(sim_results, cross_results, user_specified_results)

    session_manager = SessionManager()
    prev_model = session_manager.get(DEFAULT_UI_KEYS.previous_prediction_results, None)
    prev_results_list = prev_model.unified_results if prev_model is not None else None

    current_unis = _normalize_target_list(input_data.get("target_universities", []))
    current_majors = _normalize_target_list(input_data.get("target_majors", []))

    has_prev, prev_prob_map, has_overlap = DeltaCalculator.should_show_delta(
        current_unis,
        current_majors,
        input_data.get("background_university", ""),
        input_data.get("background_major", ""),
        prev_results_list,
        session_manager.get(DEFAULT_UI_KEYS.previous_input_data, None),
    )

    show_delta = has_prev and has_overlap

    ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
        prev_prob_map=prev_prob_map if show_delta else None,
        delta_calculator=DeltaCalculator() if show_delta else None,
    ).display()
    current_hash = _compute_results_hash(sim_results, cross_results, user_specified_results)

    if current_hash != session_manager.get(
        DEFAULT_UI_KEYS.last_saved_results_hash, ""
    ) and not session_manager.get(DEFAULT_SESSION_KEYS.form_data_changed, False):
        session_manager.set(last_saved_results_hash=current_hash)
        log_interaction_event(
            "prediction_results",
            {
                "results_hash": current_hash,
                "similarity_count": len(sim_results or []),
                "cross_count": len(cross_results or []),
                "user_specified_count": len(user_specified_results or []),
                "best_probability": _best_probability(all_results),
                "target_universities": len(input_data.get("target_universities", [])),
                "target_majors": len(input_data.get("target_majors", [])),
            },
        )


def _best_probability(results: list[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return round(max(float(r.get("probability", 0) or 0) for r in results), PROBABILITY_PRECISION)
