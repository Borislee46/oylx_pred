import hashlib
import json
from typing import Any

import pandas as pd

from src.pages.prediction.result_display import ResultsDisplay
from src.utils.session_manager import SessionManager

PROBABILITY_PRECISION = 6


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

    ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
    ).display()

    session_manager = SessionManager()
    current_hash = _compute_results_hash(sim_results, cross_results, user_specified_results)

    if current_hash != session_manager.get(
        "last_saved_results_hash", ""
    ) and not session_manager.get("form_data_changed", False):
        session_manager.set(last_saved_results_hash=current_hash)
