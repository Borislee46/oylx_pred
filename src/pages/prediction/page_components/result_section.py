import hashlib
import json
from typing import Any, Optional

import pandas as pd

from src.pages.prediction.result_display import ResultsDisplay
from src.utils.session_manager import SessionManager

PROBABILITY_PRECISION = 6


def _extract_key_fields(results: Optional[list[dict[str, Any]]]) -> list[tuple[str, str, float]]:
    if not results:
        return []
    return [
        (
            str(r.get("university") or ""),
            str(r.get("major") or ""),
            round(float(r.get("probability", 0.0) or 0.0), PROBABILITY_PRECISION),
        )
        for r in results
        if isinstance(r, dict) and r.get("university") and r.get("major")
    ]


def _compute_results_hash(
    sim_results: Optional[list[dict[str, Any]]],
    cross_results: Optional[list[dict[str, Any]]],
    user_specified_results: Optional[list[dict[str, Any]]],
) -> str:
    combined = {
        "sim": _extract_key_fields(sim_results),
        "cross": _extract_key_fields(cross_results),
        "user": _extract_key_fields(user_specified_results),
    }
    content = json.dumps(combined, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(content.encode()).hexdigest()


def display_results_section(
    input_data: dict[str, Any],
    sim_results: Optional[list[dict[str, Any]]],
    cross_results: Optional[list[dict[str, Any]]],
    user_specified_results: Optional[list[dict[str, Any]]],
    cases_df: pd.DataFrame,
    submitted: bool = True,
) -> None:
    if all(x is None for x in [sim_results, cross_results, user_specified_results]):
        return

    session_manager = SessionManager()
    background_university = input_data.get("background_university")
    background_major = input_data.get("background_major")

    results_display = ResultsDisplay(
        top_similarity_results=sim_results,
        top_cross_major_results=cross_results,
        user_specified_results=user_specified_results,
    )

    results_display.display()

    current_results_hash = _compute_results_hash(sim_results, cross_results, user_specified_results)
    last_saved_hash = session_manager.get("last_saved_results_hash", "")
    form_data_changed = bool(session_manager.get("form_data_changed", False))

    if current_results_hash and current_results_hash != last_saved_hash and not form_data_changed:
        session_manager.set(
            last_saved_results_hash=current_results_hash,
        )
