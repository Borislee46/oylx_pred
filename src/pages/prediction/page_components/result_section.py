import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.result_display import ResultsDisplay
from src.utils import log_interaction_event
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


def _get_all_results(
    sim_results: list[dict] | None,
    cross_results: list[dict] | None,
    user_specified_results: list[dict] | None,
) -> list[dict]:
    all_results = (sim_results or []) + (cross_results or []) + (user_specified_results or [])
    return [r for r in all_results if isinstance(r, dict)]


def _render_stats_bar(results: list[dict]) -> None:
    if not results:
        return
    total = len(results)
    probs = [r.get("probability", 0) or 0 for r in results]
    reach = sum(1 for p in probs if p < 0.3)
    match = sum(1 for p in probs if 0.3 <= p < 0.6)
    safety = sum(1 for p in probs if p >= 0.6)

    html = (
        '<div class="hk-stat-row">'
        f'<div class="hk-stat-card"><div class="hk-stat-value">{total}</div>'
        f'<div class="hk-stat-label">推荐项目</div></div>'
        f'<div class="hk-stat-card"><div class="hk-stat-value" style="color:var(--hk-cyan)">{safety}</div>'
        f'<div class="hk-stat-label">录取概率较高</div></div>'
        f'<div class="hk-stat-card"><div class="hk-stat-value" style="color:#d97706">{match}</div>'
        f'<div class="hk-stat-label">录取概率中等</div></div>'
        f'<div class="hk-stat-card"><div class="hk-stat-value" style="color:#dc2626">{reach}</div>'
        f'<div class="hk-stat-label">需冲刺</div></div>'
        "</div>"
    )
    st.html(html)


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

    all_results = _get_all_results(sim_results, cross_results, user_specified_results)
    st.html('<div class="hk-section-label">预测结果</div>')
    _render_stats_bar(all_results)

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
