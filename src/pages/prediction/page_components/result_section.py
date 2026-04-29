import hashlib
import json
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.result_display import ResultsDisplay
from src.pages.prediction.result_modifier.config import TOP_N_RECOMMENDATIONS
from src.utils import log_interaction_event
from src.utils.session_manager import SessionManager

PROBABILITY_PRECISION = 6


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


def _percentile(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (p / 100) * (n - 1)
    f = int(k)
    c = k - f
    if f + 1 < n:
        return sorted_vals[f] + c * (sorted_vals[f + 1] - sorted_vals[f])
    return sorted_vals[min(f, n - 1)]


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


def _render_stats_bar(results: list[dict]) -> None:
    if not results:
        return

    probs = sorted(r.get("probability", 0) or 0 for r in results)
    low = _percentile(probs, 33)
    high = _percentile(probs, 66)

    total = len(results)
    reach = [r for r in results if (r.get("probability", 0) or 0) < low]
    match = [r for r in results if low <= (r.get("probability", 0) or 0) < high]
    safety = [r for r in results if (r.get("probability", 0) or 0) >= high]

    def _with_tier(r: dict) -> dict:
        p = r.get("probability", 0) or 0
        r["_tier"] = "较稳" if p >= high else ("适中" if p >= low else "冲刺")
        return r

    st.session_state["_stat_tiers"] = {
        "total": [_with_tier(r) for r in results],
        "high": [_with_tier(r) for r in safety],
        "mid": [_with_tier(r) for r in match],
        "low": [_with_tier(r) for r in reach],
    }

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, str(total), "推荐项目", "total"),
        (c2, str(len(safety)), "录取概率较高", "high"),
        (c3, str(len(match)), "录取概率中等", "mid"),
        (c4, str(len(reach)), "需冲刺", "low"),
    ]
    for col, val, label, key in cards:
        with col:
            if st.button(
                f"{val}  {label}",
                key=f"statbtn_{key}",
                width="stretch",
            ):
                st.session_state["_stat_dialog"] = key
                st.rerun()

    dialog = st.session_state.get("_stat_dialog")
    if dialog and dialog in st.session_state.get("_stat_tiers", {}):
        _tier_dialog(dialog)
        st.session_state["_stat_dialog"] = None


@st.dialog("预测结果明细", width="large")
def _tier_dialog(tier: str) -> None:
    tiers = st.session_state.get("_stat_tiers", {})
    items = tiers.get(tier, [])
    tier_label = {
        "total": "全部项目",
        "high": "录取概率较高",
        "mid": "录取概率中等",
        "low": "需冲刺",
    }.get(tier, tier)

    if not items:
        st.info(f"「{tier_label}」暂无项目")
    else:
        sim_items = [r for r in items if r.get("_source") == "相似专业"]
        cross_items = [r for r in items if r.get("_source") == "潜力跨专业"]
        user_items = [r for r in items if r.get("_source") == "指定专业"]

        rd = ResultsDisplay(
            top_similarity_results=sim_items or None,
            top_cross_major_results=cross_items or None,
            user_specified_results=user_items or None,
        )

        has_user = bool(user_items)
        has_sim = bool(sim_items)
        has_cross = bool(cross_items)

        if not (has_user or has_sim or has_cross):
            st.info("无推荐结果。")
            return

        st.caption(f"「{tier_label}」共 {len(items)} 项")

        if has_user:
            rd.display_dataframe(
                rd.get_result_dataframe("user_specified"), result_type="user_specified"
            )
        elif has_sim and has_cross:
            c1, c2 = st.columns(2)
            with c1:
                rd.display_dataframe(
                    rd.get_result_dataframe("similarity", max_items=TOP_N_RECOMMENDATIONS),
                    result_type="similarity",
                )
            with c2:
                rd.display_dataframe(
                    rd.get_result_dataframe("cross_major", max_items=TOP_N_RECOMMENDATIONS),
                    result_type="cross_major",
                )
        elif has_sim:
            rd.display_dataframe(
                rd.get_result_dataframe("similarity", max_items=TOP_N_RECOMMENDATIONS),
                result_type="similarity",
            )
        elif has_cross:
            rd.display_dataframe(
                rd.get_result_dataframe("cross_major", max_items=TOP_N_RECOMMENDATIONS),
                result_type="cross_major",
            )

    if st.button("关闭", key=f"stat_dialog_close_{tier}"):
        st.session_state["_stat_dialog"] = None
        st.rerun()


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
