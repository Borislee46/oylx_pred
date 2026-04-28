import hashlib
import json
import time
from typing import Any

import streamlit as st

from src.agent.context import StudentContext
from src.agent.registry import AgentRegistry
from src.pages.prediction.application_readiness import (
    build_readiness_profile,
    render_readiness_card,
)
from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS
from src.pages.prediction.page_components.display_helpers import show_explanation
from src.pages.prediction.page_components.result_section import display_results_section
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils import log_interaction_event
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

content_display_logger = setup_logger("page3", "prediction")


def _render_ai_explanation(
    prediction_results,
    background_university: str,
    background_major: str,
    gpa: float,
    language_score: float,
    language_type: str,
    experience_details: dict | None,
) -> None:
    sim = prediction_results.similarity_results or []
    cross = prediction_results.cross_major_results or []
    unified = prediction_results.unified_results or []
    if not any([sim, cross]):
        return

    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = {}

    cache_key = _build_explain_cache_key(
        sim=sim,
        cross=cross,
        unified=unified,
        background_major=background_major,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type,
        experience_details=experience_details,
    )

    cached = st.session_state["explain_cache"].get(cache_key)
    if cached:
        _show_explanation(cached)
        return

    # Manual trigger — don't auto-run
    if "explain_requested" not in st.session_state:
        st.session_state["explain_requested"] = False

    if not st.session_state["explain_requested"]:
        c1, _ = st.columns([2, 5])
        with c1:
            if st.button("AI 选校解读", key="explain_btn", use_container_width=True):
                st.session_state["explain_requested"] = True
                st.rerun()
        return

    ctx = StudentContext(
        stage="match",
        background_university=background_university,
        background_major=background_major,
        gpa=gpa,
        language_score=language_score,
        language_type=language_type or "",
        experience_details=experience_details or {},
        prediction_results={
            "similarity_results": sim,
            "cross_major_results": cross,
            "unified_results": unified,
        },
    )

    agent = AgentRegistry.get("explain")

    placeholder = st.empty()
    _render_analysis_progress(placeholder, "正在分析背景信息")
    time.sleep(0.3)
    _render_analysis_progress(placeholder, "正在评估申请竞争力")
    time.sleep(0.3)
    _render_analysis_progress(placeholder, "正在生成解读")

    buffer = ""
    for chunk in agent.stream(ctx):
        buffer += chunk or ""
    _render_analysis_progress(placeholder, "分析完成")

    result = agent.parse_stream_result() or agent._parse_response(buffer)
    st.session_state["explain_requested"] = False

    if result and result.get("overview"):
        placeholder.empty()
        st.session_state["explain_cache"][cache_key] = result
        _show_explanation(result)
    else:
        placeholder.empty()
        st.caption("解读暂不可用，可以稍后重试。")


def _render_analysis_progress(placeholder, text: str) -> None:
    placeholder.markdown(
        f'<div style="border-left:1.5px solid var(--hk-cyan);padding:0.5rem 0.8rem;'
        f'color:var(--hk-slate-500);font-size:0.85rem;font-style:italic">'
        f"{text}"
        '<span class="hk-thought-wait">'
        '<span class="hk-thought-wait-d1">.</span>'
        '<span class="hk-thought-wait-d2">.</span>'
        '<span class="hk-thought-wait-d3">.</span>'
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _build_explain_cache_key(
    *,
    sim: list,
    cross: list,
    unified: list,
    background_major: str,
    gpa: float,
    language_score: float,
    language_type: str,
    experience_details: dict | None,
) -> str:
    key_data = {
        "sim": _compact_results(sim[:5]),
        "cross": _compact_results(cross[:3]),
        "unified": _compact_results(unified[:5]),
        "background_major": background_major,
        "gpa": gpa,
        "language": [language_type, language_score],
        "experience": _compact_experience(experience_details),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()


def _compact_results(items: list[dict]) -> list[tuple[str, str, float]]:
    return [
        (
            str(item.get("university", "")),
            str(item.get("major", "")),
            round(float(item.get("probability", 0) or 0), 4),
        )
        for item in items
        if isinstance(item, dict)
    ]


def _compact_experience(experience_details: dict | None) -> dict[str, int]:
    return {
        str(k): len(str(v or ""))
        for k, v in (experience_details or {}).items()
        if v
    }


def _show_explanation(explanation: dict) -> None:
    show_explanation(explanation)


def display_content(
    session_manager: SessionManager,
    page_state: Any,
    submitted: bool,
    session_key_has_predicted: str = DEFAULT_SESSION_KEYS.has_predicted,
    session_key_input_data: str = DEFAULT_SESSION_KEYS.input_data,
    session_key_predict_lock: str = DEFAULT_SESSION_KEYS.predict_lock,
    session_key_form_data_changed: str = DEFAULT_SESSION_KEYS.form_data_changed,
) -> None:
    if not session_manager.get(session_key_has_predicted, False):
        return

    current_input_data = session_manager.get(session_key_input_data)
    if not current_input_data:
        content_display_logger.warning("has_predicted 为 True，但 session_state 中缺少输入数据。")
        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_has_predicted: False, session_key_predict_lock: False})
        st.rerun()

    form_changed = session_manager.get(session_key_form_data_changed, False)
    if not submitted and form_changed:
        st.caption("您的输入已更改，当前显示的是先前输入的预测结果。请点击预测按钮获取最新结果。")

    res_model = session_manager.get("prediction_results")
    display_results_section(
        current_input_data,
        res_model.similarity_results,
        res_model.cross_major_results,
        res_model.user_specified_results,
        page_state.cases_df,
        submitted=submitted,
    )
    readiness = build_readiness_profile(current_input_data, res_model)
    render_readiness_card(readiness)
    _log_readiness_once(session_manager, readiness)

    if submitted and res_model and (res_model.similarity_results or res_model.cross_major_results):
        st.html('<hr class="hk-section-divider">')
        _render_ai_explanation(
            res_model,
            current_input_data.get("background_university", ""),
            current_input_data.get("background_major", ""),
            current_input_data.get("gpa", 0),
            current_input_data.get("language_score", 0),
            current_input_data.get("language_type", ""),
            current_input_data.get("experience_details"),
        )

    if not submitted and form_changed:
        session_manager.set(**{session_key_form_data_changed: False})


def _log_readiness_once(
    session_manager: SessionManager,
    readiness: dict[str, Any],
) -> None:
    key_data = {
        "score": readiness.get("score"),
    }
    key = hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    if session_manager.get("last_readiness_event_hash") == key:
        return
    session_manager.set(last_readiness_event_hash=key)
    log_interaction_event(
        "application_readiness",
        {
            "score": readiness.get("score"),
            "label": readiness.get("label"),
            "factor_scores": {
                f["name"]: f.get("score") for f in readiness.get("factors", [])
            },
        },
    )
