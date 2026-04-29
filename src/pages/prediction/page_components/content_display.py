import hashlib
import json
import time
from typing import Any

import streamlit as st

from src.agent.context import StudentContext
from src.agent.registry import AgentRegistry
from src.pages.prediction.ai_report import (
    render_ai_section,
    render_ai_section_streaming,
    render_static_frame,
)
from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS
from src.pages.prediction.page_components.result_section import display_results_section
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

content_display_logger = setup_logger("page3", "prediction")


def _render_ai_explanation(
    prediction_results,
    input_data: dict[str, Any],
) -> None:
    background_university = input_data.get("background_university", "")
    background_major = input_data.get("background_major", "")
    gpa = float(input_data.get("gpa", 0) or 0)
    language_score = float(input_data.get("language_score", 0) or 0)
    language_score_raw = float(input_data.get("language_score_raw", 0) or 0)
    language_type = str(input_data.get("language_type", ""))
    experience_details = input_data.get("experience_details")
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
        render_static_frame(input_data, sim, cross, [])
        render_ai_section(cached)
        return

    # Manual trigger — no frame until clicked
    if "explain_requested" not in st.session_state:
        st.session_state["explain_requested"] = False

    if not st.session_state["explain_requested"]:
        c_left, _ = st.columns([2, 5])
        with c_left:
            if st.button("AI 选校解读", key="explain_btn", width="stretch"):
                st.session_state["explain_requested"] = True
                st.rerun()
        return

    render_static_frame(input_data, sim, cross, [])
    ctx = StudentContext(
        stage="match",
        background_university=background_university,
        background_major=background_major,
        gpa=gpa,
        language_score=language_score,
        language_score_raw=language_score_raw,
        language_type=language_type or "",
        experience_details=experience_details or {},
        prediction_results={
            "similarity_results": sim,
            "cross_major_results": cross,
            "unified_results": unified,
        },
    )
    agent = AgentRegistry.get("explain")

    stream_placeholder = st.empty()
    with stream_placeholder.container():
        render_ai_section_streaming("")
    buffer = ""
    last_update = 0.0
    for chunk in agent.stream(ctx):
        buffer += chunk or ""
        now = time.monotonic()
        if now - last_update < 0.08:
            continue
        partial = _try_extract_overview(buffer)
        with stream_placeholder.container():
            render_ai_section_streaming(partial or buffer[:300])
        last_update = now

    result = agent.parse_stream_result() or agent._parse_response(buffer)
    st.session_state["explain_requested"] = False

    if result and result.get("overview"):
        st.session_state["explain_cache"][cache_key] = result
        with stream_placeholder.container():
            render_ai_section(result)
    else:
        stream_placeholder.empty()
        st.caption("解读暂不可用，稍后重试。")


def _try_extract_overview(text: str) -> str:
    """Try to extract partial overview from streaming JSON."""
    import re

    m = re.search(r'"overview"\s*:\s*"([^"]*)', text)
    return m.group(1) if m else ""


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
    return {str(k): len(str(v or "")) for k, v in (experience_details or {}).items() if v}


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
    if res_model and (res_model.similarity_results or res_model.cross_major_results):
        st.html('<hr class="hk-section-divider">')
        _render_ai_explanation(res_model, current_input_data)

    if not submitted and form_changed:
        session_manager.set(**{session_key_form_data_changed: False})
