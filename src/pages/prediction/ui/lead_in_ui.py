from __future__ import annotations

import html
import re
import time
from typing import Any

import streamlit as st

from src.agent.context import StudentContext
from src.agent.lead_in.dispatcher import (
    DISMISS_KEY,
    IN_PROGRESS_KEY,
    INTENT_BLOCKED_KEY,
    PENDING_KEY,
    PROGRESS_STEPS_KEY,
    PROGRESS_TEXT_KEY,
    PROGRESS_VARIANT_KEY,
    RETRY_COUNT_KEY,
    RUNNING_TS_KEY,
)
from src.agent.lead_in.state_machine import LeadInTurnStateMachine
from src.pages.prediction.ui.ghost_text_area import ghost_text_area
from src.pages.prediction.ui.lead_in_echo import (
    build_field_chips,
    sanitize_feedback,
)
from src.pages.prediction.ui.lead_in_progress import entries_to_detail_lines
from src.pages.prediction.ui.lead_in_wait import render_lead_in_wait
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_ghost_config_cache: dict[str, str] | None = None
_ghost_config_warned: bool = False


def _get_ghost_config() -> dict[str, str]:
    global _ghost_config_cache, _ghost_config_warned
    if _ghost_config_cache is not None:
        return _ghost_config_cache
    cfg = load_app_config()
    api_key = cfg.get("GHOST_API_KEY", "")
    if not api_key and not _ghost_config_warned:
        _ghost_config_warned = True
        logger.warning(
            "GHOST_API_KEY 未配置：幽灵补全将仅提供规则补全，不发 LLM 请求。"
            "注意：不要回退到 OPEN_AI_API_KEY——共享凭据会被下发到浏览器。"
        )
    _ghost_config_cache = {
        "api_key": api_key,
        "base_url": cfg.get("GHOST_API_BASE_URL", "https://api.deepseek.com/beta"),
        "model": cfg.get("GHOST_API_MODEL", "deepseek-v4-flash"),
    }
    return _ghost_config_cache


_LAST_APPLIED_KEY = "_lead_in_last_applied"
_LEAD_IN_RUNNING_TS = RUNNING_TS_KEY
_LEAD_IN_RETRY_COUNT = RETRY_COUNT_KEY
_LEAD_IN_PROGRESS_STEPS = PROGRESS_STEPS_KEY
_LEAD_IN_PROGRESS_TEXT = PROGRESS_TEXT_KEY
_IN_PROGRESS_KEY = IN_PROGRESS_KEY
_DISMISS_KEY = DISMISS_KEY


def _inline_bold_markdown(line: str) -> str:
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", line):
        out.append(html.escape(line[pos : m.start()]))
        out.append(f"<strong>{html.escape(m.group(1))}</strong>")
        pos = m.end()
    out.append(html.escape(line[pos:]))
    return "".join(out)


def _format_agent_feedback_html(text: str) -> str:
    paragraphs: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        paragraphs.append(
            f'<p class="hk-lead-in-feedback-p"'
            f' style="font-size:0.82rem;line-height:1.55;margin:0 0 0.35rem;">'
            f"{_inline_bold_markdown(line)}</p>"
        )
    return (
        "".join(paragraphs)
        if paragraphs
        else f'<p class="hk-lead-in-feedback-p">{html.escape(text)}</p>'
    )


def _render_chips_html(applied: dict[str, Any], low_conf: dict[str, Any] | None = None) -> str:
    chips_data = build_field_chips(applied, low_confidence=low_conf)
    if not chips_data:
        return ""
    spans: list[str] = []
    for c in chips_data:
        label = html.escape(c["label"])
        value = html.escape(c["value"])
        is_low = c.get("confidence") == "low"
        cls = "hk-field-chip hk-field-chip-low-conf" if is_low else "hk-field-chip"
        label_display = f"{label} ⚠" if is_low else label
        spans.append(
            f'<span class="{cls}">'
            f'<span class="hk-chip-label">{label_display}</span>'
            f'<span class="hk-chip-value">{value}</span>'
            f"</span>"
        )
    return f'<div class="hk-field-chip-grid">{"".join(spans)}</div>'


def _feedback_card_title(text: str) -> str:
    if any(k in text for k in ("补充", "还差", "还需要", "请告诉我", "方便说", "缺")):
        return "还需补充"
    if any(k in text for k in ("记下", "已记录", "整理", "识别到", "确认")):
        return "背景摘要"
    return "顾问说明"


def _render_feedback_html(text: str) -> None:
    text = sanitize_feedback(text)
    if not text:
        return
    title = _feedback_card_title(text)
    body = _format_agent_feedback_html(text)
    st.html(
        f'<div class="hk-lead-in-feedback">'
        f'<div class="hk-lead-in-feedback-title">{html.escape(title)}</div>'
        f'<div class="hk-lead-in-feedback-body">{body}</div>'
        f"</div>"
    )


def _render_clarify_bubbles(questions: list[str]) -> None:
    items = [str(q).strip() for q in questions if q and str(q).strip()][:4]
    if not items:
        return
    st.markdown(
        '<div class="hk-clarify-label" style="font-size:0.78rem;color:var(--hk-slate-500);margin:4px 0">'
        "可继续补充：</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(min(len(items), 2))
    for i, q in enumerate(items):
        if cols[i % len(cols)].button(q, key=f"lead_in_clarify_{hash(q) & 0xFFFF}"):
            st.session_state["lead_in_ghost_text"] = q
            # 版本号递增，通知前端 ghost 组件把输入框文本替换为气泡内容
            st.session_state["_ghost_text_revision"] = (
                int(st.session_state.get("_ghost_text_revision", 0)) + 1
            )
            st.session_state["_ghost_analyze_text"] = q
            st.rerun(scope="app")


def _render_persisted(session_manager: Any) -> None:
    try:
        sm = LeadInTurnStateMachine(session_manager)
        state = sm.get_state()
        has_sm = True
    except Exception:
        state = None
        has_sm = False
        sm = None

    dismissed = (
        state.feedback_dismissed
        if has_sm and state is not None
        else st.session_state.get(_DISMISS_KEY, False)
    )
    if dismissed:
        return

    in_progress = (
        sm.is_extracting() or sm.is_gating()
        if has_sm
        else st.session_state.get(_IN_PROGRESS_KEY, False)
    )
    if in_progress:
        details = (
            list(state.progress_details)
            if has_sm and state is not None and state.progress_details
            else st.session_state.get("_lead_in_progress_details") or []
        )
        if not details and has_sm and state is not None and state.last_trace:
            details = entries_to_detail_lines(state.last_trace)
        render_lead_in_wait(
            state.progress_steps
            if has_sm and state is not None
            else st.session_state.get(_LEAD_IN_PROGRESS_STEPS, []),
            state.progress_text
            if has_sm and state is not None
            else st.session_state.get(_LEAD_IN_PROGRESS_TEXT, ""),
            elapsed=time.time() - st.session_state.get(_LEAD_IN_RUNNING_TS, time.time()),
            retry=state.retry_count
            if has_sm and state is not None
            else st.session_state.get(_LEAD_IN_RETRY_COUNT, 0),
            variant=state.progress_variant
            if has_sm and state is not None
            else st.session_state.get(PROGRESS_VARIANT_KEY, "default"),
            details=details,
            path_hint=str(st.session_state.get("_lead_in_path_hint") or ""),
            ctx=st.session_state.get("lead_in_ctx"),
            applied=(
                state.last_applied_fields
                if has_sm and state is not None and state.last_applied_fields
                else st.session_state.get(_LAST_APPLIED_KEY)
            )
            or {},
            sse_port=int(st.session_state.get("_lead_in_sse_port") or 0),
            sse_run_id=str(st.session_state.get("_lead_in_sse_run_id") or ""),
            sse_url=str(st.session_state.get("_lead_in_sse_url") or ""),
        )
        return

    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())
    feedback = sanitize_feedback(ctx.quick_assessment or "")

    intent_blocked = (
        state.intent_blocked
        if has_sm and state is not None
        else st.session_state.get(INTENT_BLOCKED_KEY, False)
    )
    if intent_blocked and feedback:
        st.warning(feedback, icon="⚠")
        return

    applied = (
        state.last_applied_fields
        if has_sm and state is not None
        else st.session_state.get(_LAST_APPLIED_KEY)
    ) or {}
    if not feedback and not applied:
        return

    low_conf = (
        state.low_confidence_display
        if has_sm and state is not None
        else st.session_state.get("_lead_in_low_conf_display")
    ) or {}

    if applied:
        chips_html = _render_chips_html(applied, low_conf)
        if chips_html:
            st.html(chips_html)

    if low_conf:
        labels = [str(v) for v in low_conf.values() if v]
        if labels:
            st.caption("以下字段未能精确匹配，请确认：" + "、".join(labels))

    clarify = (
        state.clarifying_questions
        if has_sm and state is not None
        else st.session_state.get("_lead_in_clarify_questions")
    ) or []
    if clarify:
        _render_clarify_bubbles(clarify)

    if feedback:
        _render_feedback_html(feedback)


_PLACEHOLDER = "例如：本科中山大学金融，均分85，雅思6.5，想去港新读金融或商科，有两段实习"
_DONE_PLACEHOLDER = "方案已生成。粘贴新学生背景开始下一轮"


def render_lead_in_ghost(session_manager: Any) -> str:
    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())
    ghost_key = "lead_in_ghost_text"
    if ghost_key not in st.session_state:
        st.session_state[ghost_key] = ctx.raw_input or ""

    consumed = session_manager.get("lead_in_consumed", False)
    placeholder = _DONE_PLACEHOLDER if consumed else _PLACEHOLDER

    ghost_cfg = _get_ghost_config()
    try:
        busy = LeadInTurnStateMachine(session_manager).is_busy()
    except Exception:
        busy = bool(st.session_state.get(IN_PROGRESS_KEY, False))

    if busy:
        logger.info("LEAD_IN_GHOST | busy=True pending=%s", bool(st.session_state.get(PENDING_KEY)))

    returned = ghost_text_area(
        enabled=bool(ghost_cfg["api_key"]),
        api_model=ghost_cfg["model"],
        placeholder=placeholder,
        initial_text=st.session_state[ghost_key],
        height=80,
        key="lead_in_ghost_component",
        lead_in_busy=busy,
    )
    if returned:
        st.session_state[ghost_key] = returned
    return st.session_state[ghost_key]


def render_lead_in_actions(session_manager: Any) -> None:
    from src.pages.prediction.ui.lead_in_dispatch import run_lead_in_dispatch

    try:
        run_lead_in_dispatch(session_manager)
    finally:
        _render_persisted(session_manager)
