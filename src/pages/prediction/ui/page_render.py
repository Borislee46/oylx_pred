from __future__ import annotations

import html
from typing import Any
from urllib.parse import urlparse

import streamlit as st

from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_INTERNAL_KEYS,
)
from src.pages.prediction.ui.background_summary import build_background_summary
from src.pages.prediction.ui.lead_in_ui import (
    render_lead_in_actions,
    render_lead_in_ghost,
)
from src.pages.prediction.ui.timeline import (
    timeline_current_index,
    timeline_phases,
)
from src.pages.prediction.ui.ui_elements import (
    display_back_to_homepage,
    display_feedback_section,
)
from src.utils import SUPPORT_EMAIL
from src.utils.analytics import track as _track
from src.utils.env_config_loader import load_app_config


def render_page_footer(session_manager: Any) -> None:
    cfg = load_app_config()
    base_url = cfg["STREAMLIT_APP_BASE_URL"]
    app_path = urlparse(base_url).path.rstrip("/")
    tech_report_url = f"{app_path}/tech_report"

    st.html(
        '<div class="hk-footer">'
        '<span style="font-size:0.82rem;color:var(--hk-slate-400);letter-spacing:0.04em;font-weight:500">'
        "Signals &middot; 留学择校系统"
        "</span>"
        '<div class="hk-footer-dot"></div>'
        f'<a href="{tech_report_url}" target="_blank" style="font-size:0.78rem;color:var(--hk-slate-300)">'
        "技术报告"
        "</a>"
        '<div class="hk-footer-dot"></div>'
        f'<a href="mailto:{SUPPORT_EMAIL}" style="font-size:0.78rem;color:var(--hk-slate-300)">'
        f"技术支持：{SUPPORT_EMAIL}"
        "</a>"
        "</div>"
    )
    display_feedback_section(session_manager.get("session_id"))
    display_back_to_homepage()


def render_background_summary_bar(
    session_manager: Any,
) -> None:
    label = "本次方案基于"
    with st.container(key="hk_summary_bar"):
        col_info, col_edit = st.columns([5, 1])
        with col_info:
            st.html(
                '<div class="hk-summary-inner">'
                f'<span class="hk-summary-label">{label}</span>'
                f'<span class="hk-summary-text">{build_background_summary(session_manager)}</span>'
                "</div>"
            )
        with col_edit:
            if st.button(
                "修改背景",
                type="tertiary",
                width="content",
                icon=":material/edit:",
                key="hk_edit_bg_btn",
            ):
                session_manager.set(
                    **{DEFAULT_INTERNAL_KEYS.edit_background: True},
                    submitted=False,
                    form_data_changed=False,
                )
                st.rerun()


def render_lead_in_section(
    session_manager: Any,
) -> None:
    render_lead_in_ghost(session_manager)
    render_lead_in_actions(session_manager)

    if session_manager.pop(DEFAULT_INTERNAL_KEYS.lead_in_processed, False):
        session_manager.set(**{DEFAULT_INTERNAL_KEYS.auto_submit_lead_in: True})
        st.session_state["form_expander"] = True

        _missing = session_manager.get(DEFAULT_FORM_KEYS.lead_in_missing_fields, None) or []
        _low_conf = (
            session_manager.get(DEFAULT_INTERNAL_KEYS.lead_in_low_confidence_labels, None) or []
        )
        _track(
            "lead_in_complete",
            fields_missing=len(_missing),
            low_confidence=len(_low_conf),
            auto_submit=True,
        )


def render_timeline_rail(
    session_manager: Any,
) -> None:
    phases = timeline_phases()
    current = timeline_current_index(session_manager)

    parts: list[str] = []
    for i, label in enumerate(phases):
        state = "done" if i < current else ("active" if i == current else "")
        parts.append(
            f'<div class="hk-timeline-dot {state}" data-label="{html.escape(label)}"></div>'
        )
        if i < len(phases) - 1:
            seg_state = "done" if i < current else ""
            parts.append(f'<div class="hk-timeline-segment {seg_state}"></div>')

    st.html(
        '<div class="hk-timeline-rail">'
        '<div class="hk-timeline-track">' + "".join(parts) + "</div>"
        "</div>"
    )
