from __future__ import annotations

import streamlit as st

from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_INTERNAL_KEYS,
    DEFAULT_SESSION_KEYS,
    DEFAULT_UI_KEYS,
)
from src.pages.prediction.ui.page_state_machine import HKPagePhase

TIMELINE_PHASES = ["智能录入", "背景解析", "预测方案", "策略推荐", "方案报告"]

TIMELINE_STATE_INDEX: dict[str, int] = {
    "parse": 1,
    "predict": 2,
    "strategy": 3,
    "report": 4,
}


def timeline_phases() -> list[str]:
    return TIMELINE_PHASES


def mark_timeline_report_engaged(session_manager) -> None:
    session_manager.set(**{DEFAULT_INTERNAL_KEYS.hk_timeline_report_engaged: True})


def _is_report_engaged(session_manager) -> bool:
    keys = DEFAULT_INTERNAL_KEYS
    if st.session_state.get(keys.hk_pdf_bytes):
        return True
    if session_manager.get(keys.hk_timeline_report_engaged, False):
        return True
    if st.session_state.get("explain_generating"):
        return True
    return False


def _is_cross_faculty_pending(session_manager, phase: str) -> bool:
    if phase == HKPagePhase.AWAITING_CONFIRM:
        return True
    return bool(session_manager.get(DEFAULT_UI_KEYS.pending_cross_faculty_prediction, False))


def timeline_confirm_stage_done(session_manager) -> bool:
    return False


def timeline_current_index(session_manager) -> int:
    idx_map = TIMELINE_STATE_INDEX
    phase = session_manager.get(DEFAULT_UI_KEYS.hk_ui_phase, HKPagePhase.IDLE)
    has_predicted = session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False)
    has_lead_in = bool(session_manager.get(DEFAULT_FORM_KEYS.lead_in_form_summary, ""))
    editing = bool(session_manager.get(DEFAULT_INTERNAL_KEYS.edit_background, False))

    if editing and phase not in (HKPagePhase.RUNNING, HKPagePhase.AWAITING_CONFIRM):
        return idx_map["parse"] if has_lead_in else 0

    if phase == HKPagePhase.RUNNING:
        return idx_map["predict"]

    if _is_cross_faculty_pending(session_manager, phase):
        return idx_map["predict"]

    if has_predicted or phase == HKPagePhase.DONE:
        if _is_report_engaged(session_manager):
            return idx_map["report"]
        return idx_map["strategy"]

    if has_lead_in:
        return idx_map["parse"]

    return 0
