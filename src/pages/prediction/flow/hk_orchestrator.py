from __future__ import annotations

from typing import Any

import streamlit as st

from src.pages.prediction.handler_config import (
    DEFAULT_INTERNAL_KEYS,
    DEFAULT_SESSION_KEYS,
    DEFAULT_UI_KEYS,
    PREDICTION_STATE_PREFIXES,
    FormSubmissionContext,
    PendingSubmissionData,
)
from src.pages.prediction.results_handler import clear_pending_prediction_state
from src.pages.prediction.ui.handler import handle_form_submission
from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine
from src.pages.prediction.ui.progress_text import soften_progress_text, status_label
from src.pages.prediction.ui.scroll_utils import scroll_to_anchor
from src.pages.prediction.ui.ui_elements import (
    render_thought_bubble,
    render_thought_bubble_with_wait_pulse,
)
from src.utils.analytics import track as _track
from src.utils.logger import setup_logger

_orchestrator_logger = setup_logger("hk_orchestrator", "prediction")


def reset_prediction_ui_state(
    session_manager: Any,
    *,
    reset_cross_faculty_confirmed: bool = False,
    reset_cross_faculty_cancelled: bool = False,
    hk_ui_phase: str = HKPagePhase.IDLE,
    hk_last_error: str | None = None,
) -> None:
    clear_pending_prediction_state(
        session_manager,
        reset_cross_faculty_confirmed=reset_cross_faculty_confirmed,
        reset_cross_faculty_cancelled=reset_cross_faculty_cancelled,
    )
    state_machine = PageStateMachine(session_manager)
    state_machine.transition(hk_ui_phase)
    session_manager.set(
        submitted=False,
        hk_last_error=hk_last_error,
        **{DEFAULT_INTERNAL_KEYS.edit_background: False},
    )


def run_prediction(
    session_manager: Any,
    submission_data: dict,
    page_state: Any,
    progress_area: Any,
) -> None:
    for _prefix in PREDICTION_STATE_PREFIXES:
        session_manager.clear_prefix(_prefix)

    pending = (
        submission_data
        if isinstance(submission_data, PendingSubmissionData)
        else PendingSubmissionData(
            input_data=submission_data.get("input_data", {}),
            all_universities=submission_data.get("all_universities", []),
            all_majors=submission_data.get("all_majors", []),
            original_form=submission_data.get("original_form"),
        )
    )

    is_pending_retry = session_manager.get(DEFAULT_UI_KEYS.pending_cross_faculty_prediction, False)

    _orchestrator_logger.info(
        "开始预测 | is_retry=%s | 院校=%s 专业=%s",
        is_pending_retry,
        pending.input_data.get("background_university", "")[:40],
        pending.input_data.get("background_major", "")[:40],
    )

    submission_ctx = FormSubmissionContext(
        session_manager=session_manager,
        page_state=page_state,
        input_data_from_form=pending.input_data,
        all_universities_target=pending.all_universities,
        all_majors_target=pending.all_majors,
        original_form_data=pending.original_form,
        session_keys=DEFAULT_SESSION_KEYS,
    )

    try:
        with progress_area:
            scroll_to_anchor("hk-predict-start", delay_ms=150)

            with st.status(status_label("start"), expanded=True) as status:
                thought_placeholder = st.empty()
                logs: list[str] = []

                def progress_cb(text: str) -> None:
                    if not text:
                        return
                    text = soften_progress_text(str(text))
                    logs.append(text)
                    render_thought_bubble_with_wait_pulse(logs, thought_placeholder)

                state_machine = PageStateMachine(session_manager)
                state_machine.transition(HKPagePhase.RUNNING)
                session_manager.set(hk_last_error=None)
                try:
                    handle_form_submission(submission_ctx, progress_cb=progress_cb)
                finally:
                    if logs:
                        render_thought_bubble(logs, thought_placeholder)

                if state_machine.is_awaiting_confirm():
                    session_manager.set(prediction_submit_lock=False)
                    status.update(
                        label=status_label("running"),
                        state="running",
                        expanded=True,
                    )
                else:
                    if is_pending_retry:
                        clear_pending_prediction_state(
                            session_manager, reset_cross_faculty_cancelled=True
                        )
                    session_manager.set(prediction_submit_lock=False)
                    if session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False):
                        status.update(
                            label=status_label("complete"),
                            state="complete",
                            expanded=True,
                        )
                    else:
                        status.update(
                            label="预测未完成",
                            state="error",
                            expanded=True,
                        )

            if not state_machine.is_awaiting_confirm():
                if session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False):
                    state_machine.transition(HKPagePhase.DONE)
                    session_manager.set(
                        lead_in_consumed=True,
                        **{
                            DEFAULT_INTERNAL_KEYS.results_fresh_submission: True,
                            DEFAULT_INTERNAL_KEYS.edit_background: False,
                        },
                    )
                    _orchestrator_logger.info("预测完成并展示 | phase=done")
                else:
                    state_machine.transition(HKPagePhase.ERROR)
                    _orchestrator_logger.info("预测未产生结果，进入错误态 | phase=error")
    except Exception as e:
        reset_prediction_ui_state(
            session_manager,
            reset_cross_faculty_confirmed=is_pending_retry,
            reset_cross_faculty_cancelled=is_pending_retry,
            hk_ui_phase=HKPagePhase.ERROR,
            hk_last_error=str(e),
        )
        _track("prediction_error", error_type=type(e).__name__, error_snippet=str(e)[:200])
        _orchestrator_logger.exception("预测过程中发生异常")


def dispatch_prediction(
    session_manager: Any,
    page_state: Any,
    progress_area: Any,
) -> None:
    sm = PageStateMachine(session_manager)

    if sm.is_running():
        pending = session_manager.get(DEFAULT_INTERNAL_KEYS.pending_submission_data)
        if pending:
            session_manager.delete(DEFAULT_INTERNAL_KEYS.pending_submission_data)
            run_prediction(session_manager, pending, page_state, progress_area)
        return

    pending_cross = session_manager.get(DEFAULT_UI_KEYS.pending_cross_faculty_prediction, False)
    if pending_cross:
        pending_data = session_manager.get(DEFAULT_UI_KEYS.pending_prediction_data)
        if pending_data:
            sm.transition(HKPagePhase.RUNNING)
            session_manager.set(**{DEFAULT_INTERNAL_KEYS.pending_submission_data: pending_data})
            st.rerun()
        else:
            reset_prediction_ui_state(
                session_manager,
                reset_cross_faculty_confirmed=True,
                reset_cross_faculty_cancelled=True,
                hk_ui_phase=HKPagePhase.IDLE,
            )
        return
