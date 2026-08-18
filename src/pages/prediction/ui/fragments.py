from __future__ import annotations

from dataclasses import asdict
from typing import Any

import streamlit as st

from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_INTERNAL_KEYS,
    DEFAULT_SESSION_KEYS,
    DEFAULT_UI_KEYS,
    PendingSubmissionData,
)
from src.pages.prediction.input_form import create_input_form
from src.pages.prediction.results_handler import clear_pending_prediction_state
from src.pages.prediction.ui.content_display import display_content
from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine
from src.pages.prediction.ui.scroll_utils import scroll_to_anchor
from src.utils import SUPPORT_EMAIL


def _snapshot_submittable(input_data: dict) -> bool:
    if not input_data:
        return False
    if not input_data.get("background_university"):
        return False
    if not input_data.get("background_major"):
        return False
    gpa = input_data.get("gpa")
    lang = input_data.get("language_score")
    return isinstance(gpa, (int, float)) and gpa > 0 and isinstance(lang, (int, float)) and lang > 0


@st.fragment
def form_fragment(
    session_manager: Any,
    page_state: Any,
) -> None:
    sm = PageStateMachine(session_manager)

    _fast_attempted = False
    if session_manager.get("submitted", False):
        _fast_attempted = True
        is_new_submission, input_data, all_unis, all_majors, original_form = create_input_form(
            session_manager,
            page_state.cases_df,
            parent_container=None,
            wrap_container=False,
        )
        if is_new_submission and input_data.get("background_university"):
            submission_data = PendingSubmissionData(
                input_data=input_data,
                all_universities=all_unis,
                all_majors=all_majors,
                original_form=original_form,
            )
            clear_pending_prediction_state(session_manager)
            sm.transition(HKPagePhase.RUNNING)
            session_manager.set(
                submitted=False,
                **{DEFAULT_INTERNAL_KEYS.pending_submission_data: asdict(submission_data)},
            )
            st.rerun(scope="app")
        session_manager.set(submitted=False, prediction_submit_lock=False)

    if not _fast_attempted:
        has_lead_in = bool(session_manager.get(DEFAULT_FORM_KEYS.lead_in_form_summary, ""))
        form_title = "手动确认学生背景表单" if has_lead_in else "手动补充学生背景表单"

        if "form_expander" not in st.session_state:
            st.session_state["form_expander"] = False

        exp = st.expander(form_title, key="form_expander", on_change="rerun")
        auto_submit = session_manager.get(DEFAULT_INTERNAL_KEYS.auto_submit_lead_in, False)

        if exp.open or auto_submit:
            with exp:
                missing = session_manager.get(DEFAULT_FORM_KEYS.lead_in_missing_fields, None) or []
                if missing:
                    missing_str = "、".join(missing)
                    st.markdown(
                        f'<p style="color:var(--hk-slate-400);font-size:0.8rem;margin:0 0 0.5rem 0;">'
                        f"AI 未识别: {missing_str} → 请手动补充</p>",
                        unsafe_allow_html=True,
                    )
                low_conf_labels = (
                    session_manager.get(DEFAULT_INTERNAL_KEYS.lead_in_low_confidence_labels, None)
                    or []
                )
                if low_conf_labels:
                    labels_str = "、".join(low_conf_labels)
                    st.markdown(
                        f'<p style="color:var(--hk-amber-400);font-size:0.8rem;margin:0 0 0.5rem 0;">'
                        f"⚠ AI 匹配置信度较低: {labels_str} → 请核对后手动选择</p>",
                        unsafe_allow_html=True,
                    )
                is_new_submission, input_data, all_unis, all_majors, original_form = (
                    create_input_form(
                        session_manager,
                        page_state.cases_df,
                        parent_container=None,
                        wrap_container=False,
                    )
                )
        else:
            is_new_submission = False
            input_data, all_unis, all_majors, original_form = {}, [], [], {}
    else:
        is_new_submission = False
        input_data, all_unis, all_majors, original_form = {}, [], [], {}

    can_submit = _snapshot_submittable(input_data)

    if is_new_submission and can_submit:
        submission_data = PendingSubmissionData(
            input_data=input_data,
            all_universities=all_unis,
            all_majors=all_majors,
            original_form=original_form,
        )
        clear_pending_prediction_state(session_manager)
        sm.transition(HKPagePhase.RUNNING)
        session_manager.set(
            submitted=False,
            **{DEFAULT_INTERNAL_KEYS.pending_submission_data: asdict(submission_data)},
        )
        st.rerun(scope="app")

    if session_manager.pop(DEFAULT_INTERNAL_KEYS.auto_submit_lead_in, False):
        if can_submit:
            submission_data = PendingSubmissionData(
                input_data=input_data,
                all_universities=all_unis,
                all_majors=all_majors,
                original_form=original_form or {},
            )
            clear_pending_prediction_state(session_manager)
            sm.transition(HKPagePhase.RUNNING)
            session_manager.set(
                **{DEFAULT_INTERNAL_KEYS.pending_submission_data: asdict(submission_data)}
            )
            st.rerun(scope="app")


@st.fragment
def results_fragment(
    session_manager: Any,
    page_state: Any,
) -> None:
    sm = PageStateMachine(session_manager)

    if sm.is_error():
        user_message = session_manager.pop(DEFAULT_FORM_KEYS.user_message, None)
        st.error(user_message or f"预测未完成，请稍后重试。如反复出现，请联系：{SUPPORT_EMAIL}")
        last_error = session_manager.get(DEFAULT_UI_KEYS.hk_last_error)
        if last_error:
            st.caption(f"技术细节：{last_error}")
        return

    if not session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False):
        return

    submitted = session_manager.pop(DEFAULT_INTERNAL_KEYS.results_fresh_submission, False)

    if submitted:
        scroll_to_anchor("hk-results-anchor", delay_ms=150)
        user_message = session_manager.get(DEFAULT_FORM_KEYS.user_message)
        if user_message:
            st.info(user_message, icon=":material/info:")
            session_manager.delete(DEFAULT_FORM_KEYS.user_message)

    display_content(session_manager, page_state, submitted)
