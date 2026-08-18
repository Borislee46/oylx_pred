from functools import lru_cache

import pandas as pd
import streamlit as st

from src.pages.prediction.app_data import load_raw_cases_data, load_school_major_details_df
from src.pages.prediction.core.ui_messages import (
    CROSS_FACULTY_MESSAGES,
    CROSS_FACULTY_MESSAGES_SALES,
)
from src.pages.prediction.results_handler import clear_pending_prediction_state
from src.utils.session_manager import SessionManager


@st.dialog(CROSS_FACULTY_MESSAGES["dialog_title"], width="small")
def cross_faculty_confirm_dialog(
    session_manager: SessionManager, background_faculty: str, target_faculties: set[str]
) -> None:
    msgs = CROSS_FACULTY_MESSAGES_SALES
    target_str = "、".join(sorted(target_faculties))
    st.markdown(
        msgs["dialog_body"].format(bg_faculty=background_faculty, target_faculties=target_str)
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
            msgs["confirm_button"],
            type="primary",
            width="stretch",
            key="cross_faculty_confirm_btn",
            shortcut="Enter",
        ):
            session_manager.set(
                cross_faculty_confirmed=True,
                cross_faculty_cancelled=False,
                pending_cross_faculty_prediction=True,
            )
            st.rerun()

    with col2:
        if st.button(
            msgs["cancel_button"],
            width="stretch",
            key="cross_faculty_cancel_btn",
            shortcut="Esc",
        ):
            clear_pending_prediction_state(session_manager)
            from src.pages.prediction.ui.page_state_machine import HKPagePhase, PageStateMachine

            sm = PageStateMachine(session_manager)
            sm.transition(HKPagePhase.IDLE)
            session_manager.set(
                cross_faculty_confirmed=False,
                cross_faculty_cancelled=True,
                prediction_submit_lock=False,
                submitted=False,
                hk_last_error=None,
            )
            st.rerun()


@lru_cache(maxsize=1)
def _get_major_to_faculty_map() -> dict[str, str]:
    details_df = load_school_major_details_df()
    if details_df is None or details_df.empty or "专业大类" not in details_df.columns:
        return {}

    result: dict[str, str] = {}
    df = details_df.dropna(subset=["专业大类"])

    if "专业英文名称_聚合" in df.columns:
        for major, faculty in zip(
            df["专业英文名称_聚合"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            if major.strip():
                result[major.strip()] = faculty.strip()

    if "专业英文名称" in df.columns:
        for major, faculty in zip(
            df["专业英文名称"].astype(str), df["专业大类"].astype(str), strict=True
        ):
            major_key = major.strip()
            if major_key and major_key not in result:
                result[major_key] = faculty.strip()

    return result


def quick_cross_faculty_check(
    background_major: str | None,
    selected_categories: list[str] | None,
    selected_majors: list[str] | None,
    cases_df: pd.DataFrame | None = None,
) -> tuple[bool, str | None, set[str], bool]:
    selected_categories = selected_categories or []
    selected_majors = selected_majors or []

    if not background_major or (not selected_categories and not selected_majors):
        return False, None, set(), False

    if cases_df is None:
        cases_df = load_raw_cases_data()

    from src.pages.prediction.core.utils import get_background_faculty

    background_faculty = get_background_faculty(background_major, cases_df)
    if not background_faculty:
        return False, None, set(), False

    target_faculties: set[str] = set(selected_categories)

    cross_majors = []
    if selected_majors:
        major_to_faculty = _get_major_to_faculty_map()
        for major in selected_majors:
            major_str = str(major).strip()
            faculty = major_to_faculty.get(major_str)
            if faculty:
                target_faculties.add(faculty)
                if faculty != background_faculty:
                    cross_majors.append(major_str)

    has_cross = any(f and f != background_faculty for f in target_faculties)
    agent_approved = False

    return has_cross, background_faculty, target_faculties, agent_approved
