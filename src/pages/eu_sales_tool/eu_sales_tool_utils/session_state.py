import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_CURRENT_STEP,
    KEY_SCORE_KEY,
    KEY_SCORE_RANGE,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_LANGUAGE,
    KEY_SELECTED_PROGRAM,
    KEY_SELECTED_STUDY_PROGRAM,
    KEY_SHOW_ADMIN_PANEL,
)

from .modify_permission_check import check_admin_permission


def initialize_session_state():
    st.session_state.setdefault(KEY_CURRENT_STEP, 1)
    st.session_state.setdefault(KEY_SELECTED_COUNTRY, None)
    st.session_state.setdefault(KEY_SELECTED_LANGUAGE, None)
    st.session_state.setdefault(KEY_SCORE_RANGE, None)
    st.session_state.setdefault(KEY_SCORE_KEY, None)
    st.session_state.setdefault(KEY_SELECTED_STUDY_PROGRAM, None)
    st.session_state.setdefault(KEY_SELECTED_PROGRAM, None)
    if check_admin_permission():
        st.session_state.setdefault(KEY_SHOW_ADMIN_PANEL, False)
