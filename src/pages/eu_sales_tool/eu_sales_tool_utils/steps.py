import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_SCORE_RANGE,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_LANGUAGE,
    KEY_SELECTED_PROGRAM,
    KEY_SELECTED_STUDY_PROGRAM,
)


def reset_selections_from_step(step):
    if step <= 1:
        st.session_state[KEY_SELECTED_COUNTRY] = None
    if step <= 2:
        st.session_state[KEY_SELECTED_LANGUAGE] = None
    if step <= 3:
        st.session_state[KEY_SCORE_RANGE] = None
    if step <= 4:
        st.session_state[KEY_SELECTED_STUDY_PROGRAM] = None
    if step <= 5:
        st.session_state[KEY_SELECTED_PROGRAM] = None
