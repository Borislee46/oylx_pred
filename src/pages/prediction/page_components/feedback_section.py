import streamlit as st

from src.utils.logger import setup_logger

page_components_logger = setup_logger("page3", "prediction")


def display_feedback_section(session_id):
    if hasattr(st, "feedback"):
        feedback = st.feedback("thumbs", key=f"feedback_{session_id}")
        if feedback is not None:
            page_components_logger.info(f"用户反馈: {'满意' if feedback == 1 else '不满意'}")
