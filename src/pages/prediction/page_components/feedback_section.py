import streamlit as st

from src.utils.logger import setup_logger

page_components_logger = setup_logger("page3", "prediction")


def display_feedback_section(session_id: str) -> None:
    key = f"feedback_{session_id}"
    prev_key = f"{key}_prev"
    current = st.feedback("thumbs", key=key)
    prev = st.session_state.get(prev_key)
    if current != prev:
        if current is None and prev is not None:
            prev_text = "满意" if prev == 1 else "不满意"
            page_components_logger.info(f"用户取消反馈: {prev_text}, session_id: {session_id}")
        elif current is not None:
            current_text = "满意" if current == 1 else "不满意"
            page_components_logger.info(f"用户反馈: {current_text}, session_id: {session_id}")

            if current == 1:
                st.toast("感谢您的肯定！我们会继续努力！")
            else:
                st.toast("收到您的反馈，我们会持续改进！")

        st.session_state[prev_key] = current
    elif prev_key not in st.session_state:
        st.session_state[prev_key] = current
