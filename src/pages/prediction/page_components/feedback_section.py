import streamlit as st

from src.utils.logger import setup_logger

page_components_logger = setup_logger("page3", "prediction")


def display_feedback_section(session_id: str) -> None:
    """显示用户反馈组件（需要 Streamlit >= 1.28.0）"""
    try:
        if hasattr(st, "feedback"):
            feedback = st.feedback("thumbs", key=f"feedback_{session_id}")
            if feedback is not None:
                feedback_text = "满意" if feedback == 1 else "不满意"
                page_components_logger.info(f"用户反馈: {feedback_text}, session_id: {session_id}")
        else:
            # Streamlit 版本可能较旧，静默处理
            page_components_logger.debug("当前 Streamlit 版本不支持 feedback 组件")
    except Exception as e:
        page_components_logger.error(f"反馈组件显示失败: {e}", exc_info=True)
