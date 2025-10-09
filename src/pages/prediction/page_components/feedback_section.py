import streamlit as st

from src.utils.logger import setup_logger

page_components_logger = setup_logger("page3", "prediction")


def display_feedback_section(session_id):
    feedback_key = f"feedback_{session_id}_prediction_page"
    feedback_mapping = {0: "不满意", 1: "满意"}

    if hasattr(st, "feedback"):
        selected_feedback = st.feedback("thumbs", key=feedback_key)
        if selected_feedback is not None:
            feedback_value = feedback_mapping.get(selected_feedback, "未知选择")
            page_components_logger.info(f"用户反馈: {feedback_value}")
            if selected_feedback == 0:
                st.page_link(
                    "pages/hk_grad_feedback.py",
                    label="点击这里提供详细反馈，帮助我们改进产品，感谢您的支持！",
                )
            else:
                st.page_link(
                    "pages/hk_grad_feedback.py", label="感谢您的反馈！点击这里留下您的宝贵意见"
                )
    else:
        page_components_logger.warning(
            "st.feedback attribute 找不到. 反馈组件不会被渲染. 如果需要, 请更新 Streamlit 版本."
        )
