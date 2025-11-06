from typing import Any

import streamlit as st

from src.pages.prediction.page_components.combination_analysis_section import (
    display_combination_analysis_section,
)
from src.pages.prediction.page_components.result_section import display_results_section
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

content_display_logger = setup_logger("page3", "prediction")


def display_content(
    session_manager: SessionManager,
    page_state: Any,
    submitted: bool,
    session_key_has_predicted: str,
    session_key_input_data: str,
    session_key_predict_lock: str,
    session_key_form_data_changed: str,
) -> None:
    if session_manager.get(session_key_has_predicted, False):
        current_input_data = session_manager.get(session_key_input_data)
        if not current_input_data:
            st.warning("会话状态中缺少有效的输入数据，无法显示预测结果。请重新提交。")
            content_display_logger.warning(
                "has_predicted 为 True，但 session_state 中缺少有效的输入数据。"
            )
            reset_prediction_results(session_manager)
            session_manager.set(
                **{session_key_has_predicted: False, session_key_predict_lock: False}
            )
            # 使用 rerun 但添加保护，避免无限循环
            try:
                st.rerun()
            except Exception as e:
                content_display_logger.error(f"页面刷新失败: {e}")
            return

        prediction_results_model = session_manager.get("prediction_results")
        if prediction_results_model is None:
            st.error("预测结果模型不存在，无法显示结果。")
            content_display_logger.error("prediction_results_model 为 None")
            return

        sim_results_display = prediction_results_model.similarity_results
        cross_results_display = prediction_results_model.cross_major_results
        user_specified_results_display = prediction_results_model.user_specified_results

        # 合并重复的条件判断
        form_data_changed = session_manager.get(session_key_form_data_changed, False)
        if not submitted and form_data_changed:
            st.warning(
                "您的输入已更改，当前显示的是基于先前输入的预测结果。请点击预测按钮获取最新结果。"
            )

        display_results_section(
            current_input_data,
            sim_results_display,
            cross_results_display,
            user_specified_results_display,
            page_state.cases_df,
            submitted=submitted,
        )

        if not submitted and form_data_changed:
            session_manager.set(**{session_key_form_data_changed: False})

        display_combination_analysis_section(
            session_manager,
            prediction_results_model,
            current_input_data,
            page_state.cases_df,
        )

    else:
        st.info("请填写表单并点击预测以查看结果。")
