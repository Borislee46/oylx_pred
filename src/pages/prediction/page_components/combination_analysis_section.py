from typing import Any, Optional

import pandas as pd

from src.pages.prediction.admission_probability_calculator import (
    AdmissionProbabilityCalculator,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel, SessionManager

page_components_logger = setup_logger("page3", "prediction")


def display_combination_analysis_section(
    session_manager: SessionManager,
    prediction_results: Optional[PredictionResultModel],
    current_input_data: dict[str, Any],
    cases_df: pd.DataFrame,
) -> None:
    """显示组合分析部分，允许用户选择学校专业组合进行概率计算"""
    if not prediction_results:
        return

    sim_results = prediction_results.similarity_results or []
    cross_results = prediction_results.cross_major_results or []
    user_results = prediction_results.user_specified_results or []

    # 简化条件判断
    if not any([sim_results, cross_results, user_results]):
        return

    admission_calculator_ui = AdmissionProbabilityCalculator(session_manager)

    selected_results, selected_probabilities = admission_calculator_ui.display_school_selection(
        similarity_results=sim_results,
        cross_major_results=cross_results,
        user_specified_results=user_results,
        gpa=current_input_data.get("gpa"),
        language_score=current_input_data.get("language_score"),
        disabled_status=False,
    )
