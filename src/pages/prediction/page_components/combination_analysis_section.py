import streamlit as st

from src.pages.prediction.admission_probability_calculator import AdmissionProbabilityCalculator
from src.utils.logger import setup_logger
from src.utils.session_manager import PredictionResultModel, SessionManager

page_components_logger = setup_logger("page3", "prediction")


def display_combination_analysis_section(
    session_manager: SessionManager,
    prediction_results: PredictionResultModel,
    current_input_data: dict,
    cases_df,
):
    sim_results = prediction_results.similarity_results if prediction_results else []
    cross_results = prediction_results.cross_major_results if prediction_results else []
    user_results = prediction_results.user_specified_results if prediction_results else []

    sim_results = sim_results or []
    cross_results = cross_results or []
    user_results = user_results or []

    if (
        (sim_results is not None and len(sim_results) > 0)
        or (cross_results is not None and len(cross_results) > 0)
        or (user_results is not None and len(user_results) > 0)
    ):
        try:
            admission_calculator_ui = AdmissionProbabilityCalculator(session_manager)

            selected_results, selected_probabilities = (
                admission_calculator_ui.display_school_selection(
                    similarity_results=sim_results,
                    cross_major_results=cross_results,
                    user_specified_results=user_results,
                    gpa=current_input_data.get("gpa"),
                    language_score=current_input_data.get("language_score"),
                    disabled_status=False,
                )
            )

        except Exception as e:
            error_msg = f"选校组合分析时发生错误: {str(e)}"
            page_components_logger.error(error_msg, exc_info=True)
            st.error(error_msg)
