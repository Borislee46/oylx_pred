import time
from contextlib import nullcontext

import streamlit as st

from src.pages.prediction.input_form_components import (
    FormStateManager,
    FormUIComponents,
    FormValidator,
    GPAConverter,
)
from src.pages.prediction.prediction_preparation.form_normalizer import (
    calculate_processed_gpa,
    calculate_processed_language_score,
    get_background_university_for_model,
    normalize_form_data_for_prediction,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.app_data_loader import load_school_base_data
from src.utils.logger import setup_logger
from src.utils.school_level_service import get_school_level_service
from src.utils.session_manager import SessionManager

form_logger = setup_logger("page3", "prediction")


def create_input_form(
    session_manager: SessionManager,
    cases_df,
    *,
    parent_container=None,
    wrap_container: bool = True,
    border: bool = True,
):
    FormStateManager.initialize_session_state(session_manager)

    disabled_status = session_manager.get("prediction_submit_lock", False)

    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    if session_manager.get("gpa_converter") is None:
        session_manager.set(gpa_converter=GPAConverter(session_manager.get("school_base_df")))
    gpa_converter = session_manager.get("gpa_converter")

    ui_components = FormUIComponents(session_manager)

    outer_ctx = parent_container if parent_container is not None else nullcontext()

    with outer_ctx:
        input_ctx = st.container(border=border) if wrap_container else nullcontext()
        with input_ctx:
            col1, col2 = st.columns([1, 1], gap="small")

            with col1:
                (
                    background_university,
                    selected_background_major_original,
                    background_major,
                ) = ui_components.render_background_section(cases_df)

                gpa_col, test_col = st.columns([2, 1], gap="medium")
                with gpa_col:
                    ui_components.render_gpa_section()
                with test_col:
                    exam_type, exam_score = ui_components.render_standardized_test_section()

                (
                    final_target_universities,
                    final_target_majors,
                    all_universities_target,
                    all_majors_target,
                ) = ui_components.render_target_section(cases_df)

            with col2:
                language_type, raw_language_score_value = ui_components.render_language_section()
                (
                    research_count,
                    award_count,
                    internship_count,
                    paper_count,
                    experience_details,
                ) = ui_components.render_experience_section()

            submit_button = ui_components.render_submit_button(disabled_status)

    if submit_button:
        form_data = {
            "target_majors": final_target_majors,
            "target_universities": final_target_universities,
            "background_university": background_university,
            "background_major_original": selected_background_major_original,
            "background_major": background_major,
            "gpa_raw": session_manager.get("gpa_raw_input"),
            "gpa_scale": session_manager.get("gpa_scale"),
            "exam_type": exam_type,
            "exam_score": exam_score,
            "language_type": language_type,
            "language_score_raw": raw_language_score_value,
            "language_score_input_error": session_manager.get("language_score_input_error", False),
            "research_count": research_count,
            "award_count": award_count,
            "internship_count": internship_count,
            "paper_count": paper_count,
            "experience_details": experience_details,
        }

        validation_errors = FormValidator.validate_form_data(form_data, gpa_converter)

        if validation_errors:
            error_messages = [str(err) for err in validation_errors]
            form_logger.warning(f"表单验证失败 - 错误信息: {error_messages}")
            for err in validation_errors:
                st.toast(str(err))
                time.sleep(0.3)
            session_manager.set(
                submitted=False, form_data_changed=False, prediction_submit_lock=False
            )
            reset_prediction_results(session_manager)
            st.rerun()
        else:
            session_manager.set(prediction_submit_lock=True)
            success, processed_input_data, all_unis, all_majors, original_form_data = (
                _process_successful_submission(
                    session_manager,
                    form_data,
                    cases_df,
                    all_universities_target,
                    all_majors_target,
                    gpa_converter,
                )
            )

            session_manager.set(
                _input_form_pending_submission={
                    "input_data": processed_input_data,
                    "all_unis": all_unis,
                    "all_majors": all_majors,
                    "original_form": original_form_data,
                }
            )
            st.rerun()

    pending_submission = session_manager.get("_input_form_pending_submission")
    if pending_submission:
        session_manager.set(_input_form_pending_submission=None)
        return (
            True,
            pending_submission["input_data"],
            pending_submission["all_unis"],
            pending_submission["all_majors"],
            pending_submission["original_form"],
        )

    return _get_current_form_state(
        session_manager,
        background_university,
        background_major,
        final_target_universities,
        final_target_majors,
        language_type,
        raw_language_score_value,
        research_count,
        award_count,
        internship_count,
        paper_count,
        experience_details,
        cases_df,
        all_universities_target,
        all_majors_target,
        gpa_converter,
        exam_type,
        exam_score,
    )


def _process_successful_submission(
    session_manager,
    form_data,
    cases_df,
    all_universities_target,
    all_majors_target,
    gpa_converter,
):
    from src.pages.prediction.page_data_loader import machine_learning_model

    page_state = machine_learning_model.resource_loader()

    input_data = normalize_form_data_for_prediction(
        form_data,
        cases_df,
        gpa_converter,
        background_university_set=page_state.background_universities,
    )

    session_manager.set(submitted=True, form_data_changed=False)
    return True, input_data, all_universities_target, all_majors_target, form_data


def _get_current_form_state(
    session_manager,
    background_university,
    background_major,
    final_target_universities,
    final_target_majors,
    language_type,
    raw_language_score_value,
    research_count,
    award_count,
    internship_count,
    paper_count,
    experience_details,
    cases_df,
    all_universities_target,
    all_majors_target,
    gpa_converter,
    exam_type=None,
    exam_score=None,
):
    from src.pages.prediction.page_data_loader import machine_learning_model

    page_state = machine_learning_model.resource_loader()

    school_service = get_school_level_service()
    is_overseas = (
        school_service.is_overseas_school(background_university) if background_university else False
    )

    _, current_normalized_score = calculate_processed_language_score(
        raw_language_score_value, language_type, background_university, is_overseas
    )

    current_normalized_gpa = calculate_processed_gpa(
        session_manager.get("gpa_raw_input"),
        session_manager.get("gpa_scale"),
        background_university,
        gpa_converter,
        exam_type,
        exam_score,
    )

    background_uni_for_model = get_background_university_for_model(
        background_university,
        cases_df,
        page_state.background_universities,
    )

    input_data = {
        "background_university": background_uni_for_model,
        "background_major": background_major,
        "target_universities": final_target_universities,
        "target_majors": final_target_majors,
        "gpa": current_normalized_gpa,
        "language_score": current_normalized_score,
        "language_type": language_type,
        "research_count": research_count,
        "award_count": award_count,
        "internship_count": internship_count,
        "paper_count": paper_count,
        "experience_details": experience_details,
    }

    return (
        False,
        input_data,
        all_universities_target,
        all_majors_target,
        None,
    )
