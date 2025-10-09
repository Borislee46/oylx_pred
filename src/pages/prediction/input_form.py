import time

import streamlit as st

from src.pages.prediction.input_form_components import (
    FormStateManager,
    FormUIComponents,
    FormValidator,
    GPAConverter,
)
from src.pages.prediction.results_handler import reset_prediction_results
from src.pages.prediction.user_background_analyzer import find_substitute_university
from src.utils.app_data_loader import load_school_base_data
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

form_logger = setup_logger("page3", "prediction")


def create_input_form(session_manager: SessionManager, cases_df, disabled_status=False):
    FormStateManager.initialize_session_state(session_manager)

    user_history_data = session_manager.get("user_history_data", {})
    current_user_id = session_manager.get("current_user_id")

    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    gpa_converter = GPAConverter(session_manager.get("school_base_df"))

    ui_components = FormUIComponents(session_manager)

    with st.container(border=True):
        col1, col2 = st.columns([1, 1], gap="small")

        with col1:
            background_university, selected_background_major_original, background_major = (
                ui_components.render_background_section(cases_df)
            )
            gpa_raw = ui_components.render_gpa_section()
            (
                final_target_universities,
                final_target_majors,
                all_universities_target,
                all_majors_target,
            ) = ui_components.render_target_section(cases_df)

        with col2:
            language_type, raw_language_score_value = ui_components.render_language_section()
            research_count, award_count, internship_count, paper_count, experience_details = (
                ui_components.render_experience_section()
            )

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
            "language_type": language_type,
            "language_score_raw": raw_language_score_value,
            "research_count": research_count,
            "award_count": award_count,
            "internship_count": internship_count,
            "paper_count": paper_count,
            "experience_details": experience_details,
        }

        error_messages = FormValidator.validate_form_data(form_data, gpa_converter)

        if error_messages:
            form_logger.warning(f"表单验证失败 - 错误信息: {error_messages}")
            for msg in error_messages:
                st.toast(msg)
                time.sleep(0.5)
            session_manager.set(
                submitted=False, form_data_changed=True, prediction_submit_lock=False
            )
            reset_prediction_results(session_manager)
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

            return success, processed_input_data, all_unis, all_majors, original_form_data

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
    )


def _get_background_university_for_model(selected_background_university, cases_df):
    if not selected_background_university:
        return None

    unique_background_universities = cases_df["background_university"].unique()
    if selected_background_university not in unique_background_universities:
        return find_substitute_university(selected_background_university, cases_df)

    return selected_background_university


def _process_successful_submission(
    session_manager, form_data, cases_df, all_universities_target, all_majors_target, gpa_converter
):
    normalized_gpa = FormValidator.normalize_gpa(
        form_data["gpa_raw"],
        form_data["gpa_scale"],
        form_data.get("background_university"),
        gpa_converter,
    )

    language_score_for_submission = form_data["language_score_raw"]

    final_normalized_lang_score = None
    if language_score_for_submission is not None:
        final_normalized_lang_score = FormValidator.normalize_language_score(
            language_score_for_submission, form_data["language_type"]
        )

    background_uni_for_model = _get_background_university_for_model(
        form_data["background_university"], cases_df
    )

    input_data = {
        "background_university": background_uni_for_model,
        "background_major": form_data["background_major"],
        "target_universities": form_data["target_universities"],
        "target_majors": form_data["target_majors"],
        "gpa": normalized_gpa,
        "language_score": final_normalized_lang_score,
        "language_type": form_data["language_type"],
        "research_count": form_data["research_count"],
        "award_count": form_data["award_count"],
        "internship_count": form_data["internship_count"],
        "paper_count": form_data["paper_count"],
        "experience_details": form_data["experience_details"],
    }

    if form_data["experience_details"]:
        for exp_type, details in form_data["experience_details"].items():
            if details and details.strip():
                exp_type_name = {
                    "research_details": "科研项目",
                    "award_details": "获奖情况",
                    "internship_details": "实习经历",
                    "paper_details": "论文发表",
                }.get(exp_type, exp_type)
                form_logger.info(f"提交表单 - {exp_type_name}详细信息: {details}")

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
):
    current_display_lang_score = raw_language_score_value

    current_normalized_score = None
    if current_display_lang_score is not None:
        current_normalized_score = FormValidator.normalize_language_score(
            current_display_lang_score, language_type
        )

    current_normalized_gpa = None
    current_raw_gpa_val = session_manager.get("gpa_raw_input")
    if current_raw_gpa_val is not None:
        current_normalized_gpa = FormValidator.normalize_gpa(
            current_raw_gpa_val,
            session_manager.get("gpa_scale"),
            background_university,
            gpa_converter,
        )

    background_uni_for_model = _get_background_university_for_model(background_university, cases_df)

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
        session_manager.get("submitted", False),
        input_data,
        all_universities_target,
        all_majors_target,
        None,
    )
