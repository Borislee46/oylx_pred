from functools import partial

import pandas as pd
import streamlit as st

from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper
from src.utils.app_data_loader import load_school_base_data
from src.utils.school_level_service import get_school_level_service

school_level_service = get_school_level_service()


def _log_background_university_change(session_manager, form_state_manager, logger):
    selected_university = session_manager.get_widget_value("background_university_selectbox")
    session_manager.set(background_university=selected_university)

    from src.pages.prediction.input_form_components.form_state import FormStateManager

    FormStateManager._clear_widget_state("language_score_input_widget")
    session_manager.set(language_score_input=None)

    logger.info(f"用户选择背景院校: {selected_university}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _log_background_major_change(session_manager, form_state_manager, logger):
    selected_major = session_manager.get_widget_value("background_major_selectbox")
    logger.info(f"用户选择背景专业: {selected_major}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _generate_university_options(session_manager, cases_df):
    universities_from_cases = []
    if cases_df is not None and "background_university" in cases_df.columns:
        universities_from_cases = cases_df["background_university"].astype(str).unique().tolist()

    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    school_base_df = session_manager.get("school_base_df")
    universities_from_school_base = []
    if school_base_df is not None and "学校名称" in school_base_df.columns:
        universities_from_school_base = school_base_df["学校名称"].astype(str).unique().tolist()

    university_counts = {}
    if cases_df is not None and "background_university" in cases_df.columns:
        university_counts = cases_df["background_university"].value_counts().to_dict()

    all_unique_universities = list(set(universities_from_cases + universities_from_school_base))

    return sorted(all_unique_universities, key=lambda x: university_counts.get(x, 0), reverse=True)


def _generate_major_options(cases_df):
    major_mapping_df = pd.DataFrame(columns=["background_major_original", "background_major"])
    if cases_df is not None and all(
        col in cases_df.columns for col in ["background_major_original", "background_major"]
    ):
        major_mapping_df = cases_df[
            ["background_major_original", "background_major"]
        ].drop_duplicates()
    major_original_to_actual_map = dict(
        zip(
            major_mapping_df["background_major_original"],
            major_mapping_df["background_major"],
            strict=False,
        )
    )
    major_counts = {}
    if cases_df is not None and "background_major" in cases_df.columns:
        major_counts = cases_df["background_major"].value_counts().to_dict()

    majors_background_display_unique = [
        major
        for major in major_mapping_df["background_major_original"].astype(str).unique()
        if major.lower() not in ["none", "nan"]
    ]

    majors_background_display = sorted(
        majors_background_display_unique,
        key=lambda orig_major: major_counts.get(
            major_original_to_actual_map.get(orig_major, orig_major), 0
        ),
        reverse=True,
    )

    return {
        "majors_display": majors_background_display,
        "major_map": major_original_to_actual_map,
    }


def render_background_section(session_manager, form_state_manager, cases_df, logger):
    st.markdown("**背景信息**")

    helper = SelectBoxHelper(session_manager, form_state_manager, logger)

    background_university = helper.render_cached_selectbox(
        label="背景院校",
        widget_key="background_university_selectbox",
        cache_key="background_universities_cache",
        history_key="background_university",
        options_generator_func=lambda: _generate_university_options(session_manager, cases_df),
        on_change_callback=partial(
            _log_background_university_change,
            session_manager,
            form_state_manager,
            logger,
        ),
    )

    selected_background_major_original = helper.render_cached_selectbox(
        label="背景专业",
        widget_key="background_major_selectbox",
        cache_key="background_majors_cache",
        history_key="background_major_original",
        options_generator_func=lambda: _generate_major_options(cases_df),
        on_change_callback=partial(
            _log_background_major_change, session_manager, form_state_manager, logger
        ),
        options_path_in_cache="majors_display",
    )

    background_majors_cache = session_manager.get("background_majors_cache", {})
    major_original_to_actual_map = background_majors_cache.get("major_map", {})

    background_major = (
        major_original_to_actual_map.get(selected_background_major_original)
        if selected_background_major_original
        else None
    )

    session_manager.set(background_university=background_university)

    return background_university, selected_background_major_original, background_major
