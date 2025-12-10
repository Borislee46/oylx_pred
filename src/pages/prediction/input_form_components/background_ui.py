from functools import partial

import streamlit as st

from src.pages.prediction.input_form_components.form_state import FormStateManager
from src.pages.prediction.input_form_components.widget_helpers import SelectBoxHelper
from src.utils.app_data_loader import load_school_base_data


def _log_background_university_change(session_manager, form_state_manager, logger):
    selected_university = session_manager.get_widget_value("background_university_selectbox")
    session_manager.set(background_university=selected_university, language_score_input=None)
    FormStateManager._clear_widget_state("language_score_input_widget")
    logger.info(f"用户选择背景院校: {selected_university}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _log_background_major_change(session_manager, form_state_manager, logger):
    selected_major = session_manager.get_widget_value("background_major_selectbox")
    logger.info(f"用户选择背景专业: {selected_major}")
    form_state_manager.on_form_change(session_manager, change_type="select")


def _generate_university_options(session_manager, cases_df):
    universities: set[str] = set()
    university_counts: dict[str, int] = {}

    if cases_df is not None and "background_university" in cases_df.columns:
        col = cases_df["background_university"]
        universities.update(col.astype(str).unique())
        university_counts = col.value_counts().to_dict()

    if session_manager.get("school_base_df") is None:
        session_manager.set(school_base_df=load_school_base_data())

    school_base_df = session_manager.get("school_base_df")
    if school_base_df is not None and "学校名称" in school_base_df.columns:
        universities.update(school_base_df["学校名称"].astype(str).unique())

    return sorted(universities, key=lambda x: university_counts.get(x, 0), reverse=True)


def _generate_major_options(cases_df):
    required_cols = ["background_major_original", "background_major"]
    if cases_df is None or not all(col in cases_df.columns for col in required_cols):
        return {"majors_display": [], "major_map": {}}

    mapping_df = cases_df[required_cols].drop_duplicates()
    major_map = dict(zip(mapping_df["background_major_original"], mapping_df["background_major"]))
    major_counts = cases_df["background_major"].value_counts().to_dict()

    majors_display = [
        m
        for m in mapping_df["background_major_original"].astype(str).unique()
        if m.lower() not in ["none", "nan"]
    ]
    majors_display.sort(key=lambda m: major_counts.get(major_map.get(m, m), 0), reverse=True)

    return {"majors_display": majors_display, "major_map": major_map}


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

    major_map = session_manager.get("background_majors_cache", {}).get("major_map", {})
    background_major = (
        major_map.get(selected_background_major_original)
        if selected_background_major_original
        else None
    )

    return background_university, selected_background_major_original, background_major
