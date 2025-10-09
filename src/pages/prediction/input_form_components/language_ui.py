from functools import partial

import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    LANGUAGE_SCORE_RANGES,
    LANGUAGE_TYPES,
    LANGUAGE_WARNING_THRESHOLDS,
)


def _check_and_show_language_warning(session_manager):
    language_score_input = session_manager.get("language_score_input")
    language_type = session_manager.get("language_type")
    if (
        language_score_input is not None
        and language_score_input > 0
        and language_score_input < LANGUAGE_WARNING_THRESHOLDS[language_type]
    ):
        warning_key = f"lang_warning_{language_score_input:.1f}_{language_type}"
        if session_manager.get("last_lang_warning_key") != warning_key:
            st.toast(f"注意！当前{language_type}成绩 {language_score_input:.1f} 远低于入学标准")
            session_manager.set(last_lang_warning_key=warning_key)


def _check_and_show_ielts_step_warning(session_manager):
    language_type = session_manager.get("language_type")
    language_score_input = session_manager.get("language_score_input")
    if (
        language_type == "雅思"
        and language_score_input is not None
        and (abs(language_score_input * 2 - round(language_score_input * 2)) > 1e-9)
    ):
        warning_key = f"ielts_step_warning_{language_score_input}"
        if session_manager.get("last_ielts_step_warning_key") != warning_key:
            st.toast("雅思成绩必须是0.5的倍数")
            session_manager.set(last_ielts_step_warning_key=warning_key)


def render_language_section(session_manager, form_state_manager, logger):
    st.markdown("**语言成绩**")
    language_type = session_manager.get("language_type")
    st.radio(
        "语言成绩类型",
        LANGUAGE_TYPES,
        index=LANGUAGE_TYPES.index(language_type),
        horizontal=True,
        on_change=partial(form_state_manager.on_language_type_change, session_manager),
        key="language_type_widget_key",
    )

    score_config = LANGUAGE_SCORE_RANGES[language_type]
    widget_key = "language_score_input_widget"
    if widget_key not in st.session_state:
        default_lang_val = session_manager.get("language_score_input")
        if default_lang_val is None:
            default_lang_val = 6.5 if language_type == "雅思" else 86
        st.session_state[widget_key] = default_lang_val

    language_score = st.number_input(
        f"{language_type}成绩",
        min_value=score_config["min"],
        max_value=score_config["max"],
        step=score_config["step"],
        format=score_config["format"],
        on_change=lambda: (
            session_manager.set(
                language_score_input=session_manager.get_widget_value("language_score_input_widget")
            ),
            form_state_manager.on_form_change(session_manager, change_type="text"),
        ),
        placeholder="",
        key=widget_key,
    )
    session_manager.set(language_score_input=language_score)

    if session_manager.get("language_score_input") is not None:
        _check_and_show_language_warning(session_manager)
        _check_and_show_ielts_step_warning(session_manager)

    return language_type, session_manager.get("language_score_input")
