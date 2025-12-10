from functools import partial

import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_LANGUAGE_SCORES,
    LANGUAGE_SCORE_RANGES,
    LANGUAGE_TYPES,
    LANGUAGE_WARNING_THRESHOLDS,
)
from src.pages.prediction.input_form_components.language_score_validator import (
    LanguageScoreValidator,
)
from src.utils.school_level_service import get_school_level_service


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
        and not LanguageScoreValidator.validate_ielts_step(language_score_input)
    ):
        warning_key = f"ielts_step_warning_{language_score_input}"
        if session_manager.get("last_ielts_step_warning_key") != warning_key:
            st.toast("雅思成绩必须是0.5的倍数")
            session_manager.set(last_ielts_step_warning_key=warning_key)


def _render_overseas_language_input(
    session_manager,
    form_state_manager,
    language_type,
    score_config,
    widget_key,
    label_suffix,
    placeholder_text,
):
    if widget_key not in st.session_state:
        default_val = session_manager.get("language_score_input")
        if default_val is None or default_val == 0:
            st.session_state[widget_key] = ""
        else:
            st.session_state[widget_key] = str(default_val)
    else:
        current_val = st.session_state[widget_key]
        if not isinstance(current_val, str):
            st.session_state[widget_key] = (
                "" if current_val is None or current_val == 0 else str(current_val)
            )

    language_score_text = st.text_input(
        f"{language_type}成绩{label_suffix}",
        on_change=lambda: (form_state_manager.on_form_change(session_manager, change_type="text"),),
        placeholder=placeholder_text,
        key=widget_key,
    )

    final_language_score, error_msg, has_input_error = (
        LanguageScoreValidator.validate_and_parse_score(language_score_text, language_type)
    )

    if error_msg:
        st.toast(error_msg)

    session_manager.set(
        language_score_input=final_language_score, language_score_input_error=has_input_error
    )


def _render_domestic_language_input(
    session_manager, form_state_manager, language_type, score_config, widget_key, placeholder_text
):
    if widget_key not in st.session_state:
        default_lang_val = session_manager.get("language_score_input")
        if default_lang_val is None or default_lang_val == 0:
            default_lang_val = DEFAULT_LANGUAGE_SCORES.get(
                language_type, 6.5 if language_type == "雅思" else 86
            )
        st.session_state[widget_key] = default_lang_val
    else:
        current_val = st.session_state[widget_key]
        if isinstance(current_val, str):
            try:
                parsed_val = (
                    float(current_val)
                    if current_val.strip()
                    else DEFAULT_LANGUAGE_SCORES.get(
                        language_type, 6.5 if language_type == "雅思" else 86
                    )
                )
                st.session_state[widget_key] = parsed_val
            except ValueError:
                st.session_state[widget_key] = DEFAULT_LANGUAGE_SCORES.get(
                    language_type, 6.5 if language_type == "雅思" else 86
                )

    language_score = st.number_input(
        f"{language_type}成绩",
        min_value=score_config["min"],
        max_value=score_config["max"],
        step=score_config["step"],
        format=score_config["format"],
        on_change=lambda: (
            session_manager.set(
                language_score_input=session_manager.get_widget_value(
                    "language_score_input_widget"
                ),
                language_score_input_error=False,
            ),
            form_state_manager.on_form_change(session_manager, change_type="text"),
        ),
        placeholder=placeholder_text,
        key=widget_key,
    )
    session_manager.set(language_score_input=language_score, language_score_input_error=False)


def render_language_section(session_manager, form_state_manager, logger):
    st.markdown("**语言成绩**")

    background_university = session_manager.get("background_university")
    school_service = get_school_level_service()
    is_overseas = (
        school_service.is_overseas_school(background_university) if background_university else False
    )

    if background_university:
        school_level = school_service.get_school_level(background_university)
        logger.debug(
            f"背景院校: {background_university}, 学校等级: {school_level}, 是否海外: {is_overseas}"
        )

    language_type = session_manager.get("language_type")
    st.segmented_control(
        "语言成绩类型",
        LANGUAGE_TYPES,
        selection_mode="single",
        default=language_type,
        on_change=partial(form_state_manager.on_language_type_change, session_manager),
        key="language_type_widget_key",
    )

    score_config = LANGUAGE_SCORE_RANGES[language_type]
    widget_key = "language_score_input_widget"

    label_suffix = "（选填）" if is_overseas else ""
    placeholder_text = "海外院校背景语言成绩为选填" if is_overseas else "请输入成绩"

    if is_overseas:
        _render_overseas_language_input(
            session_manager,
            form_state_manager,
            language_type,
            score_config,
            widget_key,
            label_suffix,
            placeholder_text,
        )
    else:
        _render_domestic_language_input(
            session_manager,
            form_state_manager,
            language_type,
            score_config,
            widget_key,
            placeholder_text,
        )

    if (
        session_manager.get("language_score_input") is not None
        and session_manager.get("language_score_input") > 0
    ):
        _check_and_show_language_warning(session_manager)
        _check_and_show_ielts_step_warning(session_manager)

    return language_type, session_manager.get("language_score_input")
