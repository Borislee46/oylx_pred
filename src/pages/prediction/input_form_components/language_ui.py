from functools import partial
import time

import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    LANGUAGE_SCORE_RANGES,
    LANGUAGE_TYPES,
    LANGUAGE_WARNING_THRESHOLDS,
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
        and (abs(language_score_input * 2 - round(language_score_input * 2)) > 1e-9)
    ):
        warning_key = f"ielts_step_warning_{language_score_input}"
        if session_manager.get("last_ielts_step_warning_key") != warning_key:
            st.toast("雅思成绩必须是0.5的倍数")
            session_manager.set(last_ielts_step_warning_key=warning_key)


def render_language_section(session_manager, form_state_manager, logger):
    st.markdown("**语言成绩**")
    
    background_university = session_manager.get("background_university")
    school_service = get_school_level_service()
    is_overseas = school_service.is_overseas_school(background_university) if background_university else False
    
    if background_university:
        school_level = school_service.get_school_level(background_university)
        logger.debug(f"背景院校: {background_university}, 学校等级: {school_level}, 是否海外: {is_overseas}")
    
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
    
    label_suffix = "（选填）" if is_overseas else ""
    placeholder_text = "海外院校背景语言成绩为选填" if is_overseas else "请输入成绩"
    
    if is_overseas:
        if widget_key not in st.session_state:
            default_val = session_manager.get("language_score_input")
            if default_val is None or default_val == 0:
                st.session_state[widget_key] = ""
            else:
                st.session_state[widget_key] = str(default_val)
        else:
            current_val = st.session_state[widget_key]
            if not isinstance(current_val, str):
                st.session_state[widget_key] = "" if current_val is None or current_val == 0 else str(current_val)
        
        language_score_text = st.text_input(
            f"{language_type}成绩{label_suffix}",
            on_change=lambda: (
                form_state_manager.on_form_change(session_manager, change_type="text"),
            ),
            placeholder=placeholder_text,
            key=widget_key,
        )
        
        final_language_score = None
        error_msg = None
        has_input_error = False
        
        if language_score_text.strip():
            try:
                score_value = float(language_score_text.strip())
                if score_value < score_config["min"] or score_value > score_config["max"]:
                    error_msg = f"{language_type}成绩必须在 {score_config['min']} 到 {score_config['max']} 之间"
                    has_input_error = True
                elif language_type == "雅思" and abs(score_value * 2 - round(score_value * 2)) > 1e-9:
                    error_msg = "雅思成绩必须是0.5的倍数"
                    has_input_error = True
                else:
                    final_language_score = score_value
            except ValueError:
                error_msg = f"请输入有效的{language_type}成绩"
                has_input_error = True
        
        if error_msg:
            st.toast(error_msg)
        
        session_manager.set(
            language_score_input=final_language_score,
            language_score_input_error=has_input_error
        )
    else:
        if widget_key not in st.session_state:
            default_lang_val = session_manager.get("language_score_input")
            if default_lang_val is None or default_lang_val == 0:
                default_lang_val = 6.5 if language_type == "雅思" else 86
            st.session_state[widget_key] = default_lang_val
        else:
            current_val = st.session_state[widget_key]
            if isinstance(current_val, str):
                try:
                    st.session_state[widget_key] = float(current_val) if current_val.strip() else (6.5 if language_type == "雅思" else 86)
                except ValueError:
                    st.session_state[widget_key] = 6.5 if language_type == "雅思" else 86
        
        language_score = st.number_input(
            f"{language_type}成绩",
            min_value=score_config["min"],
            max_value=score_config["max"],
            step=score_config["step"],
            format=score_config["format"],
            on_change=lambda: (
                session_manager.set(
                    language_score_input=session_manager.get_widget_value("language_score_input_widget"),
                    language_score_input_error=False
                ),
                form_state_manager.on_form_change(session_manager, change_type="text"),
            ),
            placeholder=placeholder_text,
            key=widget_key,
        )
        session_manager.set(language_score_input=language_score, language_score_input_error=False)

    if session_manager.get("language_score_input") is not None and session_manager.get("language_score_input") > 0:
        _check_and_show_language_warning(session_manager)
        _check_and_show_ielts_step_warning(session_manager)

    return language_type, session_manager.get("language_score_input")
