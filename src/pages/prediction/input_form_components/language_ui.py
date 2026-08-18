from functools import partial
from typing import cast

import streamlit as st

from src.pages.prediction.core.ui_messages import (
    FORM_LABELS,
    FORM_PLACEHOLDERS,
    FORM_WARNING_MESSAGES,
)
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_WIDGET_KEYS
from src.pages.prediction.input_form_components.form_config import (
    DEFAULT_LANGUAGE_SCORES,
    LANGUAGE_SCORE_RANGES,
    LANGUAGE_TYPES,
    LANGUAGE_WARNING_THRESHOLDS,
)
from src.utils.numeric import is_close_to_int
from src.utils.schools.level_service import get_school_level_service


class LanguageScoreValidator:
    @staticmethod
    def validate_ielts_step(score: float) -> bool:
        return is_close_to_int(score * 2)

    @staticmethod
    def validate_score_range(score: float, language_type: str) -> tuple[bool, str | None]:
        if language_type not in LANGUAGE_SCORE_RANGES:
            return False, f"未知的语言类型: {language_type}"

        score_config = LANGUAGE_SCORE_RANGES[language_type]
        min_score = float(cast(int | float, score_config["min"]))
        max_score = float(cast(int | float, score_config["max"]))

        if score < min_score or score > max_score:
            return (
                False,
                f"{language_type}成绩必须在 {min_score} 到 {max_score} 之间",
            )

        if language_type == "雅思" and not LanguageScoreValidator.validate_ielts_step(score):
            return False, "雅思成绩必须是0.5的倍数"

        return True, None

    @staticmethod
    def validate_and_parse_score(
        score_text: str, language_type: str
    ) -> tuple[float | None, str | None, bool]:
        if not score_text or not score_text.strip():
            return None, None, False

        try:
            score_value = float(score_text.strip())
        except ValueError:
            return None, f"请输入有效的{language_type}成绩", True

        is_valid, error_msg = LanguageScoreValidator.validate_score_range(
            score_value, language_type
        )

        if not is_valid:
            return None, error_msg, True

        return score_value, None, False


def apply_overseas_language_boost(school_name: str, language_type: str) -> float:
    school_service = get_school_level_service()

    if not school_service.is_overseas_school(school_name):
        return DEFAULT_LANGUAGE_SCORES.get(language_type, 6.5)

    base_score = DEFAULT_LANGUAGE_SCORES.get(language_type)
    if base_score is None:
        return 6.5 if language_type == "雅思" else 90

    multiplier = school_service.get_language_boost_multiplier(school_name)
    boosted_score = base_score * multiplier

    max_value = cast(int | float, LANGUAGE_SCORE_RANGES[language_type]["max"])
    max_score = float(max_value)
    return min(boosted_score, max_score)


def _check_and_show_language_warning(session_manager):
    language_score_input = session_manager.get("language_score_input")
    language_type = session_manager.get("language_type")
    if (
        language_score_input is not None
        and language_score_input > 0
        and language_score_input < LANGUAGE_WARNING_THRESHOLDS[language_type]
    ):
        threshold = LANGUAGE_WARNING_THRESHOLDS[language_type]
        score_display = (
            f"{int(language_score_input)}"
            if language_type == "托福"
            else f"{language_score_input:.1f}"
        )
        threshold_display = f"{int(threshold)}" if language_type == "托福" else f"{threshold:.1f}"
        warning_key = f"lang_warning_{score_display}_{language_type}"
        if session_manager.get("last_lang_warning_key") != warning_key:
            st.toast(
                FORM_WARNING_MESSAGES["language_score_below_threshold"].format(
                    language_type=language_type, score=score_display, threshold=threshold_display
                )
            )
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
            st.toast(FORM_WARNING_MESSAGES["ielts_step_warning"])
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

    if final_language_score != session_manager.get(
        "language_score_input"
    ) or has_input_error != session_manager.get("language_score_input_error"):
        updates = {
            "language_score_input": final_language_score,
            "language_score_input_error": has_input_error,
        }
        if final_language_score is not None and final_language_score > 0:
            updates[DEFAULT_FORM_KEYS.language_score_user_provided] = True
        session_manager.set(**updates)


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
        FORM_LABELS["language_score_label"].format(language_type=language_type),
        min_value=score_config["min"],
        max_value=score_config["max"],
        step=score_config["step"],
        format=score_config["format"],
        on_change=lambda: (
            session_manager.set(
                language_score_input=session_manager.get_widget_value(
                    DEFAULT_WIDGET_KEYS.language_score
                ),
                language_score_input_error=False,
                **{DEFAULT_FORM_KEYS.language_score_user_provided: True},
            ),
            form_state_manager.on_form_change(session_manager, change_type="text"),
        ),
        placeholder=placeholder_text,
        key=widget_key,
    )
    user_provided = session_manager.get(DEFAULT_FORM_KEYS.language_score_user_provided, False)
    default_score = DEFAULT_LANGUAGE_SCORES.get(
        language_type, 6.5 if language_type == "雅思" else 86
    )
    is_placeholder_default = not user_provided and language_score == default_score

    if is_placeholder_default:
        if session_manager.get("language_score_input") is not None:
            session_manager.set(language_score_input=None, language_score_input_error=False)
    elif (
        language_score != session_manager.get("language_score_input")
        or session_manager.get("language_score_input_error") is not False
    ):
        session_manager.set(language_score_input=language_score, language_score_input_error=False)


def render_language_section(session_manager, form_state_manager, logger):
    st.markdown(FORM_LABELS["language_score"])

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
    if language_type not in LANGUAGE_TYPES:
        language_type = LANGUAGE_TYPES[0]
        if session_manager.get("language_type") != language_type:
            session_manager.set(language_type=language_type)

    if (
        DEFAULT_WIDGET_KEYS.language_type not in st.session_state
        or st.session_state.get(DEFAULT_WIDGET_KEYS.language_type) not in LANGUAGE_TYPES
    ):
        st.session_state[DEFAULT_WIDGET_KEYS.language_type] = language_type

    st.segmented_control(
        FORM_LABELS["language_type_label"],
        LANGUAGE_TYPES,
        selection_mode="single",
        on_change=partial(form_state_manager.on_language_type_change, session_manager),
        key=DEFAULT_WIDGET_KEYS.language_type,
    )

    score_config = LANGUAGE_SCORE_RANGES[language_type]
    widget_key = DEFAULT_WIDGET_KEYS.language_score

    label_suffix = FORM_LABELS["language_score_optional"] if is_overseas else ""
    placeholder_text = (
        FORM_PLACEHOLDERS["language_score_overseas"]
        if is_overseas
        else FORM_PLACEHOLDERS["language_score_domestic"]
    )

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

    user_provided = session_manager.get(DEFAULT_FORM_KEYS.language_score_user_provided, False)
    lang_input = session_manager.get("language_score_input")
    if not user_provided:
        return language_type, None
    return language_type, lang_input
