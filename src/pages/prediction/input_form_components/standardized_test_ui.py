import streamlit as st

from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_WIDGET_KEYS
from src.pages.prediction.input_form_components.form_config import STANDARDIZED_TEST_TYPES
from src.pages.prediction.input_form_components.form_validator import FormValidator


def render_standardized_test_section(session_manager, form_state_manager, logger):
    current_test_type = session_manager.get(DEFAULT_FORM_KEYS.standardized_test_type)
    if current_test_type is None or current_test_type not in STANDARDIZED_TEST_TYPES:
        current_test_type = "GRE"
        session_manager.set(standardized_test_type="GRE")

    if (
        DEFAULT_WIDGET_KEYS.standardized_test_type not in st.session_state
        or st.session_state.get(DEFAULT_WIDGET_KEYS.standardized_test_type)
        not in STANDARDIZED_TEST_TYPES
    ):
        st.session_state[DEFAULT_WIDGET_KEYS.standardized_test_type] = current_test_type

    def on_test_type_change():
        new_type = session_manager.get_widget_value(DEFAULT_WIDGET_KEYS.standardized_test_type)
        if new_type and new_type in STANDARDIZED_TEST_TYPES:
            session_manager.set(standardized_test_type=new_type)

    exam_type = st.segmented_control(
        "标化成绩 (选填)",
        options=STANDARDIZED_TEST_TYPES,
        selection_mode="single",
        key=DEFAULT_WIDGET_KEYS.standardized_test_type,
        on_change=on_test_type_change,
    )

    if not exam_type:
        exam_type = current_test_type

    score_key = f"standardized_test_score_text_{exam_type}"

    if score_key not in st.session_state:
        saved_score = session_manager.get(score_key)
        st.session_state[score_key] = saved_score if saved_score else ""

    def on_score_change():
        session_manager.set(**{score_key: st.session_state[score_key]})
        form_state_manager.on_form_change(session_manager, change_type="text")

    score_input = st.text_input(
        f"{exam_type} 总分",
        placeholder="无",
        key=score_key,
        on_change=on_score_change,
    )

    exam_score = None
    if score_input and score_input.strip():
        is_valid, error_msg, parsed_score = FormValidator.validate_standardized_test_score(
            exam_type, score_input.strip()
        )

        if is_valid:
            exam_score = parsed_score
        else:
            st.toast(error_msg)
            exam_score = None

    if exam_score != session_manager.get("current_exam_score"):
        session_manager.set(current_exam_score=exam_score)

    return exam_type, exam_score
