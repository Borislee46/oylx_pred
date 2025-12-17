import streamlit as st

from src.pages.prediction.input_form_components.form_config import STANDARDIZED_TEST_TYPES
from src.pages.prediction.input_form_components.form_validator import FormValidator


def render_standardized_test_section(session_manager, form_state_manager, logger):
    current_test_type = session_manager.get("standardized_test_type")
    if current_test_type is None or current_test_type not in STANDARDIZED_TEST_TYPES:
        current_test_type = "GRE"
        session_manager.set(standardized_test_type="GRE")

    if (
        "standardized_test_type_widget" not in st.session_state
        or st.session_state.get("standardized_test_type_widget") not in STANDARDIZED_TEST_TYPES
    ):
        st.session_state["standardized_test_type_widget"] = current_test_type

    def on_test_type_change():
        prev_type = session_manager.get("standardized_test_type") or "GRE"
        if prev_type not in STANDARDIZED_TEST_TYPES:
            prev_type = "GRE"

        new_type = session_manager.get_widget_value("standardized_test_type_widget")
        if new_type not in STANDARDIZED_TEST_TYPES:
            session_manager.set(standardized_test_type=prev_type)
            st.session_state["standardized_test_type_widget"] = prev_type
            return

        session_manager.set(standardized_test_type=new_type)
        form_state_manager.on_form_change(session_manager, change_type="select")

    exam_type = st.segmented_control(
        "标化成绩 (选填)",
        options=STANDARDIZED_TEST_TYPES,
        selection_mode="single",
        key="standardized_test_type_widget",
        on_change=on_test_type_change,
    )

    score_key = f"standardized_test_score_text_{exam_type}"

    if score_key not in st.session_state:
        saved_score = session_manager.get(score_key)
        st.session_state[score_key] = saved_score if saved_score else ""

    def on_score_change():
        session_manager.set(**{score_key: st.session_state[score_key]})
        form_state_manager.on_form_change(session_manager, change_type="text")

    score_input = st.text_input(
        f"{exam_type} 总分",
        value=st.session_state[score_key],
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

    session_manager.set(current_exam_score=exam_score)

    return exam_type, exam_score
