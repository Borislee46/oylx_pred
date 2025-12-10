from functools import partial

import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    GPA_SCALES,
    GPA_WARNING_THRESHOLDS,
)


def _check_and_show_gpa_warning(session_manager):
    gpa_raw_input = session_manager.get("gpa_raw_input")
    gpa_scale = session_manager.get("gpa_scale")
    if (
        gpa_raw_input is not None
        and gpa_raw_input > 0
        and gpa_raw_input < GPA_WARNING_THRESHOLDS[gpa_scale]
    ):
        warning_key = f"gpa_warning_{gpa_raw_input:.2f}_{gpa_scale}"
        if session_manager.get("last_gpa_warning_key") != warning_key:
            st.toast(f"注意！当前GPA {gpa_raw_input:.2f} 远低于入学标准")
            session_manager.set(last_gpa_warning_key=warning_key)


def render_gpa_section(session_manager, form_state_manager, logger):
    st.segmented_control(
        "GPA 分制",
        options=list(GPA_SCALES.keys()),
        selection_mode="single",
        default=session_manager.get("gpa_scale"),
        on_change=partial(form_state_manager.gpa_scale_changed, session_manager),
        key="gpa_scale_widget_key",
    )

    current_gpa_scale_details = GPA_SCALES[session_manager.get("gpa_scale")]
    if "gpa_raw_input_widget" not in st.session_state:
        default_gpa_value = session_manager.get("gpa_raw_input")
        st.session_state["gpa_raw_input_widget"] = (
            default_gpa_value if default_gpa_value is not None else 3.0
        )
    gpa_raw = st.number_input(
        f"GPA (满分 {session_manager.get('gpa_scale')})",
        min_value=0.0,
        max_value=float(current_gpa_scale_details["max"]),
        step=float(current_gpa_scale_details["step"]),
        format=current_gpa_scale_details["format"],
        on_change=lambda: (
            session_manager.set(
                gpa_raw_input=session_manager.get_widget_value("gpa_raw_input_widget")
            ),
            form_state_manager.on_form_change(session_manager, change_type="text"),
        ),
        placeholder="",
        key="gpa_raw_input_widget",
    )
    session_manager.set(gpa_raw_input=gpa_raw)

    if session_manager.get("gpa_raw_input") is not None:
        _check_and_show_gpa_warning(session_manager)

    return gpa_raw
