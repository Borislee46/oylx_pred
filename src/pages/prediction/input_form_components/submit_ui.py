from functools import partial

import streamlit as st


def render_submit_button(session_manager, form_state_manager, disabled_status=False):
    is_currently_submitting = session_manager.get("submitted", False) and not session_manager.get(
        "form_data_changed", False
    )
    final_disabled = disabled_status or is_currently_submitting

    return st.button(
        "预测",
        on_click=partial(form_state_manager.on_submit_click, session_manager),
        disabled=final_disabled,
        key="submit_button_key",
    )
