from functools import partial

import streamlit as st


def render_submit_button(session_manager, form_state_manager, disabled_status=False):
    help_text_submit = None
    processing_lock = session_manager.get("processing_lock", False)
    
    if processing_lock:
        help_text_submit = "优化正在进行中，请等待完成后再进行预测。"
    elif disabled_status:
        help_text_submit = "输入内容未改变，无需重复预测。如需重新预测，请更改表单输入。"

    is_currently_submitting = session_manager.get("submitted", False) and not session_manager.get(
        "form_data_changed", False
    )
    final_disabled = disabled_status or is_currently_submitting or processing_lock

    return st.button(
        "预测",
        on_click=partial(form_state_manager.on_submit_click, session_manager),
        disabled=final_disabled,
        key="submit_button_key",
        help=help_text_submit,
    )
