import logging

import streamlit as st

from src.pages.prediction.flow.hk_orchestrator import dispatch_prediction
from src.pages.prediction.handler_config import (
    DEFAULT_FORM_KEYS,
    DEFAULT_INTERNAL_KEYS,
    DEFAULT_SESSION_KEYS,
)
from src.pages.prediction.page_data_loader import machine_learning_model
from src.pages.prediction.results_handler import initialize_session_states
from src.pages.prediction.ui import (
    get_product_logo_image_as_base64,
    render_header,
)
from src.pages.prediction.ui.fragments import form_fragment, results_fragment
from src.pages.prediction.ui.page_render import (
    render_background_summary_bar,
    render_lead_in_section,
    render_page_footer,
    render_timeline_rail,
)
from src.pages.prediction.ui.page_state_machine import PageStateMachine
from src.pages.prediction.ui.scroll_utils import scroll_to_anchor
from src.utils.analytics import track as _track
from src.utils.data_safety.clipboard_guard import inject_clipboard_guard
from src.utils.logger import log_once, setup_logger
from src.utils.page_init import init_page
from src.utils.session_manager import SessionManager

_page_logger = setup_logger("hk", "prediction")

user_info = init_page(
    page_title="Signals 留学择校系统",
    current_page_path="app_pages/hk.py",
    layout="wide",
    default_nickname="E2访客",
    module_name="hk",
    hide_sidebar=True,
    additional_css_files=[
        "assets/hk_style/00_tokens.css",
        "assets/hk_style/10_glass.css",
        "assets/hk_style/20_header.css",
        "assets/hk_style/30_controls.css",
        "assets/hk_style/40_components.css",
        "assets/hk_style/50_ux.css",
        "assets/hk_style/52_timeline.css",
        "assets/hk_style/54_workbench.css",
        "assets/hk_style/54a_pathfinder_cards.css",
        "assets/hk_style/54b_cta_buttons.css",
        "assets/hk_style/54c_school_cards.css",
        "assets/hk_style/56_view_modes.css",
        "assets/hk_style/58_toast_feedback.css",
    ],
)

session_manager = SessionManager()
inject_clipboard_guard()

log_once(
    _page_logger,
    "hk_page_load",
    logging.INFO,
    "页面加载 | user=%s email=%s",
    user_info.get("user_nickname", "?"),
    user_info.get("user_email", "?"),
)


def prediction_page_content() -> None:
    logo_base64 = get_product_logo_image_as_base64("assets/product_logo.png")
    page_state = machine_learning_model.resource_loader()

    _track("page_view", is_new_session=True)
    _ref = st.query_params.get("ref")
    if _ref:
        _track("marketing_referral", source=_ref)
    initialize_session_states(session_manager)

    if not session_manager.get(DEFAULT_FORM_KEYS.user_nickname):
        session_manager.set(user_nickname=user_info.get("user_nickname", "E2访客"))

    render_timeline_rail(session_manager)
    render_header(logo_base64)

    page_sm = PageStateMachine(session_manager)
    is_running = page_sm.is_running()

    has_predicted_before = session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False)
    editing = session_manager.get(DEFAULT_INTERNAL_KEYS.edit_background, False)
    show_input = ((not has_predicted_before) or editing) and not is_running

    with st.container(key="hk_stage_lead_in"):
        st.html('<div id="hk-form-anchor"></div>')
        if show_input:
            render_lead_in_section(session_manager)

    if show_input:
        form_fragment(session_manager, page_state)

    if (
        page_sm.is_idle()
        and not has_predicted_before
        and not session_manager.get(DEFAULT_INTERNAL_KEYS.form_anchor_scrolled, False)
    ):
        session_manager.set(**{DEFAULT_INTERNAL_KEYS.form_anchor_scrolled: True})
        scroll_to_anchor("hk-form-anchor", delay_ms=200)

    summary_placeholder = st.empty()

    progress_area = st.container()
    dispatch_prediction(session_manager, page_state, progress_area)

    has_predicted_after = session_manager.get(DEFAULT_SESSION_KEYS.has_predicted, False)
    has_form_data = bool(session_manager.get(DEFAULT_SESSION_KEYS.input_data))
    editing_after = session_manager.get(DEFAULT_INTERNAL_KEYS.edit_background, False)
    if (has_predicted_before or has_predicted_after or has_form_data) and not editing_after:
        with summary_placeholder.container():
            render_background_summary_bar(session_manager)

    results_fragment(session_manager, page_state)
    render_page_footer(session_manager)


if __name__ == "__main__" or st.runtime.exists():
    prediction_page_content()
