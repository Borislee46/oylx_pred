from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import streamlit as st

from src.pages.prediction.input_form_components.background_ui import (
    render_background_section as render_background_section_component,
)
from src.pages.prediction.input_form_components.experience_ui import (
    render_experience_section as render_experience_section_component,
)
from src.pages.prediction.input_form_components.form_state import FormStateManager
from src.pages.prediction.input_form_components.gpa_ui import (
    render_gpa_section as render_gpa_section_component,
)
from src.pages.prediction.input_form_components.language_ui import (
    render_language_section as render_language_section_component,
)
from src.pages.prediction.input_form_components.standardized_test_ui import (
    render_standardized_test_section as render_standardized_test_section_component,
)
from src.pages.prediction.input_form_components.target_ui import (
    render_target_section as render_target_section_component,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

if TYPE_CHECKING:
    pass

form_ui_logger = setup_logger("page3", "prediction")


def _render_submit_button(session_manager, form_state_manager, disabled_status=False):
    lead_in_pending_review = (
        session_manager.get("lead_in_form_filled", False)
        and not session_manager.get("form_data_changed", False)
        and not session_manager.get("submitted", False)
    )
    is_currently_submitting = session_manager.get("submitted", False) and not session_manager.get(
        "form_data_changed", False
    )
    final_disabled = disabled_status or lead_in_pending_review or is_currently_submitting

    return st.button(
        "预测",
        on_click=partial(form_state_manager.on_submit_click, session_manager),
        disabled=final_disabled,
        key="submit_button_key",
        shortcut="enter",
    )


class FormUIComponents:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.form_state_manager = FormStateManager()

    def render_background_section(self, cases_df):
        (
            background_university,
            selected_background_major_original,
            background_major,
            background_major_2_original,
            background_major_2,
            is_dual_degree,
            dual_alpha,
        ) = render_background_section_component(
            self.session_manager, self.form_state_manager, cases_df, form_ui_logger
        )
        return (
            background_university,
            selected_background_major_original,
            background_major,
            background_major_2_original,
            background_major_2,
            is_dual_degree,
            dual_alpha,
        )

    def render_gpa_section(self):
        return render_gpa_section_component(
            self.session_manager, self.form_state_manager, form_ui_logger
        )

    def render_standardized_test_section(self):
        return render_standardized_test_section_component(
            self.session_manager, self.form_state_manager, form_ui_logger
        )

    def render_target_section(self, cases_df):
        return render_target_section_component(
            self.session_manager, self.form_state_manager, cases_df, form_ui_logger
        )

    def render_language_section(self):
        return render_language_section_component(
            self.session_manager, self.form_state_manager, form_ui_logger
        )

    def render_experience_section(self):
        return render_experience_section_component(
            self.session_manager, self.form_state_manager, form_ui_logger
        )

    def render_submit_button(self, disabled_status=False):
        return _render_submit_button(self.session_manager, self.form_state_manager, disabled_status)
