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
from src.pages.prediction.input_form_components.submit_ui import (
    render_submit_button as render_submit_button_component,
)
from src.pages.prediction.input_form_components.target_ui import (
    render_target_section as render_target_section_component,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

form_ui_logger = setup_logger("page3", "prediction")


class FormUIComponents:
    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager
        self.form_state_manager = FormStateManager()

    def render_background_section(self, cases_df):
        return render_background_section_component(
            self.session_manager, self.form_state_manager, cases_df, form_ui_logger
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
        return render_submit_button_component(
            self.session_manager, self.form_state_manager, disabled_status
        )
