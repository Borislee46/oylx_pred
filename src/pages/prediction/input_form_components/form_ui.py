import streamlit as st

from src.pages.prediction.input_form_components.background_ui import (
    render_background_section as render_background_section_component,
)
from src.pages.prediction.input_form_components.experience_ui import (
    render_experience_section as render_experience_section_component,
)
from src.pages.prediction.input_form_components.form_config import (
    GPA_WARNING_THRESHOLDS,
    LANGUAGE_WARNING_THRESHOLDS,
)
from src.pages.prediction.input_form_components.form_state import FormStateManager
from src.pages.prediction.input_form_components.gpa_ui import (
    render_gpa_section as render_gpa_section_component,
)
from src.pages.prediction.input_form_components.language_ui import (
    render_language_section as render_language_section_component,
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

    def _log_background_university_change(self):
        selected_university = self.session_manager.get_widget_value(
            "background_university_selectbox"
        )
        form_ui_logger.info(f"用户选择背景院校: {selected_university}")
        self.form_state_manager.on_form_change(self.session_manager, change_type="select")

    def _log_background_major_change(self):
        selected_major = self.session_manager.get_widget_value("background_major_selectbox")
        form_ui_logger.info(f"用户选择背景专业: {selected_major}")
        self.form_state_manager.on_form_change(self.session_manager, change_type="select")

    def _log_gpa_input_change(self):
        gpa_value = self.session_manager.get_widget_value("gpa_raw_input_widget")
        self.session_manager.set(gpa_raw_input=gpa_value)
        self.form_state_manager.on_form_change(self.session_manager, change_type="text")

    def _log_language_score_change(self):
        score_value = self.session_manager.get_widget_value("language_score_input_widget")
        self.session_manager.set(language_score_input=score_value)
        self.form_state_manager.on_form_change(self.session_manager, change_type="text")

    def _check_and_show_gpa_warning(self):
        gpa_raw_input = self.session_manager.get("gpa_raw_input")
        gpa_scale = self.session_manager.get("gpa_scale")
        if (
            gpa_raw_input is not None
            and gpa_raw_input > 0
            and gpa_raw_input < GPA_WARNING_THRESHOLDS[gpa_scale]
        ):
            warning_key = f"gpa_warning_{gpa_raw_input:.2f}_{gpa_scale}"
            if self.session_manager.get("last_gpa_warning_key") != warning_key:
                st.toast(f"注意！当前GPA {gpa_raw_input:.2f} 远低于入学标准")
                self.session_manager.set(last_gpa_warning_key=warning_key)

    def _check_and_show_language_warning(self):
        language_score_input = self.session_manager.get("language_score_input")
        language_type = self.session_manager.get("language_type")
        if (
            language_score_input is not None
            and language_score_input > 0
            and language_score_input < LANGUAGE_WARNING_THRESHOLDS[language_type]
        ):
            warning_key = f"lang_warning_{language_score_input:.1f}_{language_type}"
            if self.session_manager.get("last_lang_warning_key") != warning_key:
                st.toast(f"注意！当前{language_type}成绩 {language_score_input:.1f} 远低于入学标准")
                self.session_manager.set(last_lang_warning_key=warning_key)

    def _check_and_show_ielts_step_warning(self):
        language_type = self.session_manager.get("language_type")
        language_score_input = self.session_manager.get("language_score_input")
        if (
            language_type == "雅思"
            and language_score_input is not None
            and (abs(language_score_input * 2 - round(language_score_input * 2)) > 1e-9)
        ):
            warning_key = f"ielts_step_warning_{language_score_input}"
            if self.session_manager.get("last_ielts_step_warning_key") != warning_key:
                st.toast("雅思成绩必须是0.5的倍数")
                self.session_manager.set(last_ielts_step_warning_key=warning_key)

    def _log_experience_change(self, experience_type):
        value = self.session_manager.get_widget_value(f"{experience_type}_count_input", 0)
        form_ui_logger.info(f"用户输入{experience_type}数量: {value}")
        self.form_state_manager.on_form_change(self.session_manager, change_type="text")

    def _log_experience_details_change(self, experience_type):
        details = self.session_manager.get_widget_value(f"{experience_type}_details_input", "")
        if details.strip():
            form_ui_logger.info(
                f"用户输入{experience_type}详细信息: {details[:100]}{'...' if len(details) > 100 else ''}"
            )
        else:
            form_ui_logger.info(f"用户清空{experience_type}详细信息")
        self.form_state_manager.on_form_change(self.session_manager, change_type="text")

    def render_background_section(self, cases_df):
        return render_background_section_component(
            self.session_manager, self.form_state_manager, cases_df, form_ui_logger
        )

    def render_gpa_section(self):
        return render_gpa_section_component(
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
