from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.pages.prediction.page_data_loader import machine_learning_model
    from src.utils.session_manager import SessionManager


@dataclass
class SessionKeys:
    form_data_changed: str = "form_data_changed"
    input_data: str = "input_data"
    predict_lock: str = "prediction_submit_lock"
    has_predicted: str = "has_predicted"
    is_school_selection_submit: str = "is_school_selection_submit"
    last_submission_logged: str = "last_submission_logged"


@dataclass
class FormSubmissionContext:
    session_manager: "SessionManager"
    page_state: "machine_learning_model"
    input_data_from_form: dict
    all_universities_target: list[str]
    all_majors_target: list[str]
    original_form_data: dict | None
    session_keys: SessionKeys

    @classmethod
    def create(
        cls,
        session_manager: "SessionManager",
        page_state: "machine_learning_model",
        input_data_from_form: dict,
        all_universities_target: list[str],
        all_majors_target: list[str],
        original_form_data: dict | None = None,
        session_keys: SessionKeys | None = None,
    ) -> "FormSubmissionContext":
        return cls(
            session_manager=session_manager,
            page_state=page_state,
            input_data_from_form=input_data_from_form,
            all_universities_target=all_universities_target,
            all_majors_target=all_majors_target,
            original_form_data=original_form_data,
            session_keys=session_keys or SessionKeys(),
        )


DEFAULT_SESSION_KEYS = SessionKeys()
