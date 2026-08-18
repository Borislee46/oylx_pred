from .data_safety.clipboard_guard import inject_clipboard_guard
from .env_config_loader import load_app_config
from .logger import setup_logger
from .auth.auth_json_loader import load_auth_config
from .auth.page_auth import (
    check_module_permission,
    check_user_access_permission,
    get_current_nickname,
    get_current_username,
    get_user_accessible_modules,
    is_admin,
    require_login,
)
from .page_init import init_page
from .schools.constants import SCHOOL_LEVEL_PRIORITY
from .schools.level_service import get_school_level_service


def __getattr__(name: str):
    if name == "load_school_major_details_df":
        from src.pages.prediction.app_data import load_school_major_details_df as _f

        return _f
    if name == "load_model_dependencies":
        from src.pages.prediction.model_loader import load_model_dependencies as _f

        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


SUPPORT_EMAIL = "support@demo.local"
from .session_manager import (
    PredictionResultModel,
    SessionManager,
    UserDataModel,
)

__all__ = [
    "PredictionResultModel",
    "SCHOOL_LEVEL_PRIORITY",
    "SUPPORT_EMAIL",
    "SessionManager",
    "UserDataModel",
    "check_module_permission",
    "check_user_access_permission",
    "get_current_nickname",
    "get_current_username",
    "get_school_level_service",
    "get_user_accessible_modules",
    "init_page",
    "inject_clipboard_guard",
    "is_admin",
    "load_app_config",
    "load_auth_config",
    "load_model_dependencies",
    "load_school_major_details_df",
    "require_login",
    "setup_logger",
]
