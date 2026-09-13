from .auth.auth_json_loader import load_auth_config
from .auth.permission_checker import (
    check_module_permission,
    check_user_access_permission,
    get_user_accessible_modules,
    is_admin,
)
from .data_safety.clipboard_guard import inject_clipboard_guard
from .env_config_loader import load_app_config
from .logger import setup_logger
from .page_auth import ensure_login
from .page_init import init_page
from .schools.constants import SCHOOL_LEVEL_PRIORITY
from .schools.level_service import get_school_level_service

SUPPORT_EMAIL = "support@example.com"


def __getattr__(name: str):
    if name == "load_school_major_details_df":
        from src.pages.prediction.app_data import load_school_major_details_df as _f

        return _f
    if name == "load_model_dependencies":
        from src.pages.prediction.model_loader import load_model_dependencies as _f

        return _f
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


from .session_manager import (  # noqa: E402
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
    "ensure_login",
    "get_school_level_service",
    "get_user_accessible_modules",
    "init_page",
    "inject_clipboard_guard",
    "is_admin",
    "load_app_config",
    "load_auth_config",
    "load_model_dependencies",
    "load_school_major_details_df",
    "setup_logger",
]
