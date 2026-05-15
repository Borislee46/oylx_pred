from .app_data_loader import load_school_major_details_df
from .auth.auth_json_loader import load_auth_config
from .auth.dev_config_loader import load_dev_config
from .auth.e2_handler import handle_e2_callback
from .auth.permission_checker import (
    check_module_permission,
    check_user_access_permission,
    get_user_accessible_modules,
    is_admin,
)
from .data_safety.clipboard_guard import inject_clipboard_guard
from .env_config_loader import load_app_config
from .interaction_events import log_interaction_event
from .logger import setup_logger
from .model_loader import load_model_dependencies
from .page_auth import handle_e2_login
from .page_init import init_page
from .school_constants import SCHOOL_LEVEL_PRIORITY

SUPPORT_EMAIL = "support@signals-app.com"
from .school_level_service import get_school_level_service
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
    "get_school_level_service",
    "get_user_accessible_modules",
    "handle_e2_callback",
    "handle_e2_login",
    "init_page",
    "inject_clipboard_guard",
    "is_admin",
    "load_app_config",
    "load_auth_config",
    "load_dev_config",
    "load_model_dependencies",
    "load_school_major_details_df",
    "log_interaction_event",
    "setup_logger",
]
