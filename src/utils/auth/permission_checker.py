from src.utils.auth.config_processor import get_processed_auth_config
from src.utils.auth.dev_config_loader import load_dev_config

MODULE_IDS = (
    "hk",
    "admin",
    "cs_survey",
    "hr_dashboard",
    "hr_profile",
    "hr_structure_dashboard",
    "hk_dashboard",
)


def check_user_access_permission(user_email: str) -> bool:
    if not user_email:
        return False

    if load_dev_config().get("DEBUG_MODE", False):
        return True

    if is_admin(user_email):
        return True

    processed_config = get_processed_auth_config()
    if processed_config.get("AUTH_CONFIG_BROKEN", False):
        return False
    user_email_lower = user_email.lower()

    whitelist = processed_config["EMAIL_WHITELIST_LOWER"]
    if not whitelist:
        return False

    return user_email_lower in whitelist


def _check_standard_module(user_email_lower: str, module_name: str, processed_config: dict) -> bool:
    perms = processed_config.get("MODULE_PERMISSIONS_LOWER", {}).get(module_name)
    if not perms:
        return False
    return user_email_lower in perms


def check_module_permission(user_email: str, module_name: str) -> bool:
    if load_dev_config().get("DEBUG_MODE", False):
        return True

    if not check_user_access_permission(user_email):
        return False

    if module_name not in MODULE_IDS:
        return False

    if is_admin(user_email):
        return True

    processed_config = get_processed_auth_config()
    return _check_standard_module(user_email.lower(), module_name, processed_config)


def is_admin(user_email: str) -> bool:
    if not user_email:
        return False

    if load_dev_config().get("DEBUG_MODE", False):
        return True

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()

    return user_email_lower in processed_config["ADMIN_EMAILS_LOWER"]


def get_user_accessible_modules(user_email: str) -> dict:
    if load_dev_config().get("DEBUG_MODE", False):
        return dict.fromkeys(MODULE_IDS, True)

    empty = dict.fromkeys(MODULE_IDS, False)

    if not check_user_access_permission(user_email):
        return empty

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()
    module_perms = processed_config.get("MODULE_PERMISSIONS_LOWER", {})
    is_user_admin = user_email_lower in processed_config["ADMIN_EMAILS_LOWER"]

    result = {}
    for mid in MODULE_IDS:
        if mid == "admin":
            result[mid] = is_user_admin
        else:
            if is_user_admin:
                result[mid] = True
            else:
                perms = module_perms.get(mid)
                result[mid] = user_email_lower in perms if perms else False

    return result


def can_view_headcount_trend(user_email: str) -> bool:
    if not user_email:
        return False
    processed_config = get_processed_auth_config()
    allowed = processed_config.get("HR_STRUCTURE_HEADCOUNT_TREND_LOWER") or set()
    return str(user_email).strip().lower() in allowed
