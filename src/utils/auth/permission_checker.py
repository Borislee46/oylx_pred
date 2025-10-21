from src.utils.auth.config_processor import get_processed_auth_config
from src.utils.auth.dev_config_loader import load_dev_config


def check_user_access_permission(user_email: str) -> bool:
    dev_config = load_dev_config()
    if dev_config.get("DEBUG_MODE", False):
        debug_user_email = dev_config.get("DEBUG_USER", {}).get("email")
        if debug_user_email and user_email.lower() == debug_user_email.lower():
            return True

    if not user_email:
        return False

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()

    whitelist = processed_config["EMAIL_WHITELIST_LOWER"]
    if whitelist:
        return user_email_lower in whitelist

    return True


def check_module_permission(user_email: str, module_name: str) -> bool:
    dev_config = load_dev_config()
    if dev_config.get("DEBUG_MODE", False):
        debug_user_email = dev_config.get("DEBUG_USER", {}).get("email")
        if debug_user_email and user_email.lower() == debug_user_email.lower():
            return True

    if not check_user_access_permission(user_email):
        return False

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()
    module_permissions = processed_config.get("MODULE_PERMISSIONS_LOWER", {})

    authorized_emails = module_permissions.get(module_name)

    if not authorized_emails:
        return True

    return user_email_lower in authorized_emails


def is_admin(user_email: str) -> bool:
    if not user_email:
        return False

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()

    dev_config = load_dev_config()
    if dev_config.get("DEBUG_MODE", False):
        debug_user_email = dev_config.get("DEBUG_USER", {}).get("email")
        if debug_user_email and user_email_lower == debug_user_email.lower():
            return True

    return user_email_lower in processed_config["ADMIN_EMAILS_LOWER"]


def get_user_accessible_modules(user_email: str) -> dict:
    dev_config = load_dev_config()
    if dev_config.get("DEBUG_MODE", False):
        debug_user_email = dev_config.get("DEBUG_USER", {}).get("email")
        if debug_user_email and user_email.lower() == debug_user_email.lower():
            return {
                "hk": True,
                "admin": True,
                "hk_admin": True,
                "log_admin": True,
                "hr_dashboard": True,
                "hr_structure_dashboard": True,
            }

    empty_modules = {
        "hk": False,
        "admin": False,
        "hk_admin": False,
        "log_admin": False,
        "hr_dashboard": False,
        "hr_structure_dashboard": False,
    }

    if not check_user_access_permission(user_email):
        return empty_modules

    processed_config = get_processed_auth_config()
    user_email_lower = user_email.lower()
    module_perms = processed_config["MODULE_PERMISSIONS_LOWER"]

    def _check(module_name):
        perms = module_perms.get(module_name)
        if not perms:
            return True
        return user_email_lower in perms

    is_user_admin = user_email_lower in processed_config["ADMIN_EMAILS_LOWER"]

    return {
        "hk": _check("hk"),
        "admin": is_user_admin,
        "hk_admin": is_user_admin,
        "log_admin": is_user_admin,
        "hr_dashboard": _check("hr_dashboard"),
        "hr_structure_dashboard": _check("hr_structure_dashboard"),
    }
