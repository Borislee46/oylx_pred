import os

_AUTH = "config/auth_config.json"
_AUTH_EX = "config/auth_config.example.json"
_APP = "config/app_config.json"
_APP_EX = "config/app_config.example.json"
_HR = "config/hr_permissions.json"
_HR_EX = "config/hr_permissions.example.json"
_DEV = "config/dev_config.json"
_DEV_EX = "config/dev_config.example.json"


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def resolve_auth_config_path() -> str:
    p = os.environ.get("OYLX_AUTH_CONFIG_PATH")
    if p:
        return p
    if os.path.exists(_AUTH):
        return _AUTH
    return _AUTH_EX if os.path.exists(_AUTH_EX) else _AUTH


def resolve_app_config_path() -> str:
    p = os.environ.get("OYLX_APP_CONFIG_PATH")
    if p:
        return p
    if os.path.exists(_APP):
        return _APP
    return _APP_EX if os.path.exists(_APP_EX) else _APP


def resolve_dev_config_path() -> str:
    env_override = os.environ.get("OYLX_DEV_CONFIG_PATH")
    if env_override:
        return env_override
    if os.path.exists(_DEV):
        return _DEV
    return _DEV_EX if os.path.exists(_DEV_EX) else _DEV


def resolve_hr_permissions_path() -> str:
    p = os.environ.get("OYLX_HR_PERMISSIONS_PATH")
    if p:
        return os.path.abspath(p)

    relative_path = _HR
    candidates = [os.path.abspath(relative_path)]

    project_root = _project_root()
    candidates.append(os.path.join(project_root, "config", "hr_permissions.json"))
    for alt in (
        os.path.join(os.path.dirname(project_root), "config", "hr_permissions.json"),
        os.path.join(project_root, "..", "config", "hr_permissions.json"),
    ):
        candidates.append(os.path.abspath(alt))

    example_cwd = os.path.abspath(_HR_EX)
    example_proj = os.path.join(project_root, "config", "hr_permissions.example.json")

    for path in candidates:
        if os.path.exists(path):
            return path
    if os.path.exists(example_cwd):
        return example_cwd
    if os.path.exists(example_proj):
        return example_proj
    return candidates[0]


AUTH_CONFIG_WRITE_PATH = _AUTH
APP_CONFIG_WRITE_PATH = _APP
HR_PERMISSIONS_WRITE_PATH = _HR
