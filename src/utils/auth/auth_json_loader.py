import copy
import json
import os

import streamlit as st

from src.utils.config_local_paths import resolve_auth_config_path
from src.utils.logger import setup_logger

_auth_logger = setup_logger("page3", "prediction")

_DEFAULT_AUTH_CONFIG = {
    "EMAIL_WHITELIST": [],
    "ADMIN_EMAILS": [],
    "MODULE_PERMISSIONS": {},
}


def _default_auth_config(broken: bool = False, reason: str = "") -> dict:
    config = copy.deepcopy(_DEFAULT_AUTH_CONFIG)
    config["__AUTH_CONFIG_BROKEN__"] = broken
    if reason:
        config["__AUTH_CONFIG_ERROR__"] = reason
    return config


@st.cache_data(show_spinner=False)
def _load_auth_config_cached(resolved_path: str):
    if not os.path.exists(resolved_path):
        reason = "auth_config 不存在"
        _auth_logger.warning("%s，拒绝访问", reason)
        return _default_auth_config(broken=True, reason=reason)

    with open(resolved_path, encoding="utf-8") as f:
        raw = f.read()

    try:
        config = json.loads(raw)
    except json.JSONDecodeError as e:
        reason = f"auth_config JSON 解析失败: {e}"
        _auth_logger.warning("%s，拒绝访问", reason)
        return _default_auth_config(broken=True, reason=reason)

    if not isinstance(config, dict):
        reason = "auth_config 顶层必须是对象"
        _auth_logger.warning("%s，拒绝访问", reason)
        return _default_auth_config(broken=True, reason=reason)

    config.setdefault("EMAIL_WHITELIST", [])
    config.setdefault("ADMIN_EMAILS", [])
    config.setdefault("MODULE_PERMISSIONS", {})
    config["__AUTH_CONFIG_BROKEN__"] = False
    return config


def load_auth_config(path: str | None = None):
    resolved = path if path is not None else resolve_auth_config_path()
    return _load_auth_config_cached(resolved)


def clear_load_auth_config_cache() -> None:
    _load_auth_config_cached.clear()
