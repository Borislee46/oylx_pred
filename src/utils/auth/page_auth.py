from __future__ import annotations

import time

import streamlit as st

from src.utils.auth.session import SESSION_COOKIE_NAME, validate_session_token

SESSION_TTL_SECONDS = 24 * 3600
DEFAULT_USERNAME = "demo"
DEFAULT_NICKNAME = "演示用户"


def _apply_session(payload: dict) -> None:
    username = str(payload.get("username") or DEFAULT_USERNAME)
    nickname = str(payload.get("nickname") or DEFAULT_NICKNAME)
    st.session_state.demo_username = username
    st.session_state.demo_nickname = nickname
    st.session_state.demo_user_email = f"{username}@demo.local"
    st.session_state.demo_user_nickname = nickname
    st.session_state.is_authenticated = True
    st.session_state["login_time"] = time.time()


def _restore_from_cookie() -> bool:
    try:
        raw = st.context.cookies.get(SESSION_COOKIE_NAME)
    except Exception:
        raw = None
    if not raw:
        return False
    payload = validate_session_token(raw)
    if not payload:
        return False
    _apply_session(payload)
    return True


def _is_expired() -> bool:
    login_time = st.session_state.get("login_time")
    if not login_time:
        return False
    try:
        return (time.time() - float(login_time)) > SESSION_TTL_SECONDS
    except (TypeError, ValueError):
        return True


def ensure_login() -> str:
    """保证当前 Streamlit 会话已通过访问码门禁。

    已登录则刷新活跃时间并返回用户名；否则尝试用签名 Cookie 恢复；
    两者都不成立时渲染提示并终止当前页面。
    """
    if st.session_state.get("is_authenticated") and not _is_expired():
        st.session_state["login_time"] = time.time()
        return str(st.session_state.get("demo_username") or DEFAULT_USERNAME)

    if _restore_from_cookie():
        return str(st.session_state.get("demo_username") or DEFAULT_USERNAME)

    st.error("未登录或登录已过期，请重新输入访问码。", icon=":material/lock:")
    st.stop()
    return DEFAULT_USERNAME


def require_login() -> str | None:
    if st.session_state.get("is_authenticated") and st.session_state.get("demo_username"):
        return str(st.session_state.demo_username)
    if _restore_from_cookie():
        return str(st.session_state.demo_username)
    return None


def get_current_username() -> str:
    return str(st.session_state.get("demo_username", ""))


def get_current_nickname() -> str:
    return str(st.session_state.get("demo_nickname", DEFAULT_NICKNAME))


def check_user_access_permission(_email: str | None = None) -> bool:
    return True


def check_module_permission(_email: str | None = None, _module: str | None = None) -> bool:
    return True


def get_user_accessible_modules(_email: str = "") -> dict[str, bool]:
    return {"hk": True}


def is_admin(_email: str | None = None) -> bool:
    return False
