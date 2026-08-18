from __future__ import annotations

import time

import streamlit as st

from src.utils.auth.session import SESSION_COOKIE_NAME, validate_session_token


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


def _apply_session(payload: dict) -> None:
    username = str(payload.get("username", ""))
    nickname = str(payload.get("nickname", username))
    st.session_state.demo_username = username
    st.session_state.demo_nickname = nickname
    st.session_state.e2_user_email = f"{username}@demo.local"
    st.session_state.e2_user_nickname = nickname
    st.session_state.is_authenticated = True
    st.session_state["login_time"] = time.time()


def require_login() -> str | None:
    if st.session_state.get("is_authenticated") and st.session_state.get("demo_username"):
        return str(st.session_state.demo_username)
    if _restore_from_cookie():
        return str(st.session_state.demo_username)
    return None


def get_current_username() -> str:
    return str(st.session_state.get("demo_username", ""))


def get_current_nickname() -> str:
    return str(st.session_state.get("demo_nickname", "演示用户"))


def check_user_access_permission(_email: str | None = None) -> bool:
    return True


def check_module_permission(_email: str | None = None, _module: str | None = None) -> bool:
    return True


def get_user_accessible_modules(_email: str = "") -> dict[str, bool]:
    return {"hk": True}


def is_admin(_email: str | None = None) -> bool:
    return False

