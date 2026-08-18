import os

import streamlit as st

from src.utils.data_safety.watermark import generate_watermark_css
from src.utils.logger import setup_logger
from src.utils.auth.page_auth import require_login
from src.utils.session_manager import SessionManager

page_init_logger = setup_logger("page3", "prediction")

_css_cache: dict[str, tuple[float, str]] = {}


def _read_css_cached(path: str) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return ""
    entry = _css_cache.get(path)
    if entry is not None and entry[0] == mtime:
        return entry[1]
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return ""
    _css_cache[path] = (mtime, content)
    return content


def _resolve_contract_tier(email: str | None) -> str:
    try:
        from src.utils.contract_config import get_tier_for_email

        return get_tier_for_email(email)
    except Exception:
        return "joint_hkmo_sg"


def _sync_auth_to_session(session_mgr: SessionManager) -> None:
    if session_mgr.get("user_info", {}).get("username"):
        return

    username = st.session_state.get("demo_username") or st.session_state.get("e2_user_email")
    nickname = st.session_state.get("demo_nickname") or st.session_state.get("e2_user_nickname")
    if username:
        contract_tier = _resolve_contract_tier(username)
        session_mgr.set(
            user_info={
                "username": username,
                "nickname": nickname or username,
                "contract_tier": contract_tier,
            },
            is_logged_in=True,
        )
        return

    if st.session_state.get("is_authenticated", False):
        legacy_username = st.session_state.get("username", "")
        if legacy_username:
            contract_tier = _resolve_contract_tier(legacy_username)
            session_mgr.set(
                user_info={
                    "username": legacy_username,
                    "nickname": legacy_username,
                    "contract_tier": contract_tier,
                },
                is_logged_in=True,
            )


def init_page(
    page_title: str,
    current_page_path: str,
    layout: str = "wide",
    initial_sidebar_state: str = "collapsed",
    default_nickname: str = "访客",
    additional_css_files: list[str] | None = None,
    watermark_config: dict | None = None,
    skip_auth: bool = False,
    skip_watermark: bool = False,
    module_name: str | None = None,
    admin_only: bool = False,
    require_whitelist: bool = True,
    hide_sidebar: bool = False,
):
    favicon = "assets/favicon.ico" if os.path.exists("assets/favicon.ico") else ""
    st.set_page_config(
        page_title=page_title,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        page_icon=favicon,
    )

    if hide_sidebar:
        st.markdown(
            """
            <style>
            [data-testid="stSidebarNav"] { display: none !important; }
            section[data-testid="stSidebar"] { display: none !important; }
            [data-testid="collapsedControl"] { display: none !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    css_files_to_load = ["assets/style.css"]
    if additional_css_files:
        css_files_to_load.extend(additional_css_files)

    for css_file in css_files_to_load:
        css_content = _read_css_cached(css_file)
        if css_content:
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    if not skip_auth:
        if require_login() is None:
            st.error("登录会话无效或已过期，请重新登录。", icon=":material/lock:")
            st.markdown('<a href="/login" target="_self">前往登录页</a>', unsafe_allow_html=True)
            st.stop()

    session_mgr = SessionManager()
    _sync_auth_to_session(session_mgr)

    user_nickname = st.session_state.get("demo_nickname", default_nickname)
    user_email = st.session_state.get("e2_user_email", "DEMO_USER_NOT_LOGGED_IN")

    if not skip_watermark:
        if watermark_config is None:
            watermark_config = {
                "opacity": 0.03,
                "color": "#333333",
                "font_size": "15px",
                "x_spacing": 250,
                "y_spacing": 100,
            }

        watermark_css = generate_watermark_css(user_nickname=user_nickname, **watermark_config)
        st.markdown(watermark_css, unsafe_allow_html=True)

    return {"user_nickname": user_nickname, "user_email": user_email}
