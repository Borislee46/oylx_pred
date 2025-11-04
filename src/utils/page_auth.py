import time
import urllib.parse
import uuid

import streamlit as st

from src.utils.auth.dev_config_loader import load_dev_config
from src.utils.auth.permission_checker import (
    check_module_permission,
    check_user_access_permission,
    is_admin,
)
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

page_auth_logger = setup_logger("page3", "prediction")


def handle_e2_login(
    current_page_path: str, module_name: str | None = None, admin_only: bool = False
) -> None:
    dev_config = load_dev_config()
    if dev_config.get("DEBUG_MODE", False):
        debug_user = dev_config.get("DEBUG_USER", {})
        st.session_state.e2_user_email = debug_user.get("email", "developer@example.com")
        st.session_state.e2_user_nickname = debug_user.get("nickname", "开发者")
        st.session_state.is_authenticated = True
        return

    TTL_SECONDS = 24 * 3600
    is_authenticated = st.session_state.get("is_authenticated", False)
    login_time = st.session_state.get("login_time")
    is_expired = False
    if is_authenticated and login_time:
        is_expired = (time.time() - float(login_time)) > TTL_SECONDS

    if (not is_authenticated) or is_expired:
        with st.spinner("请稍候，正在跳转到E2登录页面..."):
            APP_CONFIG = load_app_config()
            E2_X3ID_CONFIG = APP_CONFIG.get("E2_X3ID_CONFIG")
            E2_LOGIN_URL_TEMPLATE = APP_CONFIG.get("E2_LOGIN_URL_TEMPLATE")
            E2_CALLBACK_BASE_URL = APP_CONFIG.get("E2_CALLBACK_BASE_URL")
            CALLBACK_PAGE_PATH = APP_CONFIG.get("CALLBACK_PAGE_PATH")

            app_generated_state = str(uuid.uuid4())
            st.session_state["e2_state"] = app_generated_state

            st.session_state.intended_page_after_e2_login = current_page_path

            base_return_url = urllib.parse.urljoin(E2_CALLBACK_BASE_URL, CALLBACK_PAGE_PATH)

            params_for_return_url = {
                "our_app_state_check": app_generated_state,
                "next": current_page_path,
            }
            query_string_for_return_url = urllib.parse.urlencode(params_for_return_url)
            raw_return_url_for_e2_param = f"{base_return_url}?{query_string_for_return_url}"

            encoded_final_return_url = urllib.parse.quote(raw_return_url_for_e2_param, safe="")

            e2_login_page_url = f"{E2_LOGIN_URL_TEMPLATE}?x3id={E2_X3ID_CONFIG}&state={app_generated_state}&returnUrl={encoded_final_return_url}"

            st.markdown(
                f"""<meta http-equiv="refresh" content="0; url={e2_login_page_url}">""",
                unsafe_allow_html=True,
            )
            st.stop()

    if is_authenticated and not is_expired:
        st.session_state["login_time"] = time.time()

    user_email = st.session_state.get("e2_user_email", "")

    if current_page_path == "main.py":
        return

    if not check_user_access_permission(user_email):
        st.error("抱歉，您没有访问此系统的权限。")
        page_auth_logger.info(f"用户 {user_email} 没有访问此系统的权限。")
        st.page_link("main.py", label="返回主页")
        st.stop()

    if admin_only:
        if not is_admin(user_email):
            st.error("抱歉，您没有管理员权限。此功能仅限管理员访问。")
            page_auth_logger.info(f"用户 {user_email} 没有管理员权限。")
            st.page_link("main.py", label="返回主页")
            st.stop()
        return

    if module_name and not check_module_permission(user_email, module_name):
        st.error("抱歉，您没有访问此功能模块的权限。")
        page_auth_logger.info(f"用户 {user_email} 所在组 {module_name} 没有访问此功能模块的权限。")
        st.page_link("main.py", label="返回主页")
        st.stop()
