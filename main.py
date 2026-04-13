import re
import time

import streamlit as st
import streamlit.components.v1 as components

from src.utils.auth.auth_json_loader import load_auth_config
from src.utils.auth.permission_checker import (
    check_user_access_permission,
    get_user_accessible_modules,
)
from src.utils.logger import setup_logger
from src.utils.page_init import init_page
from src.utils.ui.main_page_button import render_buttons_grid
from src.utils.ui.main_page_header import render_header

main_logger = setup_logger("page3", "prediction")


def _handle_oauth_callback_if_present() -> None:
    query_params = st.query_params
    callback_params = ["code", "state", "e2e", "our_app_state_check"]
    if all(key in query_params for key in callback_params):
        if st.session_state.get("is_authenticated", False):
            new_params = {k: v for k, v in query_params.items() if k not in callback_params}
            st.query_params.clear()
            if new_params:
                st.query_params.update(new_params)
            return

        from src.utils.auth.e2_handler import handle_e2_callback

        handle_e2_callback()
        st.stop()


def _initialize_page_and_state():
    user_info = init_page(
        page_title="前途欧亚留学数据科学平台",
        current_page_path="main.py",
        layout="wide",
        hide_sidebar=True,
    )

    user_nickname = user_info["user_nickname"]

    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None

    return user_info, user_nickname


def _enforce_access_and_get_modules(user_email: str):
    has_access = check_user_access_permission(user_email)
    if not has_access:
        st.error("抱歉，您没有访问权限。")
        main_logger.info(f"用户 {user_email} 没有访问权限。")
        st.stop()

    accessible_modules = get_user_accessible_modules(user_email)
    is_user_admin = accessible_modules.get("admin", False)

    auth_config = load_auth_config()
    if auth_config.get("MAINTENANCE_MODE", False) and not is_user_admin:
        st.title("系统维护")
        st.warning("系统目前正在进行维护，除管理员外，所有用户均无法访问。请稍后重试。")
        main_logger.warning(f"用户 {user_email} 由于维护模式被拒绝访问。")
        st.stop()

    return accessible_modules, is_user_admin


def _collect_available_buttons(accessible_modules: dict, is_user_admin: bool, user_email: str):
    available_buttons = []

    if accessible_modules.get("hk", False):
        available_buttons.append(("EasyApply 留学择校系统", "pages/hk.py", False))
        available_buttons.append(
            (
                "Power BI 完整版案例库",
                "https://qtpbi.staff.xdf.cn/powerbi/index.html#/home",
                True,
            )
        )

    if is_user_admin:
        available_buttons.append(("权限管理", "pages/admin.py", False))
        available_buttons.append(("algorithm_lab", "pages/algorithm_lab.py", False))

    if any(
        accessible_modules.get(k, False)
        for k in ("hr_dashboard", "hr_profile", "hr_structure_dashboard")
    ):
        available_buttons.append(("人力数据中心", "pages/hr_hub.py", False))

    return available_buttons


def main() -> None:
    user_info, user_nickname = _initialize_page_and_state()
    user_email = user_info["user_email"]

    accessible_modules, is_user_admin = _enforce_access_and_get_modules(user_email)

    render_header(user_nickname)

    if "scroll_to" in st.query_params:
        scroll_to = st.query_params.get("scroll_to")
        if scroll_to and re.match(r"^[A-Za-z0-9_\-]+$", scroll_to):
            components.html(
                f"""
                <script>
                setTimeout(() => {{
                    const el = window.parent.document.getElementById("{scroll_to}");
                    if (el) {{
                        el.scrollIntoView({{behavior: "smooth", block: "start"}});
                    }}
                }}, 50);
               </script>
                """,
                height=0,
                width=0,
            )
        new_params = {k: v for k, v in st.query_params.items() if k != "scroll_to"}
        st.query_params.clear()
        if new_params:
            st.query_params.update(new_params)

    available_buttons = _collect_available_buttons(accessible_modules, is_user_admin, user_email)
    button_names = [name for name, _, _ in available_buttons]

    if len(available_buttons) > 0:
        render_buttons_grid(available_buttons)
        last_log = st.session_state.get("modules_access_last_log", (None, 0))
        now = time.time()
        if last_log[0] != user_email or (now - last_log[1]) > 60:
            main_logger.info(f"用户 {user_email} 具有以下模块的访问权限: {button_names}")
            st.session_state.modules_access_last_log = (user_email, now)
    else:
        st.info("暂无可用模块，请联系管理员开通权限。")
        main_logger.info(f"用户 {user_email} 暂无可用模块。")


_handle_oauth_callback_if_present()
main()
