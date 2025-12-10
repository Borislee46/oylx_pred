import streamlit as st

from src.utils.auth.auth_json_loader import load_auth_config
from src.utils.auth.permission_checker import (
    check_user_access_permission,
    get_user_accessible_modules,
)
from src.utils.logger import setup_logger
from src.utils.page_init import init_page
from src.utils.session_manager import SessionManager
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

    SessionManager()

    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None

    if "last_seen_version" not in st.session_state:
        st.session_state.last_seen_version = None

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
        available_buttons.append(("EasyApply 选校预测系统", "pages/hk.py", False))
        available_buttons.append(
            (
                "Power BI 完整版案例库",
                "https://qtpbi.staff.xdf.cn/powerbi/index.html#/home",
                True,
            )
        )
        available_buttons.append(("测试页面", "pages/test.py", False))

    if is_user_admin:
        available_buttons.append(("权限管理", "pages/admin.py", False))

    if accessible_modules.get("hr_dashboard", False):
        available_buttons.append(("人力薪资数据看板", "pages/hr_dashboard.py", False))

    if accessible_modules.get("hr_structure_dashboard", False):
        available_buttons.append(("人力结构数据看板", "pages/hr_structure_dashboard.py", False))

    return available_buttons


def main() -> None:
    user_info, user_nickname = _initialize_page_and_state()
    user_email = user_info["user_email"]

    accessible_modules, is_user_admin = _enforce_access_and_get_modules(user_email)

    render_header(user_nickname)

    available_buttons = _collect_available_buttons(accessible_modules, is_user_admin, user_email)
    button_names = [name for name, _, _ in available_buttons]

    if len(available_buttons) > 0:
        render_buttons_grid(available_buttons)
        if not st.session_state.get("modules_access_logged", False):
            main_logger.info(f"用户 {user_email} 具有以下模块的访问权限: {button_names}")
            st.session_state.modules_access_logged = True
    else:
        st.info("暂无可用模块，请联系管理员开通权限。")
        main_logger.info(f"用户 {user_email} 暂无可用模块。")


_handle_oauth_callback_if_present()
main()
