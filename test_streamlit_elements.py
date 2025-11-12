import uuid

import streamlit as st

from src.utils.announcements import (
    filter_announcements_for_user,
    generate_announcement_css,
    generate_announcement_html,
)
from src.utils.auth.auth_json_loader import load_auth_config
from src.utils.auth.permission_checker import (
    check_user_access_permission,
    get_user_accessible_modules,
)
from src.utils.logger import setup_logger
from src.utils.page_init import init_page
from test_components.nivo_charts_test import render_draggable_nivo_charts

main_logger = setup_logger("page3", "prediction")


def _handle_oauth_callback_if_present() -> None:
    query_params = st.query_params
    if all(key in query_params for key in ["code", "state", "e2e", "our_app_state_check"]):
        from src.utils.auth.e2_handler import handle_e2_callback

        handle_e2_callback()
        st.stop()


def _initialize_page_and_state():
    user_info = init_page(
        page_title="前途欧亚留学数据科学平台",
        current_page_path="main.py",
        layout="centered",
        hide_sidebar=True,
    )

    user_nickname = user_info["user_nickname"]

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())[:8]

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


def _render_announcements(user_email: str, is_user_admin: bool, accessible_modules: dict) -> None:
    user_announcements = filter_announcements_for_user(
        user_email, is_user_admin, accessible_modules
    )
    if user_announcements:
        announcement_css = generate_announcement_css()
        announcement_html = generate_announcement_html(user_announcements)
        st.markdown(announcement_css, unsafe_allow_html=True)
        st.markdown(announcement_html, unsafe_allow_html=True)


def _collect_available_buttons(accessible_modules: dict, is_user_admin: bool, user_email: str):
    available_buttons = []

    if accessible_modules.get("hk", False):
        available_buttons.append(("EasyApply 选校预测系统", "pages/hk.py", "🎓"))
        available_buttons.append(("案例库极速版", "pages/case_lib.py", "⚡"))

    if is_user_admin:
        available_buttons.append(("权限管理", "pages/admin.py", "⚙️"))
        available_buttons.append(("公告管理", "pages/announcement_admin.py", "⚙️"))

    if accessible_modules.get("hr_dashboard", False):
        available_buttons.append(("人力薪资数据看板", "pages/hr_dashboard.py", "👥"))

    if accessible_modules.get("hr_structure_dashboard", False):
        available_buttons.append(("人力结构数据看板", "pages/hr_structure_dashboard.py", "👥"))

    return available_buttons


def _render_buttons_grid(available_buttons) -> None:
    st.markdown(
        """
    <style>
    .stButton > button {
        height: 60px;
        font-size: 18px;
        width: 100%;
    }
    </style>
    <div style='margin: 40px 0;'></div>
    """,
        unsafe_allow_html=True,
    )

    if len(available_buttons) == 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            button_text, page_path, icon = available_buttons[0]
            if st.button(f"{icon} {button_text}"):
                st.switch_page(page_path)
    elif len(available_buttons) == 2:
        col1, col2 = st.columns(2)
        with col1:
            button_text, page_path, icon = available_buttons[0]
            if st.button(f"{icon} {button_text}"):
                st.switch_page(page_path)
        with col2:
            button_text, page_path, icon = available_buttons[1]
            if st.button(f"{icon} {button_text}"):
                st.switch_page(page_path)
    else:
        cols = st.columns(min(len(available_buttons), 3))
        for i, (button_text, page_path, icon) in enumerate(available_buttons):
            with cols[i % 3]:
                if st.button(f"{icon} {button_text}"):
                    st.switch_page(page_path)


def main() -> None:
    user_info, user_nickname = _initialize_page_and_state()
    user_email = user_info["user_email"]

    accessible_modules, is_user_admin = _enforce_access_and_get_modules(user_email)

    st.title("Streamlit Elements test")

    tab4 = st.tabs(["Monaco 编辑器", "Nivo 图表"])

    with tab4:
        render_draggable_nivo_charts()

    _render_announcements(user_email, is_user_admin, accessible_modules)

    available_buttons = _collect_available_buttons(accessible_modules, is_user_admin, user_email)
    button_names = [name for name, _, _ in available_buttons]

    if len(available_buttons) > 0:
        _render_buttons_grid(available_buttons)
        main_logger.info(f"用户 {user_email} 具有以下模块的访问权限: {button_names}")
    else:
        st.info("暂无可用模块，请联系管理员开通权限。")
        main_logger.info(f"用户 {user_email} 暂无可用模块。")


_handle_oauth_callback_if_present()
main()
