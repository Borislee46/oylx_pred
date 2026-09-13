import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import re
import time

import streamlit as st

from src.utils.auth.permission_checker import (
    check_user_access_permission,
    get_user_accessible_modules,
)
from src.utils.logger import setup_logger
from src.utils.page_init import init_page
from src.utils.ui.main_page_button import render_buttons_grid
from src.utils.ui.main_page_header import render_header

main_logger = setup_logger("page3", "prediction")


def _initialize_page_and_state():
    user_info = init_page(
        page_title="Signals 留学数据科学平台",
        current_page_path="app_pages/home.py",
        layout="wide",
        hide_sidebar=True,
        require_whitelist=False,
    )

    user_nickname = user_info["user_nickname"]

    if "logged_in_user" not in st.session_state:
        st.session_state.logged_in_user = None

    return user_info, user_nickname


def _enforce_access_and_get_modules(user_email: str):
    has_access = check_user_access_permission(user_email)
    if not has_access:
        st.error("抱歉，您没有访问权限。", icon=":material/error:")
        main_logger.info(f"用户 {user_email} 没有访问权限。")
        st.stop()

    accessible_modules = get_user_accessible_modules(user_email)
    is_user_admin = accessible_modules.get("admin", False)
    return accessible_modules, is_user_admin


def _collect_available_buttons(accessible_modules: dict, is_user_admin: bool, user_email: str):
    available_buttons = []

    if accessible_modules.get("hk", False):
        available_buttons.append(("Signals 留学择校系统", "app_pages/hk.py", False))

    return available_buttons


def _handle_scroll_to_query_param() -> None:
    if "scroll_to" not in st.query_params:
        return
    scroll_to = st.query_params.get("scroll_to")
    if scroll_to and re.match(r"^[A-Za-z0-9_\-]+$", scroll_to):
        st.iframe(
            f"""
            <script>
            (function () {{
                const fe = window.frameElement;
                if (fe) {{
                    fe.style.cssText =
                        "position:absolute!important;left:-9999px!important;width:0!important;height:0!important;border:0!important;margin:0!important;padding:0!important;opacity:0!important;pointer-events:none!important;";
                    const shell = fe.closest('[data-testid="stElementContainer"]');
                    if (shell) {{
                        shell.style.cssText =
                            "display:none!important;width:0!important;height:0!important;margin:0!important;padding:0!important;border:0!important;overflow:hidden!important;";
                    }}
                }}
            }})();
            setTimeout(() => {{
                const el = window.parent.document.getElementById("{scroll_to}");
                if (el) {{
                    el.scrollIntoView({{behavior: "smooth", block: "start"}});
                }}
            }}, 50);
           </script>
            """,
            width=1,
            height=1,
            tab_index=-1,
        )
    new_params = {k: v for k, v in st.query_params.items() if k != "scroll_to"}
    st.query_params.clear()
    if new_params:
        st.query_params.update(new_params)


def main() -> None:
    user_info, user_nickname = _initialize_page_and_state()
    user_email = user_info["user_email"]

    accessible_modules, is_user_admin = _enforce_access_and_get_modules(user_email)

    render_header(user_nickname)

    _handle_scroll_to_query_param()

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
        st.info("暂无可用模块。", icon=":material/info:")
        main_logger.info(f"用户 {user_email} 暂无可用模块。")


main()
