import streamlit as st

from src.utils.data_safety.watermark import generate_watermark_css
from src.utils.logger import setup_logger
from src.utils.page_auth import handle_e2_login
from src.utils.session_manager import SessionManager

page_init_logger = setup_logger("page3", "prediction")


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
    hide_sidebar: bool = False,
):
    st.set_page_config(
        page_title=page_title,
        layout=layout,
        initial_sidebar_state=initial_sidebar_state,
        page_icon="assets/favicon.ico",
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
        with open(css_file, encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    if not skip_auth:
        handle_e2_login(current_page_path, module_name=module_name, admin_only=admin_only)

    SessionManager()

    user_nickname = st.session_state.get("e2_user_nickname", default_nickname)
    user_email = st.session_state.get("e2_user_email", "E2_USER_NOT_LOGGED_IN")

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
