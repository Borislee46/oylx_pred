import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import streamlit as st

from src.utils.auth.page_auth import require_login
from src.utils.logger import ensure_lead_in_console_logging, setup_logger
from src.utils.navigation import build_pages
from src.utils.ws_auth import install_ws_auth_gate

install_ws_auth_gate()

main_logger = setup_logger("page3", "prediction")
ensure_lead_in_console_logging()


def main() -> None:
    current_page = st.navigation(build_pages(), position="hidden")

    username = require_login()
    if not username:
        st.error("登录会话无效或已过期，请重新登录。", icon=":material/lock:")
        st.markdown('<a href="/login" target="_self">前往登录页</a>', unsafe_allow_html=True)
        st.stop()

    current_page.run()


main()
