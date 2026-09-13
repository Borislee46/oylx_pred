import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

import streamlit as st

from src.utils.logger import ensure_lead_in_console_logging, setup_logger
from src.utils.navigation import build_pages
from src.utils.page_auth import require_login

main_logger = setup_logger("page3", "prediction")
ensure_lead_in_console_logging()


def main() -> None:
    # HTTP 层已由 AuthEnforcementMiddleware 拦截未登录请求；
    # 这里再做一次会话校验，覆盖 WebSocket / 直接访问脚本的路径。
    if require_login() is None:
        st.error("未登录或登录已过期，请重新输入访问码。", icon=":material/lock:")
        st.stop()

    st.navigation(build_pages(), position="hidden").run()


main()
