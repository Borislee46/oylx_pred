import base64
import streamlit as st
from pathlib import Path
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

@st.cache_data(show_spinner=False)
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = Path.cwd() / path
    try:
        return base64.b64encode(full_path.read_bytes()).decode()
    except Exception as e:
        logger.error(f"读取 logo 失败 {full_path}: {e}")
        return ""

def render_header(logo_base64: str) -> None:
    if not logo_base64:
        st.title("EasyApply 选校预测系统")
        return

    html_block = f"""
        <div class="hk-header">
            <div class="hk-logo-container">
                <img class="hk-header-logo" src="data:image/png;base64,{logo_base64}" alt="logo">
                <span class="hk-logo-shine"></span>
            </div>
            <div>
                <p class="hk-header-title">EasyApply</p>
                <p class="hk-header-subtitle">留学择校系统</p>
            </div>
        </div>
    """
    try:
        st.html(html_block)
    except (AttributeError, TypeError):
        st.markdown(html_block, unsafe_allow_html=True)

def display_feedback_section(session_id: str) -> None:
    key = f"feedback_{session_id}"
    toast_key = f"{key}_toast_sent"
    
    current = st.feedback("thumbs", key=key)
    if current is None:
        return

    val = int(current) if isinstance(current, (bool, int, str)) else None
    if val is None: return

    if st.session_state.get(toast_key) != val:
        msg = "感谢您的肯定！我们会继续努力！" if val == 1 else "收到您的反馈，我们会持续改进！"
        st.toast(msg)
        st.session_state[toast_key] = val
        logger.info(f"用户反馈: {'满意' if val == 1 else '不满意'}, session: {session_id}")

def display_back_to_homepage() -> None:
    st.page_link(
        "main.py",
        label="返回首页",
        query_params={"scroll_to": "main-page-header-anchor"},
    )

