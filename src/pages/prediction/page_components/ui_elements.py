import base64
import html
from pathlib import Path

import streamlit as st

from src.utils.logger import setup_logger
from src.utils.ui.hk_shield_v2 import mount_hk_shield_v2

logger = setup_logger("page3", "prediction")


@st.cache_data(show_spinner=False)
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = Path.cwd() / path
    return base64.b64encode(full_path.read_bytes()).decode()


def render_header(logo_base64: str) -> None:
    mask_base64 = get_product_logo_image_as_base64("assets/shield_metal.png")
    metal_base64 = get_product_logo_image_as_base64("assets/shield_mask.png")

    if not logo_base64:
        st.title("EasyApply 选校预测系统")
        return

    mount_hk_shield_v2()

    html_block = f"""
        <div class="hk-header">
            <div class="hk-logo-container" 
                 style="--logo-mask-url: url(data:image/png;base64,{mask_base64}); --metal-tex-url: url(data:image/png;base64,{metal_base64})">
                <div class="hk-metal-layer"></div>
                <img class="hk-header-logo" src="data:image/png;base64,{logo_base64}" alt="logo">
            </div>
            <div>
                <p class="hk-header-title">EasyApply</p>
                <p class="hk-header-subtitle">留学择校系统</p>
            </div>
        </div>
    """
    st.html(html_block)


def display_feedback_section(session_id: str) -> None:
    key = f"feedback_{session_id}"
    toast_key = f"{key}_toast_sent"

    if (val := st.feedback("thumbs", key=key)) is not None:
        val = int(val)
        if st.session_state.get(toast_key) != val:
            st.toast(
                "感谢您的肯定！我们会继续努力！" if val == 1 else "收到您的反馈，我们会持续改进！"
            )
            st.session_state[toast_key] = val
            logger.info(f"用户反馈: {'满意' if val == 1 else '不满意'}, session: {session_id}")


def display_back_to_homepage() -> None:
    st.page_link("main.py", label="返回首页", query_params={"scroll_to": "main-page-header-anchor"})


THOUGHT_BUBBLE_STYLE = """
    border-left: 1.5px solid #efefef;
    padding-left: 0.8rem;
    margin-top: -12px;
    margin-bottom: 8px;
    color: #a0a0a0;
    font-style: italic;
    font-size: 0.82em;
    line-height: 1.3;
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
"""


def render_thought_bubble(logs: list[str], placeholder: st.delta_generator.DeltaGenerator) -> None:
    if not logs:
        return

    thought_content = f"""
    <div style="{THOUGHT_BUBBLE_STYLE}">
        {"<br>".join(logs)}
    </div>
    """
    placeholder.markdown(thought_content, unsafe_allow_html=True)


def render_thought_bubble_with_wait_pulse(
    logs: list[str], placeholder: st.delta_generator.DeltaGenerator
) -> None:
    if not logs:
        return
    escaped = [html.escape(x) for x in logs]
    pulse = (
        ' <span class="hk-thought-wait">处理中'
        '<span class="hk-thought-wait-d1">.</span>'
        '<span class="hk-thought-wait-d2">.</span>'
        '<span class="hk-thought-wait-d3">.</span>'
        "</span>"
    )
    escaped[-1] = escaped[-1] + pulse
    inner = "<br>".join(escaped)
    thought_content = f'<div style="{THOUGHT_BUBBLE_STYLE}">{inner}</div>'
    placeholder.markdown(thought_content, unsafe_allow_html=True)
