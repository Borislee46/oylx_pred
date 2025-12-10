import base64
from pathlib import Path

import streamlit as st

from src.utils.logger import setup_logger

header_logger = setup_logger("page3", "prediction")


@st.cache_data
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = Path.cwd() / path
    try:
        with open(full_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except (FileNotFoundError, IOError) as e:
        header_logger.error(f"无法读取logo文件 {full_path}: {e}")
        raise


def render_header(logo_base64: str) -> None:
    if logo_base64:
        html_block = f"""
            <div class="hk-header">
                <label class="hk-logo-container">
                    <input type="checkbox" class="hk-logo-trigger">
                    <img class="hk-header-logo" src="data:image/png;base64,{logo_base64}" alt="logo">
                    <span class="hk-logo-shine"></span>
                </label>
                <div>
                    <p class="hk-header-title">EasyApply</p>
                    <p class="hk-header-subtitle">留学择校系统</p>
                </div>
            </div>
        """
        try:
            st.html(html_block)
        except (AttributeError, TypeError) as e:
            header_logger.warning(f"无法使用 st.html，回退到 markdown: {e}")
            st.markdown(html_block, unsafe_allow_html=True)
    else:
        st.title("EasyApply 选校预测系统")
