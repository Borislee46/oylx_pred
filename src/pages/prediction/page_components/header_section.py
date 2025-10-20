import base64
from pathlib import Path

import streamlit as st


@st.cache_data
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = Path.cwd() / path
    with open(full_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()


def render_header(logo_base64: str) -> None:
    if logo_base64:
        html_block = f"""
            <style>
                .shield-logo {{
                    transform: scale(1);
                    filter: drop-shadow(0 0 3px rgba(33, 255, 244, 0.4));
                    transition: transform 0.3s ease-in-out, filter 0.3s ease-in-out;
                }}
                .shield-logo:hover {{
                    transform: scale(1.1);
                    filter: drop-shadow(0 0 10px rgba(33, 255, 244, 0.8));
                }}
            </style>
            <div style="display: flex; align-items: center; gap: 15px;">
                <img class="shield-logo" src="data:image/png;base64,{logo_base64}" alt="logo" style="width:70px; height:auto;">
                <div>
                    <p style="font-size: 28px; font-weight: bold; margin: 0; line-height: 1.2;">EasyApply</p>
                    <p style="font-size: 18px; margin: 0; line-height: 1.2; color: #555;">留学择校系统</p>
                </div>
            </div>
        """
        try:
            st.html(html_block)
        except Exception:
            st.markdown(html_block, unsafe_allow_html=True)
    else:
        st.title("EasyApply 选校预测系统")
