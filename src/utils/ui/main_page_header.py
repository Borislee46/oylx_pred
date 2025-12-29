import os
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from src.utils.ui.ui_utils import load_component_assets

LOGO_PATH = "assets/company_logo.png"


def render_header(user_nickname: str) -> None:
    assets_dir = Path(__file__).parent / "main_page_header_assets"
    style_css, script_js, template_html = load_component_assets(assets_dir)

    st.markdown('<div id="main-page-header-anchor"></div>', unsafe_allow_html=True)

    st.markdown(f"<style>{style_css}</style>", unsafe_allow_html=True)

    if os.path.exists(LOGO_PATH):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(LOGO_PATH, width="stretch")

    st.markdown(template_html, unsafe_allow_html=True)

    components.html(
        f"<script>{script_js}</script>",
        height=0,
        width=0,
    )
