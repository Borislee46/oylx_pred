import html
import os
from pathlib import Path

import streamlit as st

from src.utils.ui.ui_utils import load_component_assets

LOGO_PATH = "assets/company_logo.png"


@st.cache_resource(show_spinner=False)
def _get_main_page_header_component():
    assets_dir = Path("assets/ui/main_page_header")
    _, script_js, _ = load_component_assets(assets_dir)

    return st.components.v2.component(
        "main_page_header_parallax",
        js=script_js,
        html="",
    )


def render_header(
    user_nickname: str,
    *,
    page_title: str | None = None,
    page_subtitle: str | None = None,
) -> None:
    assets_dir = Path("assets/ui/main_page_header")
    style_css, script_js, template_html = load_component_assets(assets_dir)

    st.html('<div id="main-page-header-anchor"></div>')

    st.html(f"<style>{style_css}</style>")

    if os.path.exists(LOGO_PATH) and os.path.getsize(LOGO_PATH) > 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(LOGO_PATH)

    title = page_title or "[company_name] 数据科学平台"
    subtitle = page_subtitle or "AI驱动的智能决策与个性化数据服务"
    header_html = template_html.replace("{{PAGE_TITLE}}", html.escape(title)).replace(
        "{{PAGE_SUBTITLE}}", html.escape(subtitle)
    )
    st.html(header_html)

    comp = _get_main_page_header_component()
    comp(key="main_page_header_parallax", height=0)
