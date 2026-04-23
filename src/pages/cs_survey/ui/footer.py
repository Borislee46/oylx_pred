from __future__ import annotations

import streamlit as st


def render_footer(*, show_back_to_overview: bool, overview_url: str = "pages/cs_survey.py") -> None:
    st.markdown('<div class="pbi-nav-row"></div>', unsafe_allow_html=True)
    nav = st.columns([1, 6, 1])
    with nav[0]:
        if show_back_to_overview:
            st.page_link(overview_url, label="← 返回概览")
    with nav[2]:
        st.page_link(
            "main.py",
            label="返回首页 ↩",
            query_params={"scroll_to": "main-page-header-anchor"},
        )
