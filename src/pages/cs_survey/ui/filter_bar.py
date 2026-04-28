from __future__ import annotations

from contextlib import contextmanager

import streamlit as st


@contextmanager
def filter_bar(title: str = "Filters · 交叉筛选"):
    with st.container(border=True):
        st.markdown('<div class="cs-filter-bar-anchor"></div>', unsafe_allow_html=True)
        st.markdown(f'<p class="pbi-filter-title">{title}</p>', unsafe_allow_html=True)
        yield
