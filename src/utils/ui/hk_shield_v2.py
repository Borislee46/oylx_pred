from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.ui.ui_utils import load_component_assets


@st.cache_resource(show_spinner=False)
def _get_hk_shield_component():
    assets_dir = Path(__file__).parent / "hk_shield_assets"
    _, script_js, _ = load_component_assets(assets_dir)

    return st.components.v2.component(
        "hk_shield_v2",
        js=script_js,
        html='<div class="hk-shield-mount"></div>',
    )


def mount_hk_shield_v2(key: str = "hk_shield_v2"):
    assets_dir = Path(__file__).parent / "hk_shield_assets"
    style_css, _, _ = load_component_assets(assets_dir)
    st.markdown(f"<style>{style_css}</style>", unsafe_allow_html=True)

    comp = _get_hk_shield_component()
    return comp(key=key, height=0)
