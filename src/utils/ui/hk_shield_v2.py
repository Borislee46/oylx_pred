from pathlib import Path

import streamlit as st

from src.utils.ui.ui_utils import load_component_assets


@st.cache_resource(show_spinner=False, scope="global")
def _get_hk_shield_component():
    assets_dir = Path("assets/ui/hk_shield")
    _, script_js, _ = load_component_assets(assets_dir)

    return st.components.v2.component(
        "hk_shield_v2",
        js=script_js,
        html='<div class="hk-shield-mount"></div>',
    )


def mount_hk_shield_v2(key: str = "hk_shield_v2"):
    assets_dir = Path("assets/ui/hk_shield")
    style_css, _, _ = load_component_assets(assets_dir)
    st.html(f"<style>{style_css}</style>")

    comp = _get_hk_shield_component()
    return comp(key=key, height=0)
