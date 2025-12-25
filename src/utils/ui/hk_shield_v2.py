from __future__ import annotations

import streamlit as st

from src.utils.ui.hk_shield_effect_css import HK_SHIELD_EFFECT_CSS
from src.utils.ui.hk_shield_effect_js import HK_SHIELD_EFFECT_JS


@st.cache_resource(show_spinner=False)
def _get_hk_shield_v2_component():
    return st.components.v2.component(
        "hk_shield_v2",
        css=HK_SHIELD_EFFECT_CSS,
        js=HK_SHIELD_EFFECT_JS,
        html='<div class="hk-shield-mount"></div>',
    )


def mount_hk_shield_v2(key: str = "hk_shield_v2"):
    comp = _get_hk_shield_v2_component()
    return comp(key=key, height=0)
