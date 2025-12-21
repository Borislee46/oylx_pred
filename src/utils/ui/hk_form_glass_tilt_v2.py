from __future__ import annotations

import streamlit as st

from src.utils.ui.hk_form_glass_tilt_js import HK_FORM_GLASS_TILT_JS
from src.utils.ui.hk_form_glass_tilt_css import HK_FORM_GLASS_TILT_CSS

@st.cache_resource(show_spinner=False)
def _get_hk_form_glass_tilt_component():
    return st.components.v2.component(
        "hk_form_glass_tilt_v2",
        css=HK_FORM_GLASS_TILT_CSS,
        js=HK_FORM_GLASS_TILT_JS,
        html='<div class="hk-tilt-mount"></div>',
    )


def mount_hk_form_glass_tilt(key: str = "hk_form_glass_tilt_v2"):
    comp = _get_hk_form_glass_tilt_component()
    return comp(key=key)