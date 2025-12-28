from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.ui.hk_form_glass_tilt_css import HK_FORM_GLASS_TILT_CSS


def _load_js_module() -> str:
    js_dir = Path(__file__).parent / "hk_tilt"
    modules = [
        "constants.js",
        "utils.js",
        "performance.js",
        "core.js",
        "main.js",
    ]

    js_content = ""
    for mod in modules:
        mod_path = js_dir / mod
        if mod_path.exists():
            js_content += f"\n// --- JS 模块: {mod} ---\n"
            js_content += mod_path.read_text(encoding="utf-8")
        else:
            raise FileNotFoundError(f"缺失 JS 模块: {mod_path}")

    return f"export default function(component) {{\n{js_content}\n}}"


@st.cache_resource(show_spinner=False)
def _get_hk_form_glass_tilt_component():
    js_code = _load_js_module()
    return st.components.v2.component(
        "hk_form_glass_tilt_v2",
        css=HK_FORM_GLASS_TILT_CSS,
        js=js_code,
        html='<div class="hk-tilt-mount"></div>',
    )


def mount_hk_form_glass_tilt(key: str = "hk_form_glass_tilt_v2"):
    comp = _get_hk_form_glass_tilt_component()
    return comp(key=key)
