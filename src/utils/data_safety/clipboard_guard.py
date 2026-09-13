from pathlib import Path

import streamlit as st


def inject_clipboard_guard() -> None:
    js_path = Path("assets/ui/clipboard_guard/script.js")
    if js_path.exists():
        js_code = js_path.read_text(encoding="utf-8")
        st.iframe(
            f"<script>{js_code}</script>",
            width=1,
            height=1,
            tab_index=-1,
        )
