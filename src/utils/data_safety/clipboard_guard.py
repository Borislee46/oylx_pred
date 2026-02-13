from pathlib import Path

import streamlit.components.v1 as components


def inject_clipboard_guard() -> None:
    js_path = Path("assets/ui/clipboard_guard/script.js")
    if js_path.exists():
        js_code = js_path.read_text(encoding="utf-8")
        components.html(
            f"<script>{js_code}</script>",
            height=0,
        )
