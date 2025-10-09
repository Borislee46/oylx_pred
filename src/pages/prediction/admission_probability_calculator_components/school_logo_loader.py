import base64
import os

import streamlit as st


@st.cache_data
def get_logo_path(school_name):
    def normalize_name(name: str) -> str:
        s = name.strip()
        bad_chars = [":", " ", "\\", "/", "\t", "\n"]
        for ch in bad_chars:
            s = s.replace(ch, "_")
        return s

    safe_school_name = normalize_name(school_name)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.normpath(
        os.path.join(script_dir, "..", "..", "..", "..", "assets", "school_logos")
    )

    candidates = [
        os.path.join(base_dir, f"{safe_school_name}.png"),
        os.path.join(base_dir, f"{safe_school_name}.jpg"),
        os.path.join(base_dir, f"{safe_school_name}.jpeg"),
        os.path.join(base_dir, f"{safe_school_name}.webp"),
    ]

    for path in candidates:
        normalized_logo_path = os.path.normpath(path)
        if os.path.exists(normalized_logo_path):
            with open(normalized_logo_path, "rb") as f:
                data = f.read()
            encoded = base64.b64encode(data).decode()
            ext = os.path.splitext(normalized_logo_path)[1].lstrip(".").lower()
            if ext == "jpg":
                ext = "jpeg"
            return f"data:image/{ext};base64,{encoded}"

    return None
