import base64
import os

import streamlit as st


@st.cache_data
def get_logo_path(school_name):
    safe_school_name = school_name.strip().replace(" ", "_").replace(":", "_")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "..", "..", "..", "..", "assets", "school_logos")
    logo_path = os.path.join(base_dir, f"{safe_school_name}.png")

    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            data = f.read()
        encoded = base64.b64encode(data).decode()
        return f"data:image/png;base64,{encoded}"

    return None
