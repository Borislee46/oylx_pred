import base64
import os

import streamlit as st


@st.cache_data
def get_base64_of_bin_file(bin_file):
    if not os.path.exists(bin_file):
        return None
    try:
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception:
        return None


def set_page_background(img_path):
    img_base64 = get_base64_of_bin_file(img_path)
    if img_base64:
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.9), rgba(255, 255, 255, 0.9)), url("data:image/jpg;base64,{img_base64}");
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )
