import streamlit as st


def display_back_to_homepage() -> None:
    st.page_link("main.py", label="返回首页")
