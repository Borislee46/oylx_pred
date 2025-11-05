import streamlit as st

from src.utils.logger import setup_logger

navigation_logger = setup_logger("page3", "prediction")


def display_back_to_homepage() -> None:
    """显示返回首页的导航链接"""
    try:
        st.page_link("main.py", label="返回首页")
    except Exception as e:
        navigation_logger.error(f"导航链接显示失败: {e}", exc_info=True)
        # 回退方案：使用 markdown 链接
        st.markdown("[返回首页](main.py)")
