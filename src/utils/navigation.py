from __future__ import annotations

import streamlit as st

URL_PATH_TO_SCRIPT: dict[str, str] = {
    "": "app_pages/home.py",
    "hk": "app_pages/hk.py",
}


def _page(
    url_path: str,
    title: str,
    icon: str,
    *,
    default: bool = False,
) -> st.Page:
    return st.Page(
        URL_PATH_TO_SCRIPT[url_path],
        title=title,
        icon=icon,
        url_path=None if default else url_path,
        default=default,
    )


def build_pages() -> list[st.Page]:
    return [
        _page("", "首页", ":material/home:", default=True),
        _page("hk", "Signals 留学择校系统", ":material/school:"),
    ]


def build_pages_for_user(user_email: str) -> list[st.Page]:
    del user_email
    return build_pages()


def script_path_for_url_path(url_path: str) -> str:
    return URL_PATH_TO_SCRIPT.get(url_path, "app_pages/home.py")
