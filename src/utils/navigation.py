from __future__ import annotations

from pathlib import Path

import streamlit as st

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
) -> st.Page | None:
    script = URL_PATH_TO_SCRIPT[url_path]
    if not (_PROJECT_ROOT / script).is_file():
        return None
    return st.Page(
        script,
        title=title,
        icon=icon,
        url_path=None if default else url_path,
        default=default,
    )


def _filter_pages(pages: list[st.Page | None]) -> list[st.Page]:
    return [p for p in pages if p is not None]


def build_pages() -> list[st.Page]:
    return _filter_pages(
        [
            _page("", "首页", ":material/home:", default=True),
            _page("hk", "Signals 留学择校系统", ":material/school:"),
        ]
    )


def script_path_for_url_path(url_path: str) -> str:
    return URL_PATH_TO_SCRIPT.get(url_path, "app_pages/home.py")
