from __future__ import annotations

import pandas as pd
import streamlit as st

from src.pages.prediction.app_data import load_school_major_details_df
from src.utils.logger import setup_logger

_logger = setup_logger("page3", "prediction")

_UNI_COL = "学校"
_MAJOR_EN_COL = "专业英文名称"
_MAJOR_CN_COL = "专业中文名称"

_SKIP_IN_COMPACT = {"专业中文名称", "专业英文名称", "GPA要求", "IELTS", "TOEFL"}

_FIELD_GROUPS: dict[str, list[str]] = {
    "基本信息": ["专业中文名称", "学习年限", "学费", "授课语言"],
    "录取门槛": ["录取要求", "专业背景要求", "GPA要求"],
    "语言要求": ["IELTS", "TOEFL", "CET-6"],
    "考试与面试": ["考试要求", "是否面试", "是否笔试", "考核形式"],
    "申请流程": ["申请方式", "推荐信方式", "成绩送分要求", "特殊要求"],
    "其他": ["申请注意事项", "专业网址"],
}

_FIELD_LABELS: dict[str, str] = {
    "学习年限": "学制",
    "专业中文名称": "中文名称",
    "专业背景要求": "背景要求",
    "申请注意事项": "注意事项",
    "专业网址": "网址",
    "成绩送分要求": "送分要求",
    "推荐信方式": "推荐信",
    "是否面试": "面试",
    "是否笔试": "笔试",
}


def _lookup_row(df: pd.DataFrame, university: str, major: str) -> pd.Series | None:
    if df.empty or _UNI_COL not in df.columns:
        _logger.warning("_lookup_row: school_major_details empty or missing '%s' column", _UNI_COL)
        return None
    sub = df[df[_UNI_COL] == university]
    if sub.empty:
        _logger.info("_lookup_row: no rows for university=%s", university)
        return None
    for col in (_MAJOR_EN_COL, _MAJOR_CN_COL):
        if col in sub.columns:
            hit = sub[sub[col] == major]
            if not hit.empty:
                return hit.iloc[0]
    _logger.info("_lookup_row: no match for uni=%s major=%s", university, major)
    return None


def _format_groups(row: pd.Series) -> list[dict]:
    groups: list[dict] = []
    for group_name, fields in _FIELD_GROUPS.items():
        items: list[tuple[str, str]] = []
        for f in fields:
            val = row.get(f)
            if pd.notna(val) and str(val).strip():
                display_val = str(val).strip()
                if f == "专业网址":
                    display_val = f"[链接]({display_val})"
                items.append((f, display_val))
        if items:
            groups.append({"name": group_name, "items": items})
    return groups


def _escape_attr(value: str) -> str:
    return (
        value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")
    )


def _render_spec_html(row: pd.Series) -> str:
    # (label, 纯文本, 已安全 HTML)；网址字段直接使用 <a> 链接，普通字段输出前转义
    items: list[tuple[str, str, str]] = []
    for _group_name, fields in _FIELD_GROUPS.items():
        for f in fields:
            if f in _SKIP_IN_COMPACT:
                continue
            val = row.get(f)
            if pd.notna(val) and str(val).strip():
                label = _FIELD_LABELS.get(f, f)
                display_val = str(val).strip()
                if f == "专业网址":
                    href_val = _escape_attr(display_val)
                    html_val = (
                        f'<a href="{href_val}" target="_blank" rel="noopener">'
                        f"{_escape_attr(display_val[:50])}"
                        f'{"…" if len(display_val) > 50 else ""}</a>'
                    )
                    items.append((label, display_val, html_val))
                else:
                    items.append((label, display_val, _escape_attr(display_val)))

    if not items:
        return ""

    cells = "".join(
        f'<div class="hk-spec-item">'
        f'<span class="hk-spec-label">{label}</span>'
        f'<span class="hk-spec-value" title="{_escape_attr(text)}">{html}</span>'
        f"</div>"
        for label, text, html in items
    )
    return f'<div class="hk-major-spec-grid">{cells}</div>'


def render_major_detail_compact_html(university: str, major: str) -> str:
    df = load_school_major_details_df()
    row = _lookup_row(df, university, major)
    if row is None:
        return ""
    return _render_spec_html(row)


def render_major_detail(university: str, major: str) -> None:
    df = load_school_major_details_df()
    row = _lookup_row(df, university, major)
    if row is None:
        st.caption("暂无该专业的详细信息。")
        return

    groups = _format_groups(row)
    if not groups:
        st.caption("该专业暂无可展示的详细字段。")
        return

    for g in groups:
        st.markdown(f"**{g['name']}**")
        for label, value in g["items"]:
            st.markdown(f"- {label}：{value}")
