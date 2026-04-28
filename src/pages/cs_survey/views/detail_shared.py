from __future__ import annotations

import pandas as pd
import streamlit as st

from ..ui.blocks import render_info_cards, render_page_hero
from ..ui.charts import render_hbar_feedback
from ..ui.filter_bar import filter_bar
from ..ui.tables import theme_without_other
from ..ui.theme_css import format_metric


def feedback_tone(opinion_rate: float | None, praise_rate: float | None) -> str:
    op = 0.0 if opinion_rate is None else float(opinion_rate)
    pr = 0.0 if praise_rate is None else float(praise_rate)
    if pr >= op + 3:
        return "positive"
    if op >= pr + 3:
        return "warning"
    return "neutral"


def render_detail_hero(
    title: str,
    subtitle: str,
    *,
    badge: str | None,
    chips: list[str],
    summary: str,
    stats: list[dict[str, str]],
) -> None:
    render_page_hero(
        title,
        subtitle,
        badge=badge,
        eyebrow="Insight View",
        summary=summary,
        chips=chips,
        stats=stats,
    )


def render_detail_kpis(kpi: dict, *, scope_label: str, focus_label: str, columns: int = 3) -> None:
    score = kpi.get("评分均值")
    qty = int(kpi.get("评分量", 0))
    opinion_rate = kpi.get("意见率")
    praise_rate = kpi.get("表扬率")
    render_info_cards(
        [
            {
                "label": "当前切片评分",
                "value": format_metric(score),
                "body": f"{scope_label}下的综合评分均值。",
                "meta": f"评分样本 {qty:,}",
                "tone": "positive" if score is not None and float(score) >= 4 else "neutral",
            },
            {
                "label": "反馈结构",
                "value": f"{format_metric(opinion_rate, '{:.1f}%')} / {format_metric(praise_rate, '{:.1f}%')}",
                "body": "意见率与表扬率共同决定当前口碑结构。",
                "meta": "左侧为意见率，右侧为表扬率",
                "tone": feedback_tone(opinion_rate, praise_rate),
            },
            {
                "label": "分析焦点",
                "value": focus_label,
                "body": "当前页面围绕这一视角展开拆解与对比。",
                "meta": "下方继续查看差异来源与原始证据",
                "tone": "neutral",
            },
        ],
        columns=columns,
    )


def _text_options(series: pd.Series | None) -> list[str]:
    if series is None:
        return []
    values = series.dropna().astype(str).str.strip()
    values = values[(values != "") & (~values.str.lower().eq("nan"))]
    return sorted(values.unique().tolist())


def _selectbox_with_state(label: str, options: list[str], key: str) -> str:
    current = st.session_state.get(key)
    if current not in options:
        current = options[0]
    return st.selectbox(label, options, index=options.index(current), key=key)


def _control_spacer() -> None:
    st.markdown(
        '<div style="height: 1.7rem;" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )


def filter_theme_feedback(
    df_feedback: pd.DataFrame,
    *,
    state_prefix: str,
    label: str = "主题范围",
    group_col: str = "三大类",
    fixed_group: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    view = df_feedback.copy()
    options = ["全部"] + _text_options(view.get(group_col))
    if fixed_group is None:
        reset_col, select_col, theme_col = st.columns([0.16, 0.42, 0.42])
        with reset_col:
            _control_spacer()
            if st.button("重置", key=f"{state_prefix}_theme_reset", use_container_width=True):
                st.session_state.pop(f"{state_prefix}_theme_group", None)
                st.session_state.pop(f"{state_prefix}_theme_value", None)
        with select_col:
            if len(options) > 1:
                selected_group = _selectbox_with_state(
                    label, options, f"{state_prefix}_theme_group"
                )
            else:
                selected_group = "全部"
        theme_options = ["全部"] + [x for x in _text_options(view.get("主题")) if x and x != "其他"]
        with theme_col:
            selected_theme = _selectbox_with_state(
                "主题定位", theme_options, f"{state_prefix}_theme_value"
            )
    else:
        selected_group = fixed_group if fixed_group in options else "全部"
        theme_options = ["全部"] + [x for x in _text_options(view.get("主题")) if x and x != "其他"]
        if len(theme_options) > 1:
            selected_theme = _selectbox_with_state(
                "主题定位", theme_options, f"{state_prefix}_theme_value"
            )
        else:
            selected_theme = "全部"
    if selected_group != "全部" and group_col in view.columns:
        view = view[view[group_col].astype(str).str.strip() == selected_group].copy()
    if selected_theme != "全部" and "主题" in view.columns:
        view = view[view["主题"].astype(str).str.strip() == selected_theme].copy()
    return view, selected_group, selected_theme


def render_evidence_filters(
    df_detail: pd.DataFrame,
    df_feedback: pd.DataFrame,
    *,
    state_prefix: str,
    group_label: str,
    detail_label: str,
    detail_group_col: str = "大类",
    detail_col: str = "明细",
    feedback_group_col: str = "三大类",
    feedback_detail_col: str = "子类",
    feedback_type_col: str = "反馈类型",
    feedback_text_col: str = "反馈",
    theme_value: str | None = None,
    feedback_theme_col: str = "主题",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_view = df_detail.copy()
    feedback_view = df_feedback.copy()

    detail_group_values = set(_text_options(detail_view.get(detail_group_col)))
    feedback_group_values = set(_text_options(feedback_view.get(feedback_group_col)))
    group_options = ["全部"] + sorted(detail_group_values & feedback_group_values)
    show_group_filter = len(group_options) > 2
    type_options = ["全部"] + _text_options(feedback_view.get(feedback_type_col))

    with filter_bar("Evidence Link · 明细与原文联动"):
        col_widths = [1.05, 0.8, 1.2]
        if show_group_filter:
            col_widths.insert(0, 0.95)
        cols = st.columns(col_widths + [0.22])
        selected_group = "全部"
        detail_col_idx = 0
        if show_group_filter:
            with cols[0]:
                selected_group = _selectbox_with_state(
                    group_label, group_options, f"{state_prefix}_evidence_group"
                )
            detail_col_idx = 1
        if selected_group != "全部":
            if detail_group_col in detail_view.columns:
                detail_view = detail_view[
                    detail_view[detail_group_col].astype(str).str.strip() == selected_group
                ].copy()
            if feedback_group_col in feedback_view.columns:
                feedback_view = feedback_view[
                    feedback_view[feedback_group_col].astype(str).str.strip() == selected_group
                ].copy()

        detail_options = ["全部"] + sorted(
            set(_text_options(detail_view.get(detail_col)))
            & set(_text_options(feedback_view.get(feedback_detail_col)))
        )
        with cols[detail_col_idx]:
            selected_detail = _selectbox_with_state(
                detail_label, detail_options, f"{state_prefix}_evidence_detail"
            )
        with cols[detail_col_idx + 1]:
            selected_type = _selectbox_with_state(
                "反馈类型", type_options, f"{state_prefix}_evidence_type"
            )
        with cols[detail_col_idx + 2]:
            keyword = st.text_input(
                "关键词",
                key=f"{state_prefix}_evidence_keyword",
                placeholder="输入关键词过滤原始反馈",
            ).strip()
        with cols[-1]:
            _control_spacer()
            if st.button("重置", key=f"{state_prefix}_evidence_reset", use_container_width=True):
                st.session_state.pop(f"{state_prefix}_evidence_group", None)
                st.session_state.pop(f"{state_prefix}_evidence_detail", None)
                st.session_state.pop(f"{state_prefix}_evidence_type", None)
                st.session_state.pop(f"{state_prefix}_evidence_keyword", None)

    if selected_detail != "全部":
        if detail_col in detail_view.columns:
            detail_view = detail_view[
                detail_view[detail_col].astype(str).str.strip() == selected_detail
            ].copy()
        if feedback_detail_col in feedback_view.columns:
            feedback_view = feedback_view[
                feedback_view[feedback_detail_col].astype(str).str.strip() == selected_detail
            ].copy()

    if selected_type != "全部" and feedback_type_col in feedback_view.columns:
        feedback_view = feedback_view[
            feedback_view[feedback_type_col].astype(str).str.strip() == selected_type
        ].copy()

    if theme_value and theme_value != "全部" and feedback_theme_col in feedback_view.columns:
        feedback_view = feedback_view[
            feedback_view[feedback_theme_col].astype(str).str.strip() == theme_value
        ].copy()

    if keyword and feedback_text_col in feedback_view.columns:
        text_series = feedback_view[feedback_text_col].astype(str)
        feedback_view = feedback_view[
            text_series.str.contains(keyword, case=False, na=False)
        ].copy()

    if (
        feedback_detail_col in feedback_view.columns
        and detail_col in detail_view.columns
        and not feedback_view.empty
    ):
        matched = set(_text_options(feedback_view.get(feedback_detail_col)))
        detail_view = detail_view[
            detail_view[detail_col].astype(str).str.strip().isin(matched)
        ].copy()
    elif theme_value and theme_value != "全部":
        detail_view = detail_view.iloc[0:0].copy()

    return detail_view, feedback_view


def render_theme_bar(
    df: pd.DataFrame, color: str, height: int, key: str, top_n: int | None = None
) -> None:
    if df.empty:
        st.info("当前筛选范围暂无可展示主题，可切换到全部或查看原始反馈。")
        return
    clean = theme_without_other(df)
    if clean.empty:
        st.info("当前筛选范围只有“其他”主题，建议放宽筛选或直接查看原始反馈。")
        return
    clean = clean.assign(反馈量=pd.to_numeric(clean["反馈量"], errors="coerce")).sort_values(
        "反馈量", ascending=True
    )
    labels = [str(x) for x in clean["反馈类型"].tolist()]
    values = [float(x) for x in clean["反馈量"].tolist()]
    render_hbar_feedback(labels, values, color, height, key, top_n=top_n)


def summarize_feedback_details(
    df_feedback: pd.DataFrame,
    *,
    detail_col: str = "子类",
    type_col: str = "反馈类型",
) -> pd.DataFrame:
    if (
        df_feedback.empty
        or detail_col not in df_feedback.columns
        or type_col not in df_feedback.columns
    ):
        return pd.DataFrame()
    summary = (
        df_feedback.groupby([detail_col, type_col], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={detail_col: "明细"})
    )
    summary["反馈量"] = summary.drop(columns=["明细"], errors="ignore").sum(axis=1)
    order = ["明细", "反馈量", "意见", "表扬"]
    cols = [c for c in order if c in summary.columns] + [
        c for c in summary.columns if c not in order
    ]
    return summary[cols].sort_values("反馈量", ascending=False)
