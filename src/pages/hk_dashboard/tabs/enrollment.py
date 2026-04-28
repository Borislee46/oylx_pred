"""Tab 3: 招生转化 — funnel, consultant ranking, channel analysis."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar, donut_chart
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.metrics.funnel_metrics import (
    calculate_funnel, consultant_ranking, class_capacity_metrics,
    channel_breakdown, tmk_processing_stats,
)


def _render_funnel_html(stages: dict) -> None:
    """Render a horizontal funnel bar chart as HTML."""
    stages_list = [
        ("总资源", stages["总资源数"], "#008a6c"),
        ("已外呼", stages["已外呼"], "#10b981"),
        ("有工单", stages["有工单"], "#f7ab00"),
        ("已签约", stages["已签约"], "#f39800"),
    ]
    max_w = max(s["总资源数"] for s in [stages]) or 1
    html = '<div style="padding:1rem 0">'
    for label, val, color in stages_list:
        pct = val / max_w * 100 if max_w else 0
        html += (
            f'<div style="margin-bottom:8px">'
            f'<span style="display:inline-block;width:80px;font-weight:600;font-size:0.85rem">{label}</span>'
            f'<span style="display:inline-block;width:70px;font-weight:700;color:{color}">{val:,}</span>'
            f'<span style="display:inline-block;background:{color};height:28px;width:{pct * 0.6}%;'
            f'border-radius:4px;min-width:{max(3, pct * 0.6)}%"></span>'
            f'</div>'
        )
    html += "</div>"
    st.html(html)

    # Conversion rates
    render_metric_grid([
        {"label": "外呼率", "value": stages["外呼率"]},
        {"label": "外呼→工单率", "value": stages["外呼→工单率"]},
        {"label": "工单→签约率", "value": stages["工单→签约率"]},
    ], columns=3)


def render(data: dict[str, pd.DataFrame]) -> None:
    kehu = data["kehu_ziyuan"]
    tmk = data["tmk"]
    qianyue = data["qianyue"]
    class_master = data["class_master"]

    # ═══ Section A: 招生指标 ═══
    st.html("<h2>招生指标</h2>")
    cap = class_capacity_metrics(class_master)
    render_metric_grid([
        {"label": "行课班级", "value": str(cap["行课班级数"])},
        {"label": "满班班级", "value": str(cap["满班班级数"])},
        {"label": "满班率", "value": cap["满班率"]},
        {"label": "平均满班率", "value": cap["平均满班率"]},
    ], columns=4)

    # Class capacity table
    with st.expander("📋 班级容量明细"):
        cm = class_master.copy()
        cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
        cm["标准_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
        cm["满班率"] = (cm["当前_n"] / cm["标准_n"].replace(0, pd.NA) * 100).round(0).astype(str) + "%"
        tbl = cm[["班级编码", "班级名称", "班级状态", "当前人数", "标准人数", "最大人数", "满班率", "开课日期"]]
        tbl = tbl[tbl["班级状态"] == "正常"]
        render_filterable_table(tbl.head(100), key="capacity_detail")

    # ═══ Section B: 资源转化漏斗 ═══
    st.html("<h2>资源转化漏斗</h2>")
    funnel = calculate_funnel(kehu, tmk, qianyue)
    _render_funnel_html(funnel)

    # TMK processing stats
    st.html("<h3>TMK 处理统计</h3>")
    tstats = tmk_processing_stats(tmk)
    render_metric_grid([
        {"label": "已外呼", "value": str(tstats["已外呼"])},
        {"label": "已处理", "value": str(tstats["已处理"])},
        {"label": "待处理", "value": str(tstats["待处理"])},
        {"label": "1天内处理", "value": str(tstats["1天内处理"])},
        {"label": "平均处理时延", "value": f'{tstats["平均处理时延(天)"]}天'},
    ], columns=5)

    # ═══ Section C: 渠道分析 ═══
    st.html("<h2>渠道分析</h2>")
    col1, col2 = st.columns(2)
    with col1:
        st.html("<h3>一级渠道分布</h3>")
        ch = channel_breakdown(kehu)
        if not ch.empty:
            donut_chart(ch, "渠道", "count", max_categories=6)
    with col2:
        st.html("<h3>签约列表 二级渠道</h3>")
        if "二级获取渠道" in qianyue.columns:
            ch2 = qianyue["二级获取渠道"].value_counts().reset_index(name="count")
            ch2.columns = ["渠道", "count"]
            if not ch2.empty:
                simple_bar(ch2.head(10), "渠道", "count", horizontal=True, color="#10b981")
            else:
                st.caption("暂无二级渠道数据")

    # ═══ Section D: 顾问业绩 ═══
    st.html("<h2>顾问业绩排名</h2>")
    ranking = consultant_ranking(qianyue)
    col1, col2 = st.columns([2, 3])
    with col1:
        top15 = ranking.head(15)
        if not top15.empty:
            simple_bar(top15, "顾问姓名", "签约数", horizontal=True)
    with col2:
        render_filterable_table(ranking, key="consultant_ranking")
