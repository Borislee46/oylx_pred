"""Tab 1: 综合概览 — at-a-glance operational health."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import monthly_trend_line, simple_bar, donut_chart
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.metrics.funnel_metrics import consultant_ranking
from src.pages.hk_dashboard.metrics.revenue_metrics import (
    total_cash_income, cash_income_monthly, cash_income_by_project,
)
from src.pages.hk_dashboard.metrics.funnel_metrics import class_capacity_metrics, channel_breakdown


def render(data: dict[str, pd.DataFrame]) -> None:
    revenue = data["revenue"]
    kehu = data["kehu_ziyuan"]
    qianyue = data["qianyue"]
    class_master = data["class_master"]
    roster = data["roster"]

    # ── KPI Row ──
    total_cash = total_cash_income(revenue)
    signed_count = qianyue["签约单id"].notna().sum()
    class_count = (class_master["班级状态"] == "正常").sum()
    active_students = (roster["有效状态"] == "有效").sum()

    render_metric_grid([
        {"label": "累计现金收入", "value": f"HK${total_cash:,.0f}"},
        {"label": "签约学员数", "value": str(signed_count)},
        {"label": "行课班级数", "value": str(class_count)},
        {"label": "在读学员数", "value": str(active_students)},
    ], columns=4)

    # ── Row 2: Revenue trend + Channel donut ──
    col1, col2 = st.columns([3, 2])
    with col1:
        st.html("<h3>月度现金收入趋势</h3>")
        monthly = cash_income_monthly(revenue)
        if not monthly.empty:
            monthly_trend_line(monthly, "月份", "现金收入")
        else:
            st.caption("暂无月度收入数据")

    with col2:
        st.html("<h3>资源渠道分布</h3>")
        ch_df = channel_breakdown(kehu)
        if not ch_df.empty:
            donut_chart(ch_df, "渠道", "count", max_categories=5)
        else:
            st.caption("暂无渠道数据")

    # ── Row 3: Consultant ranking + Class capacity ──
    col1, col2 = st.columns(2)
    with col1:
        st.html("<h3>顾问签约 Top 10</h3>")
        ranking = consultant_ranking(qianyue)
        top = ranking.head(10)
        if not top.empty:
            simple_bar(top, "顾问姓名", "签约数", horizontal=True)
        else:
            st.caption("暂无签约数据")

    with col2:
        st.html("<h3>现金收入 by 产品</h3>")
        by_proj = cash_income_by_project(revenue)
        if not by_proj.empty:
            simple_bar(by_proj, "产品品类", "现金收入")
        else:
            st.caption("暂无产品收入数据")

    # ── Row 4: Class capacity summary ──
    st.html("<h3>班级容量摘要</h3>")
    cap = class_capacity_metrics(class_master)
    render_metric_grid([
        {"label": "行课班级", "value": str(cap["行课班级数"])},
        {"label": "满班班级", "value": str(cap["满班班级数"])},
        {"label": "满班率", "value": cap["满班率"]},
        {"label": "平均满班率", "value": cap["平均满班率"]},
    ], columns=4)
