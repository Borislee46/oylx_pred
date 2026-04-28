"""Tab 1: 综合概览 — at-a-glance operational health."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import monthly_trend_line, simple_bar, donut_chart
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row
from src.pages.hk_dashboard.metrics.revenue_metrics import (
    cash_income_monthly, cash_income_by_project,
)
from src.pages.hk_dashboard.metrics.funnel_metrics import consultant_ranking, channel_breakdown


def render(data: dict[str, pd.DataFrame]) -> None:
    revenue = data["revenue"]
    kehu = data["kehu_ziyuan"]
    qianyue = data["qianyue"]

    # ── Month selector ──
    monthly = cash_income_monthly(revenue)

    # ── Revenue trend + channel ──
    c1, c2 = st.columns([3, 2])
    with c1:
        st.html("<h3>月度现金收入</h3>")
        if not monthly.empty:
            monthly_trend_line(monthly, "月份", "现金收入")
    with c2:
        st.html("<h3>资源渠道分布</h3>")
        ch = channel_breakdown(kehu)
        if not ch.empty:
            donut_chart(ch, "渠道", "count", max_categories=5)

    # ── Consultant + Product breakdown ──
    c1, c2 = st.columns(2)
    with c1:
        st.html("<h3>顾问签约 Top 10</h3>")
        ranking = consultant_ranking(qianyue)
        if not ranking.empty:
            simple_bar(ranking.head(10), "顾问姓名", "签约数", horizontal=True)
    with c2:
        st.html("<h3>现金收入 by 产品</h3>")
        by_proj = cash_income_by_project(revenue)
        if not by_proj.empty:
            simple_bar(by_proj, "产品品类", "现金收入")
