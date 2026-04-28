"""Tab 2: 营收分析 — cash income + deferred revenue breakdown."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import monthly_trend_line, simple_bar, grouped_bar
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import category_filter
from src.pages.hk_dashboard.metrics.revenue_metrics import (
    cash_income_by_project, cash_income_by_quarter, cash_income_monthly,
    monthly_deferred_revenue, deferred_by_teacher,
)


def render(data: dict[str, pd.DataFrame]) -> None:
    revenue = data["revenue"]
    deferred = data["deferred_revenue"]
    class_master = data["class_master"]

    # ═══════════════════════════════════════════════
    # Section A: 现金收入分析
    # ═══════════════════════════════════════════════
    st.html("<h2>现金收入分析</h2>")

    proj_filter = category_filter(revenue, "产品品类", label="产品品类", key="rev_proj")
    filtered = revenue.copy()
    if proj_filter:
        filtered = filtered[filtered["产品品类"] == proj_filter]

    col1, col2 = st.columns(2)
    with col1:
        st.html("<h3>收入 by 产品品类</h3>")
        by_proj = cash_income_by_project(filtered)
        simple_bar(by_proj, "产品品类", "现金收入")

    with col2:
        st.html("<h3>收入 by 季度</h3>")
        by_q = cash_income_by_quarter(filtered)
        if not by_q.empty:
            grouped_bar(by_q, "季度", "现金收入", "业务归属年")
        else:
            st.caption("暂无季度数据")

    # Monthly trend
    st.html("<h3>月度收入趋势</h3>")
    monthly = cash_income_monthly(filtered)
    monthly_trend_line(monthly, "月份", "现金收入")

    # Detail table
    with st.expander("📋 收入明细表"):
        detail = (
            filtered.assign(_amt=pd.to_numeric(filtered["现金收入"], errors="coerce"))
            .groupby(["产品品类", "科目", "季度"], as_index=False)["_amt"]
            .sum()
            .rename(columns={"_amt": "现金收入"})
            .sort_values("现金收入", ascending=False)
        )
        render_filterable_table(detail, key="revenue_detail")

    # ═══════════════════════════════════════════════
    # Section B: 结转收入
    # ═══════════════════════════════════════════════
    st.html("<h2>结转收入看板</h2>")

    col1, col2 = st.columns(2)
    with col1:
        st.html("<h3>月度结转收入趋势</h3>")
        def_monthly = monthly_deferred_revenue(deferred)
        monthly_trend_line(def_monthly, "月份", "结转收入", color="#f7ab00")

    with col2:
        st.html("<h3>教师结转产能 Top 10</h3>")
        tch_df = deferred_by_teacher(deferred, class_master)
        top_teachers = tch_df.head(10)
        if not top_teachers.empty:
            simple_bar(top_teachers, "主带课教师", "结转收入", color="#f39800", horizontal=True)
        else:
            st.caption("暂无教师结转数据")

    # Deferred detail
    with st.expander("📋 结转收入明细"):
        def_detail = deferred[["班级编号", "月份", "结转收入(含税)", "累计结转收入(含税)",
                               "本月结转课时", "未完成课时(分钟)"]].dropna(subset=["结转收入(含税)"])
        render_filterable_table(def_detail.head(200), key="deferred_detail")
