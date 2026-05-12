"""Tab 2: 营收分析 — cash income + deferred revenue."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import donut_chart, grouped_bar, monthly_trend_line, simple_bar
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import category_filter
from src.pages.hk_dashboard.config import fmt_month_cn
from src.pages.hk_dashboard.metrics.revenue_metrics import (
    cash_income_by_capacity,
    cash_income_by_grade,
    cash_income_by_project,
    cash_income_by_quarter,
    cash_income_by_school,
    cash_income_by_subject,
    cash_income_monthly,
    deferred_by_teacher,
    mom_yoy_kpi,
    monthly_deferred_revenue,
    new_old_student_breakdown,
)
from src.pages.hk_dashboard.metrics.validation_helpers import build_cross_ref_matrix


def render(data: dict[str, pd.DataFrame]) -> None:
    revenue = data["revenue"]
    deferred = data["deferred_revenue"]
    class_master = data["class_master"]

    # ── Cash income ──
    st.html("<h2>现金收入</h2>")

    c1, c2 = st.columns(2)
    with c1:
        proj_filter = category_filter(revenue, "产品品类", label="产品品类", key="rev_proj")
    with c2:
        biz_filter = (
            category_filter(revenue, "业务类型", label="业务类型", key="rev_biz")
            if "业务类型" in revenue.columns
            else None
        )

    filtered = revenue.copy()
    if proj_filter:
        filtered = filtered[filtered["产品品类"] == proj_filter]
    if biz_filter:
        filtered = filtered[filtered["业务类型"] == biz_filter]

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>by 产品品类</h3>")
            by_proj = cash_income_by_project(filtered)
            simple_bar(by_proj, "产品品类", "现金收入", fmt=",.0f", height=180)
    with c2:
        with st.container(border=True):
            st.html("<h3>by 季度</h3>")
            by_q = cash_income_by_quarter(filtered)
            if not by_q.empty:
                grouped_bar(by_q, "季度", "现金收入", "业务归属年", height=180)

    with st.container(border=True):
        st.html("<h3>月度趋势</h3>")
        monthly = cash_income_monthly(filtered)
        monthly_trend_line(monthly, "月份", "现金收入", currency=True, height=180)
    # ── MoM / YoY ──
    kpi = mom_yoy_kpi(filtered)
    # Refund rate
    _all_amt = pd.to_numeric(filtered["现金收入"], errors="coerce")
    _gross = _all_amt[_all_amt > 0].sum()
    _refund = abs(_all_amt[_all_amt < 0].sum())
    _refund_rate = f"{_refund / _gross * 100:.1f}%" if _gross else "-"

    if kpi["latest_month"]:
        month_cn = fmt_month_cn(kpi["latest_month"])
        mom_str = f"{kpi['mom_pct']:+.1f}%" if kpi["mom_pct"] is not None else "-"
        yoy_str = f"{kpi['yoy_pct']:+.1f}%" if kpi["yoy_pct"] is not None else "-"
        mom_color = "normal" if kpi.get("mom_pct", 0) and kpi["mom_pct"] >= 0 else "inverse"
        yoy_color = "normal" if kpi.get("yoy_pct", 0) and kpi["yoy_pct"] >= 0 else "inverse"
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(f"{month_cn} 收入", f"{kpi['latest_amt'] / 1e4:.0f} 万")
        with c2:
            st.metric(
                "环比 (vs 上月)",
                f"{kpi['prev_amt'] / 1e4:.0f} 万",
                delta=mom_str,
                delta_color=mom_color,
            )
        with c3:
            st.metric("同比 (vs 去年同月)", "-", delta=yoy_str, delta_color=yoy_color)
        with c4:
            st.metric(
                "退费率 (整体)",
                _refund_rate,
                delta=f"退费 ¥{_refund / 1e4:.1f} 万" if _refund else None,
                delta_color="inverse",
            )

    with st.expander("收入明细"):
        detail = (
            filtered.assign(_amt=pd.to_numeric(filtered["现金收入"], errors="coerce"))
            .groupby(["产品品类", "科目", "季度"], as_index=False)["_amt"]
            .sum()
            .rename(columns={"_amt": "现金收入"})
            .sort_values("现金收入", ascending=False)
        )
        render_filterable_table(detail, key="revenue_detail")

    # ── Dimension breakdowns ──
    st.html("<h2>收入结构 (按维度)</h2>")

    c1, c2, c3 = st.columns(3)
    with c1:
        with st.container(border=True):
            st.html("<h3>by 年级</h3>")
            by_grade = cash_income_by_grade(revenue)
            if not by_grade.empty:
                simple_bar(by_grade.head(10), "年级", "现金收入", fmt=",.0f", height=180)
    with c2:
        with st.container(border=True):
            st.html("<h3>by 班容</h3>")
            by_cap = cash_income_by_capacity(revenue, class_master)
            if not by_cap.empty:
                simple_bar(by_cap.head(10), "班容", "现金收入", fmt=",.0f", height=180)
    with c3:
        with st.container(border=True):
            st.html("<h3>by 学校</h3>")
            by_school = cash_income_by_school(revenue)
            if not by_school.empty:
                simple_bar(
                    by_school.head(10), "学校", "现金收入", horizontal=True, fmt=",.0f", height=180
                )

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>by 科目</h3>")
            by_subject = cash_income_by_subject(revenue)
            if not by_subject.empty:
                simple_bar(by_subject.head(10), "科目", "现金收入", fmt=",.0f", height=180)
    with c2:
        with st.container(border=True):
            st.html("<h3>新老生收入</h3>")
            by_nos = new_old_student_breakdown(revenue)
            if not by_nos.empty:
                donut_chart(by_nos, "类型", "现金收入", max_categories=4, height=180)

    # ── Data cross-reference ──
    st.divider()
    with st.expander("验数: 数据交叉引用矩阵"):
        st.caption("检查各 CSV 之间关键字段的交集覆盖情况。100% 表示全集匹配。")
        matrix = build_cross_ref_matrix(
            data,
            "班级编码",
            ["class_master", "roster", "revenue"],
        )
        st.dataframe(matrix, width="stretch", hide_index=True)

        id_matrix = build_cross_ref_matrix(
            data,
            "资源id",
            ["kehu_ziyuan", "tmk", "qianyue"],
        )
        st.html("<h3>资源ID 交叉匹配</h3>")
        st.dataframe(id_matrix, width="stretch", hide_index=True)

        student_matrix = build_cross_ref_matrix(
            data,
            "学员编号",
            ["roster", "revenue"],
        )
        st.html("<h3>学员编号 交叉匹配</h3>")
        st.dataframe(student_matrix, width="stretch", hide_index=True)

    # ── Deferred ──
    st.html("<h2>结转收入</h2>")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>月度趋势</h3>")
            def_monthly = monthly_deferred_revenue(deferred)
            monthly_trend_line(
                def_monthly, "月份", "结转收入", color="#d97706", currency=True, height=180
            )
    with c2:
        with st.container(border=True):
            st.html("<h3>教师结转产能 Top 10</h3>")
            tch = deferred_by_teacher(deferred, class_master)
            if not tch.empty:
                simple_bar(
                    tch.head(10),
                    "主带课教师",
                    "结转收入",
                    color="#d97706",
                    horizontal=True,
                    fmt=",.0f",
                    height=180,
                )
                _def_total = pd.to_numeric(deferred["结转收入(含税)"], errors="coerce").sum()

    with st.expander("结转收入明细"):
        def_detail = deferred[
            [
                "班级编号",
                "月份",
                "结转收入(含税)",
                "累计结转收入(含税)",
                "本月结转课时",
                "未完成课时(分钟)",
            ]
        ].dropna(subset=["结转收入(含税)"])
        render_filterable_table(def_detail.head(200), key="deferred_detail")
