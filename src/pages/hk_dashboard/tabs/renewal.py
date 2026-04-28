"""Tab 5: 续费看板 — per-teacher renewal rate + bonus calculation."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import month_filter
from src.pages.hk_dashboard.metrics.renewal_metrics import (
    calculate_renewal_rate, calculate_bonus,
)


def _available_months(roster: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(roster["进班日期"], errors="coerce").dropna()
    if dates.empty:
        return []
    months = sorted(dates.dt.to_period("M").unique().astype(str))
    return months[:-2] if len(months) > 2 else []


def render(data: dict[str, pd.DataFrame]) -> None:
    roster = data["roster"]
    class_master = data["class_master"]

    months = _available_months(roster)
    if not months:
        st.warning("花名册中无可用月份数据")
        return

    selected = month_filter(months, key="renewal_month", label="计算月份")
    if not selected:
        selected = [m for m in months if m == "2026-02"][0] if "2026-02" in months else months[-1]

    st.caption(f"当前计算月份: {selected} (队列法: {selected} — 次月)")

    renewal_df = calculate_renewal_rate(roster, class_master, month=selected)
    bonus_df = calculate_bonus(renewal_df, class_master, month=selected)

    if not renewal_df.empty:
        overall_rate = renewal_df["次月在班数"].sum() / renewal_df["当月学员数"].sum() \
            if renewal_df["当月学员数"].sum() > 0 else 0
        total_bonus = bonus_df["应发奖金(HKD)"].sum() if not bonus_df.empty else 0
        total_students = int(renewal_df["当月学员数"].sum())
        retained = int(renewal_df["次月在班数"].sum())
        teacher_count = len(renewal_df)
    else:
        overall_rate = 0; total_bonus = 0; total_students = 0; retained = 0; teacher_count = 0

    render_kpi_row([
        {"value": f"{overall_rate:.0%}", "label": "整体续费率", "accent": "green"},
        {"value": str(total_students), "label": "当月学员数", "accent": "blue"},
        {"value": str(retained), "label": "次月在班数", "accent": "blue",
         "sub": f"{teacher_count} 位教师"},
        {"value": f"{total_bonus / 1e4:.0f} 万", "label": "应发奖金 (HKD)", "accent": "amber"},
    ])

    st.html("<h3>教师续费率</h3>")
    c1, c2 = st.columns([2, 3])
    with c1:
        if not renewal_df.empty:
            simple_bar(renewal_df.head(15), "教师", "续费率", horizontal=True, color="#2563eb")
    with c2:
        st.html("<h3>奖金明细</h3>")
        if not bonus_df.empty:
            display = bonus_df[["教师", "当月学员数", "次月在班数", "续费率",
                                "单价(HKD/课时)", "当月课时", "应发奖金(HKD)"]].copy()
            display["续费率"] = display["续费率"].apply(lambda x: f"{x:.0%}")
            render_filterable_table(display, key="bonus_table")

    st.html("<h3>续费率区间分布</h3>")
    if not renewal_df.empty:
        bins = [0, 0.5, 0.75, 0.85, 1.01]
        labels = ["< 50%", "50% - 75%", "75% - 85%", "85% - 100%"]
        renewal_df["续费率区间"] = pd.cut(renewal_df["续费率"], bins=bins, labels=labels, right=False)
        dist = renewal_df["续费率区间"].value_counts().reset_index(name="count")
        dist.columns = ["区间", "count"]
        simple_bar(dist, "区间", "count", color="#0d9488")
