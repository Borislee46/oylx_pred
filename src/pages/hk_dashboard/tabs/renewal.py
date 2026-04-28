"""Tab 5: 续费看板 — per-teacher renewal rate + bonus calculation."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import month_filter
from src.pages.hk_dashboard.metrics.renewal_metrics import (
    calculate_renewal_rate, calculate_bonus,
)


def _available_months(roster: pd.DataFrame) -> list[str]:
    """Extract available YYYY-MM from roster 进班日期."""
    dates = pd.to_datetime(roster["进班日期"], errors="coerce").dropna()
    if dates.empty:
        return []
    months = sorted(dates.dt.to_period("M").unique().astype(str))
    if len(months) <= 2:
        return []
    return months[:-2]  # exclude last 2 months (last incomplete + no-next-month)


def render(data: dict[str, pd.DataFrame]) -> None:
    roster = data["roster"]
    class_master = data["class_master"]

    # ── Month selector ──
    months = _available_months(roster)
    if not months:
        st.warning("花名册中无可用月份数据")
        return

    selected = month_filter(months, key="renewal_month", label="选择计算月份")
    if not selected:
        selected = [m for m in months if m == "2026-02"][0] if "2026-02" in months else months[-1]

    st.caption(f"当前计算月份: **{selected}** → 次月 (队列法)")

    # ── Calculate ──
    renewal_df = calculate_renewal_rate(roster, class_master, month=selected)
    bonus_df = calculate_bonus(renewal_df, class_master, month=selected)

    # ── Overall KPIs ──
    if not renewal_df.empty:
        overall_rate = renewal_df["次月在班数"].sum() / renewal_df["当月学员数"].sum() \
            if renewal_df["当月学员数"].sum() > 0 else 0
        total_bonus = bonus_df["应发奖金(HKD)"].sum() if not bonus_df.empty else 0
        total_students = renewal_df["当月学员数"].sum()
        retained = renewal_df["次月在班数"].sum()
    else:
        overall_rate = 0
        total_bonus = 0
        total_students = 0
        retained = 0

    render_metric_grid([
        {"label": "整体续费率", "value": f"{overall_rate:.0%}"},
        {"label": "当月学员总数", "value": str(int(total_students))},
        {"label": "次月在班数", "value": str(int(retained))},
        {"label": "应发奖金(HKD)", "value": f"HK${total_bonus:,.0f}"},
    ], columns=4)

    # ── Per-teacher renewal rate ──
    st.html("<h3>教师续费率排名</h3>")
    col1, col2 = st.columns([2, 3])
    with col1:
        if not renewal_df.empty:
            top_r = renewal_df.head(15)
            simple_bar(top_r, "教师", "续费率", horizontal=True, color="#008a6c")
        else:
            st.caption("暂无续费率数据")

    # ── Bonus table ──
    st.html("<h3>续费奖金明细</h3>")
    if not bonus_df.empty:
        display = bonus_df[["教师", "当月学员数", "次月在班数", "续费率", "单价(HKD/课时)",
                            "当月课时", "应发奖金(HKD)"]].copy()
        display["续费率"] = display["续费率"].apply(lambda x: f"{x:.0%}")
        render_filterable_table(display, key="bonus_table")
    else:
        st.caption("暂无奖金数据")

    # ── Renewal rate distribution ──
    st.html("<h3>续费率分布</h3>")
    if not renewal_df.empty:
        bins = [0, 0.5, 0.75, 0.85, 1.01]
        labels = ["<50%", "50-75%", "75-85%", "85-100%"]
        renewal_df["续费率区间"] = pd.cut(renewal_df["续费率"], bins=bins, labels=labels, right=False)
        dist = renewal_df["续费率区间"].value_counts().reset_index(name="count")
        dist.columns = ["区间", "count"]
        simple_bar(dist, "区间", "count", color="#10b981")
