"""Tab 5: 续费看板 — per-teacher renewal rate + bonus."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import month_filter
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row
from src.pages.hk_dashboard.config import fmt_month_cn
from src.pages.hk_dashboard.metrics.renewal_metrics import (
    calculate_bonus,
    calculate_renewal_rate,
)
from src.pages.hk_dashboard.metrics.validation_helpers import (
    export_excel_download,
    render_cross_validation_expander,
)


def _available_months(roster: pd.DataFrame) -> list[str]:
    dates = pd.to_datetime(roster["进班日期"], errors="coerce").dropna()
    if dates.empty:
        return []
    months = sorted(dates.dt.to_period("M").unique().astype(str))
    # Exclude current month (incomplete data) and any future months
    now_month = pd.Timestamp.now().strftime("%Y-%m")
    months = [m for m in months if m < now_month]
    return months


def render(data: dict[str, pd.DataFrame]) -> None:
    roster = data["roster"]
    class_master = data["class_master"]

    months = _available_months(roster)
    if not months:
        st.warning("花名册中无可用月份数据")
        return

    selected = month_filter(
        months,
        key="renewal_month",
        label="计算月份",
        format_func=fmt_month_cn,
        default_index=len(months) - 1,
    )

    selected_cn = fmt_month_cn(selected)
    st.caption(f"计算月份: {selected_cn} | 队列法: {selected_cn} 学员 → 次月在班 → 续费率")

    renewal_df = calculate_renewal_rate(roster, class_master, month=selected)
    bonus_df = calculate_bonus(renewal_df, class_master, month=selected)

    if not renewal_df.empty:
        overall_rate = (
            renewal_df["次月在班数"].sum() / renewal_df["当月学员数"].sum()
            if renewal_df["当月学员数"].sum() > 0
            else 0
        )
        total_bonus = bonus_df["应发奖金(HKD)"].sum() if not bonus_df.empty else 0
        total_students = int(renewal_df["当月学员数"].sum())
        retained = int(renewal_df["次月在班数"].sum())
        teacher_count = len(renewal_df)
    else:
        overall_rate = 0
        total_bonus = 0
        total_students = 0
        retained = 0
        teacher_count = 0

    render_kpi_row(
        [
            {
                "value": f"{overall_rate:.0%}",
                "label": "整体续费率",
                "accent": "green",
                "formula": "次月在班数 / 当月学员数",
            },
            {
                "value": str(total_students),
                "label": "当月学员数",
                "accent": "blue",
                "formula": "COUNT(进班 <= 月底 AND (离班 >= 月初 OR 离班 IS NULL))",
            },
            {
                "value": str(retained),
                "label": "次月在班数",
                "accent": "blue",
                "sub": f"{teacher_count} 位教师",
                "formula": "上述学员中 次月仍在班的人数",
            },
            {
                "value": f"{total_bonus / 1e4:.0f} 万",
                "label": "应发奖金 (HKD)",
                "accent": "amber",
                "formula": "SUM(单价 × 当月课时) | 单价 = LOOKUP(续费率, 班型人数)",
            },
        ]
    )

    st.html("<h3>教师续费率</h3>")
    c1, c2 = st.columns([2, 3])
    with c1:
        if not renewal_df.empty:
            simple_bar(
                renewal_df.head(15),
                "教师",
                "续费率",
                horizontal=True,
                color="#2563eb",
                height=240,
                fmt=".0%",
            )
        st.html(
            '<div class="hk-note">主带课教师 | GROUP BY 教师 | 续费率 = 次月在班 / 当月学员</div>'
        )
    with c2:
        st.html("<h3>奖金明细</h3>")
        if not bonus_df.empty:
            display = bonus_df[
                [
                    "教师",
                    "当月学员数",
                    "次月在班数",
                    "续费率",
                    "单价(HKD/课时)",
                    "当月课时",
                    "应发奖金(HKD)",
                ]
            ].copy()
            display["续费率"] = display["续费率"].apply(lambda x: f"{x:.0%}")
            render_filterable_table(display, key="bonus_table")
        st.html(
            '<div class="hk-note">奖金阶梯: 续费率[50%,75%)/[75%,85%)/[85%,100%] × 班型[1-24]/[25-49]/[50+] 人</div>'
        )

    st.html("<h3>续费率区间分布</h3>")
    if not renewal_df.empty:
        bins = [0, 0.5, 0.75, 0.85, 1.01]
        labels = ["< 50%", "50% - 75%", "75% - 85%", "85% - 100%"]
        renewal_df["续费率区间"] = pd.cut(
            renewal_df["续费率"], bins=bins, labels=labels, right=False
        )
        dist = renewal_df["续费率区间"].value_counts().reset_index(name="count")
        dist.columns = ["区间", "count"]
        simple_bar(dist, "区间", "count", color="#0d9488", height=200)

    # ── Validation expander ──
    st.divider()
    with st.expander("验数: 续费率 系统 vs Excel 对照"):
        st.caption(
            "系统教师归属: 主带课教师 优先 → 兜底 教师列第一位。"
            "Excel 使用人工指定的主教师，口径差异会导致个别教师偏差。"
        )
        render_cross_validation_expander(
            title="教师续费率逐行对照",
            sys_df=renewal_df if not renewal_df.empty else pd.DataFrame(),
            excel_df=None,
            sys_label="系统计算",
            excel_label="人工 Excel",
            key="renewal_diff",
        )

        st.html("<h3>续费率差异排名</h3>")
        if not renewal_df.empty:
            # Internal cross-check: compare teacher roster vs class_master mapping
            cm_valid = data["class_master"][["班级编码", "主带课教师", "教师"]].copy()
            cm_valid["_t"] = cm_valid["主带课教师"]
            mask = cm_valid["_t"].isna()
            cm_valid.loc[mask, "_t"] = cm_valid.loc[mask, "教师"].apply(
                lambda x: str(x).split(",")[0].split("(")[0].strip() if pd.notna(x) else None
            )
            has_primary = cm_valid["主带课教师"].notna().sum()
            total_classes = len(cm_valid)
            st.caption(
                f"教师归属: 维表 {total_classes} 个班级中 {has_primary} 个有主带课教师 "
                f"({has_primary / total_classes * 100:.0f}%)，剩余用教师列第一位兜底"
            )

            # Export
            if st.button("导出续费率明细 (Excel)", key="export_renewal"):
                export_excel_download(renewal_df, "续费率_系统计算.xlsx")
