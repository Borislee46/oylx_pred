"""Tab 4: 教务教学 — roster, attendance, teacher workload."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import category_filter


def _teacher_workload(class_master: pd.DataFrame) -> pd.DataFrame:
    cm = class_master.copy()
    hours_col = "实际上课时长（去除赠课）" if "实际上课时长（去除赠课）" in cm.columns else "课次"
    cm[hours_col] = pd.to_numeric(cm[hours_col], errors="coerce")
    cm["课次_n"] = pd.to_numeric(cm["课次"], errors="coerce")

    teacher_data = []
    for _, row in cm.iterrows():
        teachers_raw = row.get("教师", "")
        if pd.isna(teachers_raw):
            continue
        names = [t.strip().split("(")[0] for t in str(teachers_raw).split(",") if t.strip()]
        for name in names:
            teacher_data.append({
                "教师": name,
                "课时": row[hours_col] if pd.notna(row[hours_col]) else 0,
                "课次": row["课次_n"] if pd.notna(row["课次_n"]) else 0,
                "班级编码": row["班级编码"],
            })

    df = pd.DataFrame(teacher_data)
    if df.empty:
        return df
    return (
        df.groupby("教师", as_index=False)
        .agg(课次=("课次", "sum"), 课时=("课时", "sum"), 班级数=("班级编码", "nunique"))
        .sort_values("课时", ascending=False)
    )


def render(data: dict[str, pd.DataFrame]) -> None:
    roster = data["roster"]
    class_master = data["class_master"]

    # ── Roster ──
    st.html("<h2>学员名册</h2>")
    valid_count = (roster["有效状态"] == "有效").sum()
    invalid_count = (roster["有效状态"] == "无效").sum()
    total_students = roster["学员编号"].nunique()

    render_metric_grid([
        {"label": "总学员数", "value": str(total_students)},
        {"label": "在读", "value": str(valid_count)},
        {"label": "离班", "value": str(invalid_count)},
    ], columns=3)

    status_filter = category_filter(roster, "有效状态", label="学员状态", key="roster_status")
    r = roster.copy()
    if status_filter:
        r = r[r["有效状态"] == status_filter]

    c1, c2 = st.columns([2, 1])
    with c1:
        with st.expander("学员花名册", expanded=False):
            display_cols = ["学员编号", "学员姓名", "班级编码", "进班日期", "有效状态",
                            "打卡次数", "离班方式", "实缴金额"]
            avail = [c for c in display_cols if c in r.columns]
            render_filterable_table(r[avail].head(300), key="roster_table")
    with c2:
        st.html("<h3>学员年级分布</h3>")
        if "学员年级(自动更新)" in r.columns:
            grade_cnt = r["学员年级(自动更新)"].value_counts().reset_index(name="count")
            grade_cnt.columns = ["年级", "count"]
            if not grade_cnt.empty:
                simple_bar(grade_cnt.head(10), "年级", "count", horizontal=True, color="#7c3aed")

    # ── Attendance ──
    st.html("<h2>考勤统计</h2>")
    r["打卡_n"] = pd.to_numeric(r["打卡次数"], errors="coerce")
    avg_att = r["打卡_n"].mean()
    total_att = r["打卡_n"].sum()
    max_att = r["打卡_n"].max()

    render_metric_grid([
        {"label": "总打卡次数", "value": f"{total_att:,.0f}"},
        {"label": "人均打卡", "value": f"{avg_att:.1f}"},
        {"label": "最高打卡", "value": f"{max_att:,.0f}"},
    ], columns=3)

    with st.expander("班级打卡统计"):
        att_cls = (
            r.groupby("班级编码", as_index=False)
            .agg(学员数=("学员编号", "nunique"), 总打卡=("打卡_n", "sum"),
                 人均打卡=("打卡_n", "mean"))
        )
        att_cls["人均打卡"] = att_cls["人均打卡"].round(1)
        if not att_cls.empty:
            names = class_master[["班级编码", "班级名称"]].drop_duplicates()
            att_cls = att_cls.merge(names, on="班级编码", how="left")
            render_filterable_table(att_cls, key="attendance_table")

    # ── Teacher workload ──
    st.html("<h2>教师课量</h2>")
    workload = _teacher_workload(class_master)
    c1, c2 = st.columns([2, 3])
    with c1:
        if not workload.empty:
            simple_bar(workload.head(15), "教师", "课时", horizontal=True, color="#2563eb")
    with c2:
        render_filterable_table(workload, key="workload_table")
