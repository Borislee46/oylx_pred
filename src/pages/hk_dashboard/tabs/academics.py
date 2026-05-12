"""Tab 4: 教务教学 — roster, attendance, teacher workload."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import donut_chart, simple_bar
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.filters import category_filter
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.metrics.validation_helpers import build_cross_ref_matrix


def _teacher_workload(class_master: pd.DataFrame) -> pd.DataFrame:
    cm = class_master.copy()
    hours_col = "实际上课时长（去除赠课）" if "实际上课时长（去除赠课）" in cm.columns else "课次"
    cm[hours_col] = pd.to_numeric(cm[hours_col], errors="coerce")
    cm["课次_n"] = pd.to_numeric(cm["课次"], errors="coerce")
    teacher_data = []
    for _, row in cm.iterrows():
        raw = row.get("教师", "")
        if pd.isna(raw):
            continue
        names = [t.strip().split("(")[0] for t in str(raw).split(",") if t.strip()]
        for name in names:
            teacher_data.append(
                {
                    "教师": name,
                    "课时": row[hours_col] if pd.notna(row[hours_col]) else 0,
                    "课次": row["课次_n"] if pd.notna(row["课次_n"]) else 0,
                    "班级编码": row["班级编码"],
                }
            )
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

    st.html("<h2>学员名册</h2>")
    valid_count = (roster["有效状态"] == "有效").sum()
    invalid_count = (roster["有效状态"] == "无效").sum()
    total_students = roster["学员编号"].nunique()

    render_metric_grid(
        [
            {"label": "总学员", "value": str(total_students)},
            {"label": "在读", "value": str(valid_count)},
            {"label": "离班", "value": str(invalid_count)},
        ],
        columns=3,
    )
    st.html('<div class="hk-note">花名册 | NUNIQUE(学员编号) | COUNT(有效状态 = 有效|无效)</div>')

    status_filter = category_filter(roster, "有效状态", label="状态", key="roster_status")
    r = roster.copy()
    if status_filter:
        r = r[r["有效状态"] == status_filter]

    c1, c2 = st.columns([2, 1])
    with c1:
        with st.expander("学员花名册"):
            display_cols = [
                "学员编号",
                "学员姓名",
                "班级编码",
                "进班日期",
                "有效状态",
                "打卡次数",
                "离班方式",
                "实缴金额",
            ]
            avail = [c for c in display_cols if c in r.columns]
            render_filterable_table(r[avail].head(300), key="roster_table")
    with c2:
        with st.container(border=True):
            st.html("<h3>年级分布</h3>")
            if "学员年级(自动更新)" in r.columns:
                grade_cnt = r["学员年级(自动更新)"].value_counts().reset_index(name="count")
                grade_cnt.columns = ["年级", "count"]
                if not grade_cnt.empty:
                    simple_bar(
                        grade_cnt.head(10),
                        "年级",
                        "count",
                        horizontal=True,
                        color="#7c3aed",
                        height=180,
                    )

    st.html("<h2>考勤</h2>")
    r["打卡_n"] = pd.to_numeric(r["打卡次数"], errors="coerce")
    avg_att = r["打卡_n"].mean()
    total_att = r["打卡_n"].sum()
    max_att = r["打卡_n"].max()
    render_metric_grid(
        [
            {"label": "总打卡次数", "value": f"{total_att:,.0f}"},
            {"label": "人均打卡", "value": f"{avg_att:.1f}"},
            {"label": "最高打卡", "value": f"{max_att:,.0f}"},
        ],
        columns=3,
    )
    st.html('<div class="hk-note">花名册 | SUM(打卡次数) | AVG(打卡次数)</div>')

    with st.expander("班级打卡统计"):
        att_cls = r.groupby("班级编码", as_index=False).agg(
            学员数=("学员编号", "nunique"), 总打卡=("打卡_n", "sum"), 人均打卡=("打卡_n", "mean")
        )
        att_cls["人均打卡"] = att_cls["人均打卡"].round(1)
        if not att_cls.empty:
            names = class_master[["班级编码", "班级名称"]].drop_duplicates()
            att_cls = att_cls.merge(names, on="班级编码", how="left")
            render_filterable_table(att_cls, key="attendance_table")

    # ── Departure reasons ──
    st.html("<h2>离班分析</h2>")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>离班方式</h3>")
            if "离班方式" in roster.columns:
                leave_cnt = roster["离班方式"].value_counts().reset_index(name="count")
                leave_cnt.columns = ["离班方式", "count"]
                simple_bar(leave_cnt, "离班方式", "count", color="#d97706", height=180)
            st.html('<div class="hk-note">花名册 | GROUP BY 离班方式</div>')
    with c2:
        with st.container(border=True):
            st.html("<h3>有效状态分布</h3>")
            status_cnt = roster["有效状态"].value_counts().reset_index(name="count")
            status_cnt.columns = ["状态", "count"]
            if not status_cnt.empty:
                donut_chart(status_cnt, "状态", "count", max_categories=4, height=180)
            st.html('<div class="hk-note">花名册 | GROUP BY 有效状态</div>')

    st.html("<h2>教师课量</h2>")
    workload = _teacher_workload(class_master)
    c1, c2 = st.columns([2, 3])
    with c1:
        if not workload.empty:
            simple_bar(
                workload.head(15), "教师", "课时", horizontal=True, color="#2563eb", height=180
            )
    with c2:
        render_filterable_table(workload, key="workload_table")
    st.html(
        '<div class="hk-note">维表.教师 | SPLIT co-teachers | 每人按班级计课时 | SUM(实际上课时长)</div>'
    )

    # ── Classroom distribution ──
    st.html("<h2>教室与行课分布</h2>")
    cm = class_master.copy()

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>教室分布</h3>")
            if "教室" in cm.columns:
                room_cnt = cm["教室"].value_counts().reset_index(name="班级数")
                room_cnt.columns = ["教室", "班级数"]
                if not room_cnt.empty:
                    simple_bar(
                        room_cnt.head(12),
                        "教室",
                        "班级数",
                        horizontal=True,
                        color="#7c3aed",
                        height=180,
                    )
            st.html('<div class="hk-note">维表 | GROUP BY 教室 | COUNT(正常班级)</div>')

    with c2:
        with st.container(border=True):
            st.html("<h3>上课时段</h3>")
            if "上课时段" in cm.columns:
                slot_cnt = cm["上课时段"].value_counts().reset_index(name="count")
                slot_cnt.columns = ["时段", "count"]
                if not slot_cnt.empty:
                    donut_chart(slot_cnt, "时段", "count", max_categories=5, height=180)
            st.html('<div class="hk-note">维表 | GROUP BY 上课时段</div>')

    # Daily class overview
    with st.container(border=True):
        st.html("<h3>行课日期分布</h3>")
        if "开课日期" in cm.columns:
            cm["开课_dt"] = pd.to_datetime(cm["开课日期"], errors="coerce")
            date_dist = cm["开课_dt"].dropna().dt.to_period("M").value_counts().sort_index()
            date_dist.index = date_dist.index.map(lambda p: f"{p.year}年{p.month}月")
            date_dist = date_dist.reset_index(name="count")
            date_dist.columns = ["月份", "count"]
            simple_bar(date_dist, "月份", "count", color="#0d9488", height=180)
            st.html('<div class="hk-note">维表 | GROUP BY MONTH(开课日期) | 每月开课班级数</div>')

    # Class capacity detail
    st.html("<h3>班级容量分布</h3>")
    cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
    cm["标准_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
    active_cm = cm[(cm["班级状态"] == "正常") & (cm["当前_n"] > 0)]
    if not active_cm.empty:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("行课班级", len(active_cm))
        with c2:
            full = (active_cm["当前_n"] >= active_cm["标准_n"]).sum()
            st.metric("满班", full, delta=f"{full / len(active_cm) * 100:.0f}%")
        with c3:
            avg = (active_cm["当前_n"] / active_cm["标准_n"].replace(0, pd.NA)).mean()
            st.metric("平均满班率", f"{avg * 100:.0f}%" if pd.notna(avg) else "-")

    # ── Data cross-reference ──
    st.divider()
    with st.expander("验数: 花名册 × 维表 × 收入人次 交叉匹配"):
        matrix = build_cross_ref_matrix(data, "班级编码", ["roster", "class_master", "revenue"])
        st.dataframe(matrix, width="stretch", hide_index=True)
        st.caption("170 个班级编码应在三表之间全覆盖。如有 <100% 交集，请检查数据导出时间窗口。")
