"""HK Dashboard — main render entry point."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row
from src.pages.hk_dashboard.data_loader import load_all_data
from src.pages.hk_dashboard.styles import inject_hk_dashboard_styles


def _summary_ribbon(data: dict) -> None:
    revenue = data["revenue"]
    roster = data["roster"]
    class_master = data["class_master"]
    qianyue = data["qianyue"]
    tmk = data["tmk"]

    amt = pd.to_numeric(revenue["现金收入"], errors="coerce")
    net = amt.sum()
    gross = amt[amt > 0].sum()
    students = roster["学员编号"].nunique()
    classes = (class_master["班级状态"] == "正常").sum()
    signed = qianyue["签约单id"].notna().sum()
    pending = (tmk["资源状态"] == "待处理").sum()

    render_kpi_row(
        [
            {
                "value": f"{net / 1e4:.0f} 万",
                "label": "现金收入 (净额)",
                "accent": "green",
                "sub": f"流水 {gross / 1e4:.0f} 万  |  退费 {abs(amt[amt < 0].sum()) / 1e4:.0f} 万",
                "formula": "SUM(收入人次.现金收入)",
            },
            {
                "value": str(students),
                "label": "总学员",
                "accent": "blue",
                "formula": "NUNIQUE(花名册.学员编号)",
            },
            {
                "value": str(classes),
                "label": "行课班级",
                "accent": "blue",
                "formula": "COUNT(维表 WHERE 班级状态 = 正常)",
            },
            {
                "value": str(signed),
                "label": "累计签约",
                "accent": "slate",
                "sub": f"待处理资源 {pending}",
                "formula": "COUNT(签约列表.签约单id IS NOT NULL)",
            },
        ]
    )


def render() -> None:
    inject_hk_dashboard_styles()

    c_title, c_back = st.columns([6, 1])
    with c_title:
        st.title("香港运营数据看板")
    with c_back:
        st.page_link("main.py", label="← 返回首页")

    with st.spinner("加载数据中..."):
        data = load_all_data()

    missing = [k for k, v in data.items() if v.empty]
    if missing:
        st.error(f"以下数据表加载失败: {', '.join(missing)}")
        st.stop()

    _summary_ribbon(data)

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "综合概览",
            "营收分析",
            "招生转化",
            "教务教学",
            "续费看板",
        ]
    )

    from src.pages.hk_dashboard.tabs.academics import render as tab_academics
    from src.pages.hk_dashboard.tabs.enrollment import render as tab_enrollment
    from src.pages.hk_dashboard.tabs.overview import render as tab_overview
    from src.pages.hk_dashboard.tabs.renewal import render as tab_renewal
    from src.pages.hk_dashboard.tabs.revenue import render as tab_revenue

    with t1:
        tab_overview(data)
    with t2:
        tab_revenue(data)
    with t3:
        tab_enrollment(data)
    with t4:
        tab_academics(data)
    with t5:
        tab_renewal(data)
