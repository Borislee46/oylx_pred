"""HK Dashboard — main render entry point."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.styles import inject_hk_dashboard_styles
from src.pages.hk_dashboard.data_loader import load_all_data
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row


def _summary_ribbon(data: dict) -> None:
    """Top-level KPI ribbon shown before tabs."""
    revenue = data["revenue"]
    roster = data["roster"]
    class_master = data["class_master"]
    qianyue = data["qianyue"]

    cash = pd.to_numeric(revenue["现金收入"], errors="coerce").sum()
    students = roster["学员编号"].nunique()
    classes = (class_master["班级状态"] == "正常").sum()
    signed = qianyue["签约单id"].notna().sum()

    render_kpi_row([
        {"value": f"HK$ {cash / 1e4:.0f}万", "label": "累计现金收入", "accent": "positive"},
        {"value": str(students), "label": "总学员数", "accent": "accent"},
        {"value": str(classes), "label": "行课班级", "accent": "accent"},
        {"value": str(signed), "label": "签约数", "accent": "accent"},
    ])


def render() -> None:
    inject_hk_dashboard_styles()

    st.title("香港运营数据看板")

    with st.spinner("加载数据中..."):
        data = load_all_data()

    missing = [k for k, v in data.items() if v.empty]
    if missing:
        st.error(f"以下数据表加载失败: {', '.join(missing)}")
        st.stop()

    _summary_ribbon(data)

    t1, t2, t3, t4, t5 = st.tabs([
        "综合概览", "营收分析", "招生转化", "教务教学", "续费看板",
    ])

    from src.pages.hk_dashboard.tabs.overview import render as tab_overview
    from src.pages.hk_dashboard.tabs.revenue import render as tab_revenue
    from src.pages.hk_dashboard.tabs.enrollment import render as tab_enrollment
    from src.pages.hk_dashboard.tabs.academics import render as tab_academics
    from src.pages.hk_dashboard.tabs.renewal import render as tab_renewal

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
