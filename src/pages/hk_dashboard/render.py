"""HK Dashboard — main render entry point."""

import streamlit as st

from src.pages.hk_dashboard.styles import inject_hk_dashboard_styles
from src.pages.hk_dashboard.data_loader import load_all_data


def render() -> None:
    inject_hk_dashboard_styles()

    st.title("香港运营数据看板")

    with st.spinner("正在加载数据..."):
        data = load_all_data()

    # Check data health
    missing = [k for k, v in data.items() if v.empty]
    if missing:
        st.error(f"以下数据表加载失败: {', '.join(missing)}")
        st.stop()

    # ── Tabs ──
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 综合概览",
        "💰 营收分析",
        "📈 招生转化",
        "🎓 教务教学",
        "🔄 续费看板",
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
