"""Tab 3: 招生转化 — two-pipeline view, consultant, channels."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import simple_bar, donut_chart
from src.pages.hk_dashboard.components.kpi_cards import render_kpi_row, render_metric_grid
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.metrics.funnel_metrics import (
    calculate_funnel, consultant_ranking, class_capacity_metrics,
    channel_breakdown, tmk_processing_stats,
)


def _render_pipeline(kehu, tmk, qianyue) -> None:
    """Two-pipeline view: TMK outbound + overall conversion."""
    c1, c2 = st.columns(2)

    # ── TMK pipeline ──
    with c1:
        st.html("<h3>TMK 外呼管道</h3>")
        total_tmk = len(tmk)
        called = tmk["外呼次数"].notna().sum()
        work_orders = tmk["工单ID"].notna().sum()
        normal_wo = (tmk["工单状态"] == "正常").sum()

        items = [
            ("TMK 资源", total_tmk, "#2563eb"),
            ("已外呼", called, "#0d9488"),
            ("有工单", work_orders, "#d97706"),
            ("工单正常", normal_wo, "#7c3aed"),
        ]
        max_w = total_tmk or 1
        html = '<div style="padding:0.25rem 0">'
        for label, val, color in items:
            pct = val / max_w * 100
            prev = items[0][1]
            ratio = f"{(val / (items[items.index((label, val, color)) - 1][1]) * 100):.0f}%" if items.index((label, val, color)) > 0 else ""
            html += (
                f'<div style="margin-bottom:4px;display:flex;align-items:center">'
                f'<span style="width:55px;font-size:0.76rem;font-weight:600;color:#475569">{label}</span>'
                f'<span style="width:55px;font-size:0.8rem;font-weight:700;color:{color}">{val:,}</span>'
                f'<span style="width:35px;font-size:0.68rem;color:#94a3b8">{ratio}</span>'
                f'<span style="flex:1;margin-left:4px;background:{color};height:16px;'
                f'border-radius:2px;width:{max(pct * 0.45, 1)}%;min-width:{max(pct * 0.45, 1)}%"></span>'
                f'</div>'
            )
        html += "</div>"
        st.html(html)

    # ── Contract pipeline ──
    with c2:
        st.html("<h3>签约转化</h3>")
        kehu_ids = set(kehu["资源id"].dropna())
        signed_ids = set(qianyue["资源id"].dropna())
        matched = len(kehu_ids & signed_ids)

        contract_items = [
            ("总资源", len(kehu), "#2563eb"),
            ("有资源ID匹配", matched, "#0d9488"),
            ("签单数", qianyue["签约单id"].notna().sum(), "#7c3aed"),
        ]
        max_w2 = len(kehu) or 1
        html2 = '<div style="padding:0.25rem 0">'
        for label, val, color in contract_items:
            pct = val / max_w2 * 100
            html2 += (
                f'<div style="margin-bottom:4px;display:flex;align-items:center">'
                f'<span style="width:70px;font-size:0.76rem;font-weight:600;color:#475569">{label}</span>'
                f'<span style="width:55px;font-size:0.8rem;font-weight:700;color:{color}">{val:,}</span>'
                f'<span style="flex:1;margin-left:4px;background:{color};height:16px;'
                f'border-radius:2px;width:{max(pct * 0.45, 1)}%;min-width:{max(pct * 0.45, 1)}%"></span>'
                f'</div>'
            )
        html2 += "</div>"
        st.html(html2)
        matched_pct = f"{matched / len(signed_ids) * 100:.0f}%" if signed_ids else "-"
        st.caption(f"{qianyue['签约单id'].notna().sum()} 单签约，其中 {matched}/{len(signed_ids)} ({matched_pct}) 个资源 ID 匹配资源池")

    # TMK stats below
    st.html("<h3>TMK 处理统计</h3>")
    tstats = tmk_processing_stats(tmk)
    render_metric_grid([
        {"label": "已外呼", "value": str(tstats["已外呼"])},
        {"label": "已处理", "value": str(tstats["已处理"])},
        {"label": "待处理", "value": str(tstats["待处理"])},
        {"label": "1 天内处理", "value": str(tstats["1天内处理"])},
        {"label": "平均处理时延", "value": f'{tstats["平均处理时延(天)"]} 天'},
    ], columns=5)


def render(data: dict[str, pd.DataFrame]) -> None:
    kehu = data["kehu_ziyuan"]
    tmk = data["tmk"]
    qianyue = data["qianyue"]
    class_master = data["class_master"]

    # ── Capacity ──
    st.html("<h2>班级容量</h2>")
    cap = class_capacity_metrics(class_master)
    render_metric_grid([
        {"label": "行课班级", "value": str(cap["行课班级数"])},
        {"label": "满班班级", "value": str(cap["满班班级数"])},
        {"label": "满班率", "value": cap["满班率"]},
        {"label": "平均满班率", "value": cap["平均满班率"]},
    ], columns=4)

    with st.expander("班级容量明细"):
        cm = class_master.copy()
        cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
        cm["标准_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
        cm["满班率"] = (cm["当前_n"] / cm["标准_n"].replace(0, pd.NA) * 100).round(0).astype(str) + "%"
        tbl = cm[["班级编码", "班级名称", "班级状态", "当前人数", "标准人数", "最大人数", "满班率", "开课日期"]]
        render_filterable_table(tbl[tbl["班级状态"] == "正常"].head(100), key="capacity_detail")

    # ── Pipeline (two columns: TMK + Contract) ──
    st.html("<h2>资源管道</h2>")
    _render_pipeline(kehu, tmk, qianyue)

    # ── Channels ──
    st.html("<h2>渠道分析</h2>")
    c1, c2 = st.columns(2)
    with c1:
        st.html("<h3>资源渠道</h3>")
        ch = channel_breakdown(kehu)
        if not ch.empty:
            donut_chart(ch, "渠道", "count", max_categories=6)
    with c2:
        st.html("<h3>签约渠道</h3>")
        if "二级获取渠道" in qianyue.columns:
            ch2 = qianyue["二级获取渠道"].value_counts().reset_index(name="count")
            ch2.columns = ["渠道", "count"]
            if not ch2.empty:
                simple_bar(ch2.head(10), "渠道", "count", horizontal=True, color="#0d9488")

    # ── Consultant ──
    st.html("<h2>顾问业绩</h2>")
    ranking = consultant_ranking(qianyue)
    c1, c2 = st.columns([2, 3])
    with c1:
        if not ranking.empty:
            simple_bar(ranking.head(15), "顾问姓名", "签约数", horizontal=True)
    with c2:
        render_filterable_table(ranking, key="consultant_ranking")
