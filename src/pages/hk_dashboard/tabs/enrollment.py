"""Tab 3: 招生转化 — two-pipeline view, consultant, channels."""

import pandas as pd
import streamlit as st

from src.pages.hk_dashboard.charts import donut_chart, grouped_bar, monthly_trend_line, simple_bar
from src.pages.hk_dashboard.components.data_table import render_filterable_table
from src.pages.hk_dashboard.components.kpi_cards import render_metric_grid
from src.pages.hk_dashboard.metrics.funnel_metrics import (
    channel_breakdown,
    channel_product_cross,
    class_capacity_metrics,
    consultant_detail,
    consultant_ranking,
    monthly_signing_trend,
    signing_by_product,
    tmk_processing_stats,
    work_order_follow_up_stats,
)


def _render_pipeline(kehu, tmk, qianyue) -> None:
    c1, c2 = st.columns(2)

    with c1:
        st.html("<h3>TMK 外呼管道</h3>")
        total_tmk = len(tmk)
        # Nested funnel: each stage is a subset of the previous
        called_mask = tmk["外呼次数"].notna()
        called = called_mask.sum()
        has_wo = (called_mask & tmk["工单id"].notna()).sum()
        processed = (called_mask & tmk["工单id"].notna() & (tmk["资源状态"] == "已处理")).sum()
        stages = [
            ("TMK 资源", total_tmk, "#2563eb"),
            ("已外呼", called, "#0d9488"),
            ("有工单", has_wo, "#d97706"),
            ("已处理", processed, "#7c3aed"),
        ]
        html = '<div style="padding:0.15rem 0">'
        for i, (label, val, color) in enumerate(stages):
            pct = val / (stages[0][1] or 1) * 100
            ratio = f"{(val / stages[i - 1][1] * 100):.0f}%" if i > 0 and stages[i - 1][1] else ""
            bar_pct = max(pct, 2)
            html += (
                f'<div style="margin-bottom:3px;display:flex;align-items:center">'
                f'<span style="width:52px;font-size:0.72rem;font-weight:600;color:#475569">{label}</span>'
                f'<span style="width:52px;font-size:0.76rem;font-weight:700;color:{color}">{val:,}</span>'
                f'<span style="width:34px;font-size:0.64rem;color:#94a3b8">{ratio}</span>'
                f'<span style="flex:1;margin-left:3px;height:14px;background:#f1f5f9;border-radius:2px;">'
                f'<span style="display:block;height:100%;background:{color};border-radius:2px;'
                f'width:{bar_pct:.0f}%;"></span></span>'
                f"</div>"
            )
        html += "</div>"
        st.html(html)
        st.html('<div class="hk-note">客服TMK资源处理 | 总资源 → 已外呼 → 有工单 → 已处理</div>')

    with c2:
        st.html("<h3>签约转化</h3>")
        kehu_ids = set(kehu["资源id"].dropna())
        signed_ids = set(qianyue["资源id"].dropna())
        matched = len(kehu_ids & signed_ids)
        total_signed = qianyue["签约单id"].notna().sum()

        items = [
            ("总资源", len(kehu), "#2563eb"),
            ("匹配签约", matched, "#0d9488"),
            ("签单数", total_signed, "#7c3aed"),
        ]
        html2 = '<div style="padding:0.15rem 0">'
        for label, val, color in items:
            pct = val / (items[0][1] or 1) * 100
            bar_pct = max(min(pct, 100), 3)
            html2 += (
                f'<div style="margin-bottom:3px;display:flex;align-items:center">'
                f'<span style="width:58px;font-size:0.72rem;font-weight:600;color:#475569">{label}</span>'
                f'<span style="width:52px;font-size:0.76rem;font-weight:700;color:{color}">{val:,}</span>'
                f'<span style="flex:1;margin-left:3px;height:14px;background:#f1f5f9;border-radius:2px;">'
                f'<span style="display:block;height:100%;background:{color};border-radius:2px;'
                f'width:{bar_pct:.0f}%;"></span></span>'
                f"</div>"
            )
        html2 += "</div>"
        st.html(html2)
        matched_pct = f"{matched / len(signed_ids) * 100:.0f}%" if signed_ids else "-"
        st.caption(
            f"{total_signed} 单签约 | {matched}/{len(signed_ids)} ({matched_pct}) 资源 ID 可匹配到资源池"
        )
        st.html('<div class="hk-note">客服资源.资源id ∩ 签约列表.资源id | 两表通过资源ID关联</div>')

    st.html("<h3>TMK 处理统计</h3>")
    tstats = tmk_processing_stats(tmk)
    render_metric_grid(
        [
            {"label": "已外呼", "value": str(tstats["已外呼"])},
            {"label": "已处理", "value": str(tstats["已处理"])},
            {"label": "待处理", "value": str(tstats["待处理"])},
            {"label": "1 天内处理", "value": str(tstats["1天内处理"])},
            {"label": "平均处理时延", "value": f'{tstats["平均处理时延(天)"]} 天'},
        ],
        columns=5,
    )
    st.html(
        '<div class="hk-note">TMK | 分配时间 → 顾问处理时间 = 处理时延 | COUNT(外呼次数 IS NOT NULL) = 已外呼</div>'
    )


def render(data: dict[str, pd.DataFrame]) -> None:
    kehu = data["kehu_ziyuan"]
    tmk = data["tmk"]
    qianyue = data["qianyue"]
    class_master = data["class_master"]

    st.html("<h2>班级容量</h2>")
    cap = class_capacity_metrics(class_master)
    render_metric_grid(
        [
            {"label": "行课班级", "value": str(cap["行课班级数"])},
            {"label": "满班班级", "value": str(cap["满班班级数"])},
            {"label": "满班率", "value": cap["满班率"]},
            {"label": "平均满班率", "value": cap["平均满班率"]},
        ],
        columns=4,
    )
    st.html(
        '<div class="hk-note">维表 | 正常 + 当前人数>0 | 满班率 = 满班数 / 行课班级数 | 满班 = 当前人数 >= 标准人数</div>'
    )

    with st.expander("班级容量明细"):
        cm = class_master.copy()
        cm["当前_n"] = pd.to_numeric(cm["当前人数"], errors="coerce")
        cm["标准_n"] = pd.to_numeric(cm["标准人数"], errors="coerce")
        cm["满班率"] = (cm["当前_n"] / cm["标准_n"].replace(0, pd.NA) * 100).round(0).astype(
            str
        ) + "%"
        tbl = cm[
            [
                "班级编码",
                "班级名称",
                "班级状态",
                "当前人数",
                "标准人数",
                "最大人数",
                "满班率",
                "开课日期",
            ]
        ]
        render_filterable_table(tbl[tbl["班级状态"] == "正常"].head(100), key="capacity_detail")

    st.html("<h2>资源管道</h2>")
    _render_pipeline(kehu, tmk, qianyue)

    st.html("<h2>渠道分析</h2>")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>资源渠道</h3>")
            ch = channel_breakdown(kehu)
            if not ch.empty:
                donut_chart(ch, "渠道", "count", max_categories=6, height=180)
    with c2:
        with st.container(border=True):
            st.html("<h3>签约渠道</h3>")
            if "二级获取渠道" in qianyue.columns:
                ch2 = qianyue["二级获取渠道"].value_counts().reset_index(name="count")
                ch2.columns = ["渠道", "count"]
                if not ch2.empty:
                    simple_bar(
                        ch2.head(10), "渠道", "count", horizontal=True, color="#0d9488", height=180
                    )

    st.html("<h2>工单跟进</h2>")
    wo_stats = work_order_follow_up_stats(tmk, qianyue)
    render_metric_grid(
        [
            {"label": "工单总数", "value": str(wo_stats["工单总数"])},
            {"label": "正常", "value": str(wo_stats["正常"])},
            {"label": "已删除", "value": str(wo_stats["已删除"])},
            {"label": "死单标记", "value": str(wo_stats["死单标记"])},
            {"label": "有顾问跟进签约", "value": str(wo_stats["有顾问跟进签约"])},
        ],
        columns=5,
    )
    st.html('<div class="hk-note">TMK.工单状态 + 签约列表.工单跟进顾问 | 工单管道统计</div>')

    st.html("<h2>顾问业绩</h2>")
    ranking = consultant_ranking(qianyue)
    c1, c2 = st.columns([2, 3])
    with c1:
        if not ranking.empty:
            simple_bar(ranking.head(15), "顾问姓名", "签约数", horizontal=True, height=240)
    with c2:
        render_filterable_table(ranking, key="consultant_ranking")

    st.html("<h3>顾问详情（含资源签约率）</h3>")
    cdetail = consultant_detail(kehu, qianyue)
    if not cdetail.empty:
        render_filterable_table(cdetail, key="consultant_detail")
    st.html(
        '<div class="hk-note">签约列表 JOIN 客服资源 ON 资源分配顾问 | 资源签约率 = 签约使用资源数 / 分配资源数</div>'
    )

    # ── Signing trend & product ──
    st.divider()
    st.html("<h2>签约趋势与产品</h2>")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>月度签约趋势</h3>")
            sign_trend = monthly_signing_trend(qianyue)
            if not sign_trend.empty:
                monthly_trend_line(sign_trend, "月份", "签约数", color="#0d9488", height=180)
            st.html(
                '<div class="hk-note">签约列表 | GROUP BY MONTH(签约时间) | COUNT(签约单id)</div>'
            )
    with c2:
        with st.container(border=True):
            st.html("<h3>签约金额趋势</h3>")
            if not sign_trend.empty:
                monthly_trend_line(
                    sign_trend, "月份", "签约金额", color="#d97706", currency=True, height=180
                )
            st.html('<div class="hk-note">签约列表 | GROUP BY MONTH(签约时间) | SUM(学费)</div>')

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.html("<h3>签约产品分布</h3>")
            by_prod = signing_by_product(qianyue)
            if not by_prod.empty:
                simple_bar(
                    by_prod, "课程产品名称", "签约数", horizontal=True, color="#0d9488", height=180
                )
            st.html('<div class="hk-note">签约列表 | GROUP BY 课程产品名称 | COUNT(签约单id)</div>')
    with c2:
        with st.container(border=True):
            st.html("<h3>渠道 × 产品</h3>")
            cross = channel_product_cross(qianyue)
            if not cross.empty:
                grouped_bar(cross.head(20), "一级获取渠道", "签约数", "课程产品名称", height=180)
            st.html('<div class="hk-note">签约列表 | GROUP BY 一级获取渠道, 课程产品名称</div>')
