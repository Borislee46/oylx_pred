from __future__ import annotations

import pandas as pd
import streamlit as st

from ..engine.aggregations import (
    build_detail_matrix,
    build_pillar_matrix,
    score_distribution,
)
from ..engine.kpis import build_compare_lines, kpi_source
from ..engine.themes import build_feedback_detail_rows, build_theme_rows, theme_pairs_for_source
from ..filters import filter_by_product, filter_center, list_center_values
from ..layout import get_page_layout, layout_get
from ..loader import load_all_sources
from ..schema import SurveyConfig
from ..ui.blocks import panel, render_section_header
from ..ui.charts import (
    render_area_detail,
    render_bar_dist,
    render_three_pillar_bars,
)
from ..ui.filter_bar import filter_bar
from ..ui.navigation import render_view_nav
from ..ui.tables import render_table
from ..ui.theme_css import format_metric, inject_detail_css
from .detail_shared import render_detail_hero, render_detail_kpis, render_theme_bar


def _source_selector(cfg: SurveyConfig) -> str:
    ids = [s.id for s in cfg.sources]
    labels = {s.id: s.label for s in cfg.sources}
    default = st.session_state.get("cs_survey_by_product_source", ids[0])
    return st.selectbox(
        "业务条线",
        ids,
        index=ids.index(default) if default in ids else 0,
        format_func=lambda i: labels[i],
        key="cs_survey_by_product_source",
    )


def render(cfg: SurveyConfig) -> None:
    inject_detail_css()
    layout = get_page_layout("by_product")
    heights = layout_get(layout, "heights", default={})
    summary_h = int(layout_get(heights, "summary", default=280))
    detail_h = int(layout_get(heights, "detail", default=340))
    theme_h = int(layout_get(heights, "theme", default=340))
    text_h = int(layout_get(heights, "text", default=360))

    dfs = load_all_sources(cfg)
    center_spec = cfg.cross_filters.get("center")

    with filter_bar("Analysis Controls · 分析范围"):
        cols = st.columns(layout_get(layout, "filter_columns", default=[1, 1, 1]))
        with cols[0]:
            source_id = _source_selector(cfg)
        src = cfg.source(source_id)
        product_cfg = cfg.products.get(source_id)
        with cols[1]:
            labels = ["全部"] + (product_cfg.labels if product_cfg else [])
            product_label = st.selectbox(
                "产品类型", labels, key=f"cs_survey_by_product_prod_{source_id}"
            )
        with cols[2]:
            if center_spec is not None:
                center_values = list_center_values(dfs, center_spec)
                center = st.selectbox(
                    center_spec.label or "分中心",
                    center_values,
                    key="cs_survey_by_product_center",
                )
            else:
                center = None

    filtered: dict[str, pd.DataFrame] = {}
    for s in cfg.sources:
        df = dfs[s.id]
        if center_spec is not None and center is not None:
            df = filter_center(df, center_spec, center)
        filtered[s.id] = df

    if product_cfg is not None:
        filtered[source_id] = filter_by_product(filtered[source_id], product_cfg, product_label)

    df_main = filtered[source_id]
    details = cfg.details.get(source_id, [])
    dist_cols = cfg.score_distribution_cols.get(source_id, [])
    feedback_spec = cfg.feedback_detail.get(source_id, [])
    reverse_cols = (
        set(cfg.score_cols.get(source_id).reverse) if cfg.score_cols.get(source_id) else set()
    )

    df_matrix = build_pillar_matrix(df_main, cfg.pillars, source_id)
    df_detail = build_detail_matrix(df_main, details)
    dist = score_distribution(df_main, dist_cols, reverse_cols)
    df_compare = build_compare_lines(filtered, cfg)
    kpi = kpi_source(df_main, cfg, source_id)

    theme_pairs = theme_pairs_for_source(cfg, source_id)
    df_theme = pd.DataFrame(
        build_theme_rows(df_main, theme_pairs, cfg.themes["opinion"], mode="opinion"),
        columns=["反馈类型", "反馈量"],
    )
    df_praise = pd.DataFrame(
        build_theme_rows(df_main, theme_pairs, cfg.themes["praise"], mode="praise"),
        columns=["反馈类型", "反馈量"],
    )
    df_feedback_detail = pd.DataFrame(
        build_feedback_detail_rows(df_main, src.dim1_short or src.label, feedback_spec),
        columns=["条线", "三大类", "子类", "分数", "反馈类型", "反馈"],
    )

    chips = [
        f"业务条线: {src.label}",
        f"产品类型: {product_label}",
    ]
    if center_spec is not None and center:
        chips.append(f"{center_spec.label or '分中心'}: {center}")
    render_detail_hero(
        cfg.title,
        "按产品类型查看评分表现、反馈结构与业务线差异。",
        badge=f"当前切片评分 {format_metric(kpi.get('评分均值'))}",
        chips=chips,
        summary=(
            f"当前视角聚焦于 {src.label} 条线下的 {product_label} 范围。"
            f"样本内共收集 {int(kpi.get('评分量', 0)):,} 个评分点，"
            f"意见率为 {format_metric(kpi.get('意见率'), '{:.1f}%')}，"
            f"表扬率为 {format_metric(kpi.get('表扬率'), '{:.1f}%')}。"
        ),
        stats=[
            {
                "label": "评分均值",
                "value": format_metric(kpi.get("评分均值")),
                "caption": "当前切片的综合评分水平",
                "accent": True,
            },
            {
                "label": "评分量",
                "value": f"{int(kpi.get('评分量', 0)):,}",
                "caption": "本页分析所覆盖的评分样本量",
            },
            {
                "label": "意见量",
                "value": f"{int(kpi.get('意见量', 0)):,}",
                "caption": "当前切片中的意见反馈总量",
            },
            {
                "label": "表扬量",
                "value": f"{int(kpi.get('表扬量', 0)):,}",
                "caption": "当前切片中的表扬反馈总量",
            },
        ],
    )
    with panel("切换分析路径", "当你想换一个切片角度时，直接从这里进入其它分析视图。"):
        render_view_nav(cfg, current_view="by_product", key_prefix=f"detail_nav_{cfg.id}")

    render_section_header(
        "Current Slice",
        "先判断当前产品切片的整体状态",
        "先看综合表现和反馈结构，再继续读差异来源与具体明细。",
    )
    render_detail_kpis(
        kpi,
        scope_label=f"{src.label} / {product_label}",
        focus_label="产品类型",
        columns=int(layout_get(layout, "detail_kpi_columns", default=3)),
    )

    tab_main, tab_evidence = st.tabs(["核心洞察", "反馈证据"])

    with tab_main:
        top_left, top_right = st.columns(layout_get(layout, "main_row_columns", default=[1, 1]))
        with top_left:
            with panel(
                "产品大类核心指标", "当前业务条线下各产品大类的核心表现，用来先判断差异来自哪里。"
            ):
                render_table(df_matrix, summary_h)
        with top_right:
            with panel("评分量分布", "样本量分布帮助判断评分波动是否可靠。", badge="1-5 分"):
                render_bar_dist(dist, summary_h, "p1_dist")

        mid_left, mid_right = st.columns(layout_get(layout, "middle_row_columns", default=[1, 1]))
        with mid_left:
            with panel(
                "各维度意见与表扬", "按三大块对比意见量和表扬量，定位当前切片最敏感的服务环节。"
            ):
                render_three_pillar_bars(df_matrix, detail_h, "p1_pillar", cfg.pillar_names)
        with mid_right:
            with panel(
                "产品明细评分趋势", "评分均值从低到高展开，方便快速识别拉低体验的具体明细。"
            ):
                render_area_detail(df_detail, detail_h, "p1_detail")

        lower_left, lower_right = st.columns(
            layout_get(layout, "lower_row_columns", default=[1, 1])
        )
        with lower_left:
            with panel("业务线对比摘要", "保持当前筛选上下文，横向比较其它业务线的核心指标。"):
                render_table(df_compare, detail_h)
        with lower_right:
            with panel("反馈主题", "把负向和正向主题放在同一块里切换看，页面节奏会更统一。"):
                theme_op_tab, theme_pr_tab = st.tabs(["意见主题", "表扬主题"])
                with theme_op_tab:
                    render_theme_bar(df_theme, "#93C5FD", theme_h, "p1_theme_op")
                with theme_pr_tab:
                    render_theme_bar(df_praise, "#F2C811", theme_h, "p1_theme_pr")

    with tab_evidence:
        ev_left, ev_right = st.columns(layout_get(layout, "evidence_row_columns", default=[1, 1]))
        with ev_left:
            with panel("产品明细数据", "下钻到产品明细后的维度和评分表现，适合作为复盘依据。"):
                render_table(df_detail, detail_h)
        with ev_right:
            with panel("原始反馈文本", "保留原始反馈文本，便于从主题进一步回看具体样本。"):
                render_table(df_feedback_detail, text_h)
