from __future__ import annotations

import pandas as pd
import streamlit as st

from ..engine.aggregations import (
    build_product_detail,
    build_product_matrix,
    score_distribution,
)
from ..engine.kpis import build_compare_lines, kpi_source
from ..engine.themes import build_feedback_detail_rows, build_theme_rows, theme_pairs_for_source
from ..filters import filter_by_pillar, filter_center, find_pillar, list_center_values
from ..layout import get_page_layout, layout_get
from ..loader import load_all_sources
from ..schema import SurveyConfig
from ..ui.blocks import panel, render_section_header
from ..ui.charts import (
    render_bar_dist,
    render_grouped_opinion_praise,
    render_ranked_detail_scores,
)
from ..ui.filter_bar import filter_bar
from ..ui.navigation import render_view_nav
from ..ui.tables import render_table
from ..ui.theme_css import format_metric, inject_detail_css
from .detail_shared import render_detail_hero, render_detail_kpis, render_theme_bar


def _source_selector(cfg: SurveyConfig) -> str:
    ids = [s.id for s in cfg.sources]
    labels = {s.id: s.label for s in cfg.sources}
    default = st.session_state.get("cs_survey_by_pillar_source", ids[0])
    return st.selectbox(
        "业务条线",
        ids,
        index=ids.index(default) if default in ids else 0,
        format_func=lambda i: labels[i],
        key="cs_survey_by_pillar_source",
    )


def render(cfg: SurveyConfig) -> None:
    inject_detail_css()
    layout = get_page_layout("by_pillar")
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
        pillar_options = ["全部"] + cfg.pillar_names
        with cols[1]:
            pillar_name = st.selectbox("三大块", pillar_options, key="cs_survey_by_pillar_pillar")
        with cols[2]:
            if center_spec is not None:
                center_values = list_center_values(dfs, center_spec)
                center = st.selectbox(
                    center_spec.label or "分中心",
                    center_values,
                    key="cs_survey_by_pillar_center",
                )
            else:
                center = None

    filtered: dict[str, pd.DataFrame] = {}
    for s in cfg.sources:
        df = dfs[s.id]
        if center_spec is not None and center is not None:
            df = filter_center(df, center_spec, center)
        filtered[s.id] = df

    selected_pillar = find_pillar(cfg, pillar_name) if pillar_name != "全部" else None
    scoped = {s.id: filter_by_pillar(filtered[s.id], selected_pillar, s.id) for s in cfg.sources}

    df_main = scoped[source_id]
    product_cfg = cfg.products.get(source_id)
    dist_cols = cfg.score_distribution_cols.get(source_id, [])
    feedback_spec = cfg.feedback_detail.get(source_id, [])
    reverse_cols = (
        set(cfg.score_cols.get(source_id).reverse) if cfg.score_cols.get(source_id) else set()
    )

    df_matrix = build_product_matrix(df_main, product_cfg) if product_cfg else pd.DataFrame()
    df_detail = build_product_detail(df_main, product_cfg) if product_cfg else pd.DataFrame()
    dist = score_distribution(df_main, dist_cols, reverse_cols)
    df_compare = build_compare_lines(scoped, cfg)
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
        f"三大块: {pillar_name}",
    ]
    if center_spec is not None and center:
        chips.append(f"{center_spec.label or '分中心'}: {center}")
    render_detail_hero(
        cfg.title,
        "按三大块筛选后查看产品表现、业务线对比与反馈主题。",
        badge=f"当前切片评分 {format_metric(kpi.get('评分均值'))}",
        chips=chips,
        summary=(
            f"当前视角聚焦于 {src.label} 条线下的 {pillar_name} 范围。"
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
        render_view_nav(cfg, current_view="by_pillar", key_prefix=f"detail_nav_{cfg.id}")

    render_section_header(
        "Current Slice",
        "先判断当前维度切片的整体状态",
        "这一页重点回答某个三大块下，产品表现与反馈结构的差异来自哪里。",
    )
    render_detail_kpis(
        kpi,
        scope_label=f"{src.label} / {pillar_name}",
        focus_label="三大块",
        columns=int(layout_get(layout, "detail_kpi_columns", default=3)),
    )

    tab_main, tab_evidence = st.tabs(["核心洞察", "反馈证据"])

    with tab_main:
        top_left, top_right = st.columns(layout_get(layout, "main_row_columns", default=[1, 1]))
        with top_left:
            with panel(
                "产品类型核心指标", "在当前三大块下，各产品类型的核心表现会先告诉你差异集中在哪里。"
            ):
                render_table(df_matrix, summary_h)
        with top_right:
            with panel(
                "评分量分布",
                "当前筛选范围内评分样本的分布情况，用于判断结论可靠度。",
                badge="1-5 分",
            ):
                render_bar_dist(dist, summary_h, "p2_dist")

        mid_left, mid_right = st.columns(layout_get(layout, "middle_row_columns", default=[1, 1]))
        with mid_left:
            with panel(
                "各产品意见与表扬对比", "只展示反馈量更高的产品类型，帮助快速识别高波动区域。"
            ):
                render_grouped_opinion_praise(df_matrix, detail_h, "p2_group_op", top_n=8)
        with mid_right:
            with panel("产品明细评分对比", "仅展示评分最高的 Top 10 明细，便于观察优秀样本分布。"):
                render_ranked_detail_scores(df_detail, detail_h, "p2_detail", top_n=10)

        lower_left, lower_right = st.columns(
            layout_get(layout, "lower_row_columns", default=[1, 1])
        )
        with lower_left:
            with panel("业务线对比摘要", "对比表已与当前三大块筛选保持一致，方便横向验证判断。"):
                render_table(df_compare, detail_h)
        with lower_right:
            with panel("反馈主题", "把负向和正向主题放在同一块里切换看，页面节奏会更统一。"):
                theme_op_tab, theme_pr_tab = st.tabs(["意见主题", "表扬主题"])
                with theme_op_tab:
                    render_theme_bar(df_theme, "#93C5FD", theme_h, "p2_theme_op")
                with theme_pr_tab:
                    render_theme_bar(df_praise, "#F2C811", theme_h, "p2_theme_pr")

    with tab_evidence:
        ev_left, ev_right = st.columns(layout_get(layout, "evidence_row_columns", default=[1, 1]))
        with ev_left:
            with panel(
                "产品类型明细数据", "按产品明细查看分数和反馈表现，适合作为复盘与复核依据。"
            ):
                render_table(df_detail, detail_h)
        with ev_right:
            with panel("原始反馈文本", "保留原始反馈，便于回看具体样本并验证主题判断。"):
                render_table(df_feedback_detail, text_h)
