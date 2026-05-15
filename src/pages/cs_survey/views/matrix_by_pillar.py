from __future__ import annotations

import pandas as pd
import streamlit as st

from ..engine.aggregations import (
    build_product_detail,
    build_product_matrix,
    source_score_distribution,
)
from ..engine.kpis import build_compare_lines, kpi_source
from ..engine.themes import (
    annotate_feedback_themes,
    build_feedback_detail_rows,
    build_theme_rows_from_feedback,
)
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
from .detail_shared import (
    filter_theme_feedback,
    render_detail_hero,
    render_evidence_filters,
    render_theme_bar,
)


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
    feedback_spec = cfg.feedback_detail.get(source_id, [])

    df_matrix = build_product_matrix(df_main, product_cfg) if product_cfg else pd.DataFrame()
    df_detail = build_product_detail(df_main, product_cfg) if product_cfg else pd.DataFrame()
    dist = source_score_distribution(df_main, cfg, source_id)
    df_compare = build_compare_lines(scoped, cfg)
    kpi = kpi_source(df_main, cfg, source_id)

    df_feedback_detail = pd.DataFrame(
        build_feedback_detail_rows(df_main, src.dim1_short or src.label, feedback_spec),
        columns=["条线", "三大类", "子类", "分数", "反馈类型", "反馈"],
    )
    df_feedback_detail = annotate_feedback_themes(
        df_feedback_detail, cfg.themes["opinion"], cfg.themes["praise"]
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
    with panel("切换分析路径", ""):
        render_view_nav(cfg, current_view="by_pillar", key_prefix=f"detail_nav_{cfg.id}")

    tab_main, tab_evidence = st.tabs(["核心洞察", "反馈证据"])

    with tab_main:
        top_left, top_right = st.columns(
            layout_get(layout, "main_row_columns", default=[1.04, 0.96])
        )
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

        mid_left, mid_right = st.columns(
            layout_get(layout, "middle_row_columns", default=[1.02, 0.98])
        )
        with mid_left:
            with panel(
                "各产品意见与表扬对比", "完整展示全部产品类型，便于直接横向比较每一类反馈强度。"
            ):
                render_grouped_opinion_praise(df_matrix, detail_h, "p2_group_op", top_n=None)
        with mid_right:
            with panel(
                "产品明细评分对比", "按评分从高到低展示全部产品明细，方便同时识别领先项与低分项。"
            ):
                render_ranked_detail_scores(df_detail, detail_h, "p2_detail", top_n=None)

        lower_left, lower_right = st.columns(
            layout_get(layout, "lower_row_columns", default=[0.92, 1.08])
        )
        with lower_left:
            with panel("业务线对比摘要", "对比表已与当前三大块筛选保持一致，方便横向验证判断。"):
                render_table(df_compare, detail_h)
        with lower_right:
            with panel("反馈主题", "按维度与产品分类筛选后，查看意见与表扬的主题分布。"):
                theme_feedback_view, _, _ = filter_theme_feedback(
                    df_feedback_detail,
                    state_prefix="cs_survey_by_pillar",
                    label="维度",
                    fixed_group=pillar_name if pillar_name != "全部" else None,
                )
                df_theme = pd.DataFrame(
                    build_theme_rows_from_feedback(
                        theme_feedback_view, cfg.themes["opinion"], mode="opinion"
                    ),
                    columns=["反馈类型", "反馈量"],
                )
                df_praise = pd.DataFrame(
                    build_theme_rows_from_feedback(
                        theme_feedback_view, cfg.themes["praise"], mode="praise"
                    ),
                    columns=["反馈类型", "反馈量"],
                )
                theme_op_tab, theme_pr_tab = st.tabs(["意见主题", "表扬主题"])
                with theme_op_tab:
                    render_theme_bar(df_theme, "#93C5FD", theme_h, "p2_theme_op", top_n=None)
                with theme_pr_tab:
                    render_theme_bar(df_praise, "#F2C811", theme_h, "p2_theme_pr", top_n=None)

    with tab_evidence:
        render_section_header(
            "Evidence Link",
            "产品类型与原始反馈联动",
            "支持按产品类型、反馈类型和关键词筛选，右侧原文会跟随当前定位实时收窄。",
        )
        df_detail_view, df_feedback_view = render_evidence_filters(
            df_detail,
            df_feedback_detail,
            state_prefix="cs_survey_by_pillar",
            group_label="产品分组",
            detail_label="产品类型",
        )
        ev_left, ev_right = st.columns(
            layout_get(layout, "evidence_row_columns", default=[0.84, 1.16])
        )
        with ev_left:
            with panel(
                "产品类型明细数据", "左侧表会跟随筛选同步更新，适合先锁定要复盘的产品类型。"
            ):
                render_table(df_detail_view, detail_h)
        with ev_right:
            with panel(
                "原始反馈文本", "右侧自动展示与当前产品定位相关的原始反馈，便于直接回看样本。"
            ):
                render_table(df_feedback_view, text_h)
