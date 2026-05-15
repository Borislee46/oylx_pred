from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ..engine.aggregations import build_multi_select_counts, build_pillar_matrix
from ..engine.kpis import overview_kpis
from ..layout import get_page_layout, layout_get
from ..loader import load_all_sources
from ..schema import SurveyConfig
from ..ui.blocks import (
    panel,
    render_html_block,
    render_page_hero,
    render_section_header,
)
from ..ui.charts import render_multi_select_bars
from ..ui.navigation import render_view_nav
from ..ui.tables import render_table
from ..ui.theme_css import format_metric, inject_overview_css


def _resolved_updated_at(cfg: SurveyConfig, dfs: dict[str, pd.DataFrame]) -> str:
    latest = pd.NaT
    for df in dfs.values():
        for col in ("finish", "start"):
            if col not in df.columns:
                continue
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().any():
                cur = parsed.max()
                if pd.notna(cur) and (pd.isna(latest) or cur > latest):
                    latest = cur
    if pd.notna(latest):
        return latest.strftime("%Y-%m-%d")
    return cfg.updated_at


def render(cfg: SurveyConfig) -> None:
    inject_overview_css()
    layout = get_page_layout("overview")
    dfs = load_all_sources(cfg)
    k = overview_kpis(dfs, cfg)
    compare = k["compare"]
    logo_path = str(Path(__file__).resolve().parents[4] / "assets" / "company_logo.png")
    updated_at = _resolved_updated_at(cfg, dfs)

    # ── Hero ──
    render_page_hero(
        cfg.title,
        cfg.subtitle,
        badge=f"更新于 {updated_at}",
        eyebrow="Survey Overview",
        summary=(
            f"本次调研已汇总 {k['n_total']:,} 份样本、{k['score_total']:,} 个评分点。"
            f"当前综合评分为 {format_metric(k['mean_pooled'])}，"
            f"意见率 {format_metric(k['opinion_rate'], '{:.1f}%')}，"
            f"表扬率 {format_metric(k['praise_rate'], '{:.1f}%')}。"
        ),
        chips=[
            f"覆盖业务线 {len(cfg.sources)} 条",
            f"核心维度 {len(cfg.pillars)} 个",
            f"人均题数 {format_metric(k['avg_scores_per_resp'], '{:.1f}')}",
        ],
        stats=[
            {
                "label": "综合评分",
                "value": format_metric(k["mean_pooled"]),
                "caption": "加权汇总后的整体体验分",
                "accent": True,
            },
            {
                "label": "调研总量",
                "value": f"{k['n_total']:,}",
                "caption": "当前纳入总览的有效样本量",
            },
            {
                "label": "意见率",
                "value": format_metric(k["opinion_rate"], "{:.1f}%"),
                "caption": "反馈中带有意见内容的占比",
            },
            {
                "label": "表扬率",
                "value": format_metric(k["praise_rate"], "{:.1f}%"),
                "caption": "反馈中带有表扬内容的占比",
            },
        ],
        logo_path=logo_path,
        logo_alt="新东方欧亚教育",
    )

    # ── 业务线概览卡片 ──
    render_section_header(
        "业务线概览",
        "日本留学 vs 日语标化",
        "快速对比两条业务线的核心指标，确认差异方向。",
    )
    seg_cols = st.columns(max(len(cfg.sources), 1))
    comp_map = {row["维度"]: row for _, row in compare.iterrows()} if not compare.empty else {}
    for idx, src in enumerate(cfg.sources):
        with seg_cols[idx]:
            row = comp_map.get(src.label)
            if row is None:
                continue
            render_html_block(
                f"""
                <div class="pbi-seg-card">
                    <p class="pbi-seg-title">{src.label}</p>
                    <div class="pbi-seg-grid">
                        <div>
                            <div class="pbi-seg-lbl">评分均值</div>
                            <div class="pbi-seg-val">{format_metric(row.get("评分均值"), "{:.3f}")}</div>
                        </div>
                        <div>
                            <div class="pbi-seg-lbl">样本量</div>
                            <div class="pbi-seg-val">{int(row.get("样本量", 0)):,}</div>
                        </div>
                        <div>
                            <div class="pbi-seg-lbl">意见率</div>
                            <div class="pbi-seg-val">{format_metric(row.get("意见率"), "{:.1f}%")}</div>
                        </div>
                        <div>
                            <div class="pbi-seg-lbl">表扬率</div>
                            <div class="pbi-seg-val">{format_metric(row.get("表扬率"), "{:.1f}%")}</div>
                        </div>
                    </div>
                </div>
                """
            )

    # ── 三维度核心指标 ──
    render_section_header(
        "三维度核心指标",
        "产品研发 · 产品推广 · 产品服务",
        "按业务线分别展示三大维度的评分均值、意见率、表扬率及意见/表扬量。",
    )
    src_tabs = st.tabs([s.label for s in cfg.sources])
    for i, src in enumerate(cfg.sources):
        with src_tabs[i]:
            df = dfs[src.id]
            pm = build_pillar_matrix(df, cfg.pillars, src.id)
            render_table(pm, int(layout_get(layout, "pillar_table_height", default=180)))

    # ── 多选题 (Q8) ──
    multi_sources = [(sid, spec) for sid, spec in cfg.multi_select.items() if sid in dfs]
    if multi_sources:
        render_section_header(
            "核心需求分布",
            "标化产品改进方向",
            "多选题独立汇总，按选项被选次数排序。",
        )
        multi_cols = st.columns(min(len(multi_sources), 2))
        for j, (sid, spec) in enumerate(multi_sources):
            with multi_cols[j]:
                src_label = cfg.source(sid).label if cfg.source(sid) else sid
                df_multi = build_multi_select_counts(dfs[sid], spec)
                with panel(f"{src_label} — 改进方向", ""):
                    render_multi_select_bars(df_multi, 280, f"overview_q8_{sid}")

    # ── 下钻入口 ──
    render_section_header(
        "下钻分析",
        "进入产品视角或维度视角",
        "概览只展示全局，切片细节请进入对应分析页面。",
    )
    with panel("选择分析视角", ""):
        render_view_nav(
            cfg,
            current_view="overview",
            key_prefix=f"overview_nav_{cfg.id}",
            view_keys=[v.key for v in cfg.views if v.type != "overview"],
        )
