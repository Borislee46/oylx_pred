from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import streamlit as st

from ..engine.kpis import overview_kpis
from ..layout import get_page_layout, layout_get
from ..loader import load_all_sources
from ..schema import SurveyConfig
from ..ui.blocks import (
    panel,
    render_html_block,
    render_info_cards,
    render_page_hero,
    render_section_header,
)
from ..ui.navigation import render_view_nav
from ..ui.tables import render_table
from ..ui.theme_css import format_metric, inject_overview_css


def _tone_for_score(score: float | None) -> str:
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return "neutral"
    if score >= 4.3:
        return "positive"
    if score < 4.0:
        return "warning"
    return "neutral"


def _overview_cards(cfg: SurveyConfig, k: dict) -> list[dict[str, str]]:
    pillar_items = [
        (name, score)
        for name, score in k["pillars"].items()
        if score is not None and not (isinstance(score, float) and math.isnan(score))
    ]
    pillar_items.sort(key=lambda item: item[1], reverse=True)
    best = pillar_items[0] if pillar_items else None
    worst = pillar_items[-1] if pillar_items else None
    compare = k["compare"].copy()
    cards = []
    if best:
        cards.append(
            {
                "label": "摘要结论",
                "value": best[0],
                "body": f"当前最稳定的正向体验来自 {best[0]}，评分 {best[1]:.2f}，说明这一环节已有较成熟的服务动作可复用。",
                "priority": "P3 维持优势：把有效动作沉淀为标准做法，避免优势被后续波动稀释。",
                "meta": "建议优先作为横向对标样本，而不是当前阶段的主要整改入口",
                "badge": "优势项",
                "tone": "positive",
            }
        )
    if worst:
        cards.append(
            {
                "label": "摘要结论",
                "value": worst[0],
                "body": f"{worst[0]} 当前评分 {worst[1]:.2f}，是本期最优先的改进缺口，建议先下钻产品类型和原始反馈确认问题来源。",
                "priority": "P1 优先处理：先看差异最大的业务切片，再结合主题和文本做原因归类。",
                "meta": "这是最适合作为本期整改主线的维度",
                "badge": "风险项",
                "tone": "warning",
            }
        )
    if not compare.empty:
        compare["评分均值"] = compare["评分均值"].astype(float)
        leader = compare.sort_values("评分均值", ascending=False).iloc[0]
        lagger = compare.sort_values("评分均值", ascending=True).iloc[0]
        cards.append(
            {
                "label": "摘要结论",
                "value": f"{leader['维度']} vs {lagger['维度']}",
                "body": (
                    f"业务线对比上，{leader['维度']} 当前领先，评分 {format_metric(leader['评分均值'], '{:.3f}')}；"
                    f"{lagger['维度']} 相对靠后，更适合优先查看其意见主题与样本结构。"
                ),
                "priority": "P2 横向诊断：先比较领先与靠后业务线的主题分布差异，再决定是否复制打法。",
                "meta": "业务线差异适合作为管理层追问“为什么不同”的入口",
                "badge": "差异项",
                "tone": "neutral",
            }
        )
    return cards


def _pillar_cards(cfg: SurveyConfig, k: dict) -> list[dict[str, str]]:
    cards = []
    for pillar in cfg.pillars:
        score = k["pillars"].get(pillar.name)
        cards.append(
            {
                "label": pillar.name,
                "value": format_metric(score),
                "body": "该维度反映服务体验中对应环节的综合感受。",
                "meta": "分值越高代表整体反馈越稳定",
                "tone": _tone_for_score(score),
            }
        )
    return cards


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

    render_section_header(
        "Executive Snapshot",
        "先看本期最值得关注的结论",
        "这一层改成报告摘要式阅读：不是展示全部结果，而是先交代 P1/P2/P3 的关注顺序，帮助阅读者知道该先处理什么。",
    )
    render_info_cards(
        _overview_cards(cfg, k),
        columns=int(layout_get(layout, "summary_card_columns", default=3)),
    )

    render_section_header(
        "Dimension Scan",
        "核心维度判断",
        "把所有维度并列展示，但强化高低差异，让页面先回答“哪里表现最好、哪里要优先改”。",
    )
    render_info_cards(
        _pillar_cards(cfg, k),
        columns=min(
            max(len(cfg.pillars), 1),
            int(layout_get(layout, "pillar_card_max_columns", default=4)),
        ),
    )

    render_section_header(
        "Segment Comparison",
        "业务线亮点与差异",
        "业务线卡片用于快速建立横向感觉，右侧矩阵保留完整指标，兼顾汇报感和可核查性。",
    )
    left, right = st.columns(layout_get(layout, "segment_columns", default=[1.6, 1]))
    comp_map = {row["维度"]: row for _, row in compare.iterrows()} if not compare.empty else {}
    with left:
        seg_cols = st.columns(max(len(cfg.sources), 1))
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
                                <div class="pbi-seg-val">{int(row.get("评分量", 0)):,}</div>
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
    with right:
        with panel("业务线指标矩阵", "保留完整对比，便于继续确认差异是否由样本量或反馈结构造成。"):
            render_table(
                compare, int(layout_get(layout, "segment_compare_table_height", default=270))
            )

    render_section_header(
        "Next Step",
        "进入下一层分析",
        "从总览进入切片分析后，页面会开始回答差异来自哪里、哪些产品或维度在拉高或拉低整体体验。",
    )
    with panel("选择分析视角", "建议先看最关心的业务切片，再结合反馈主题与原始文本定位原因。"):
        render_view_nav(
            cfg,
            current_view="overview",
            key_prefix=f"overview_nav_{cfg.id}",
            view_keys=[v.key for v in cfg.views if v.type != "overview"],
        )
