from __future__ import annotations

import pandas as pd

from ..ui.blocks import render_info_cards, render_page_hero
from ..ui.charts import render_hbar_feedback
from ..ui.tables import theme_without_other
from ..ui.theme_css import format_metric


def feedback_tone(opinion_rate: float | None, praise_rate: float | None) -> str:
    op = 0.0 if opinion_rate is None else float(opinion_rate)
    pr = 0.0 if praise_rate is None else float(praise_rate)
    if pr >= op + 3:
        return "positive"
    if op >= pr + 3:
        return "warning"
    return "neutral"


def render_detail_hero(
    title: str,
    subtitle: str,
    *,
    badge: str | None,
    chips: list[str],
    summary: str,
    stats: list[dict[str, str]],
) -> None:
    render_page_hero(
        title,
        subtitle,
        badge=badge,
        eyebrow="Insight View",
        summary=summary,
        chips=chips,
        stats=stats,
    )


def render_detail_kpis(kpi: dict, *, scope_label: str, focus_label: str, columns: int = 3) -> None:
    score = kpi.get("评分均值")
    qty = int(kpi.get("评分量", 0))
    opinion_rate = kpi.get("意见率")
    praise_rate = kpi.get("表扬率")
    render_info_cards(
        [
            {
                "label": "当前切片评分",
                "value": format_metric(score),
                "body": f"{scope_label}下的综合评分均值。",
                "meta": f"评分样本 {qty:,}",
                "tone": "positive" if score is not None and float(score) >= 4 else "neutral",
            },
            {
                "label": "反馈结构",
                "value": f"{format_metric(opinion_rate, '{:.1f}%')} / {format_metric(praise_rate, '{:.1f}%')}",
                "body": "意见率与表扬率共同决定当前口碑结构。",
                "meta": "左侧为意见率，右侧为表扬率",
                "tone": feedback_tone(opinion_rate, praise_rate),
            },
            {
                "label": "分析焦点",
                "value": focus_label,
                "body": "当前页面围绕这一视角展开拆解与对比。",
                "meta": "下方继续查看差异来源与原始证据",
                "tone": "neutral",
            },
        ],
        columns=columns,
    )


def render_theme_bar(df: pd.DataFrame, color: str, height: int, key: str) -> None:
    if df.empty:
        return
    clean = theme_without_other(df)
    if clean.empty:
        return
    clean = clean.assign(反馈量=pd.to_numeric(clean["反馈量"], errors="coerce")).sort_values(
        "反馈量", ascending=True
    )
    labels = [str(x) for x in clean["反馈类型"].tolist()]
    values = [float(x) for x in clean["反馈量"].tolist()]
    render_hbar_feedback(labels, values, color, height, key, top_n=8)
