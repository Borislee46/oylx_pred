"""Application readiness scoring and minimal UI."""

from __future__ import annotations

import html
from typing import Any

import streamlit as st


def build_readiness_profile(
    input_data: dict[str, Any],
    prediction_results: Any,
) -> dict[str, Any]:
    results = _all_results(prediction_results)
    selection = _selection_factor(results)
    background = _background_factor(input_data, results)
    factors = [selection, background]

    available = [f for f in factors if f["score"] is not None]
    overall = round(
        sum(f["score"] * f["weight"] for f in available)
        / max(sum(f["weight"] for f in available), 1),
        1,
    )
    weakest = min(available, key=lambda f: f["score"]) if available else selection

    return {
        "score": overall,
        "label": _overall_label(overall),
        "factors": factors,
        "next_action": _next_action(weakest),
    }


def render_readiness_card(profile: dict[str, Any]) -> None:
    factor_html = "".join(_factor_html(f) for f in profile["factors"])
    st.html(
        '<style>'
        '.hk-ready-card{border:1px solid var(--hk-slate-100);border-radius:18px;'
        'padding:1rem 1.1rem;background:rgba(255,255,255,.72);'
        'box-shadow:0 12px 32px rgba(15,23,42,.06);margin:.75rem 0 1rem}'
        '.hk-ready-top{display:flex;justify-content:space-between;gap:1rem;align-items:flex-start}'
        '.hk-ready-score{font-family:var(--hk-font-display);font-size:2.1rem;'
        'font-weight:800;color:var(--hk-slate-900);line-height:1}'
        '.hk-ready-label{font-size:.78rem;color:var(--hk-slate-400);margin-top:.2rem}'
        '.hk-ready-action{font-size:.86rem;color:var(--hk-slate-600);line-height:1.55;max-width:520px}'
        '.hk-ready-factors{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));'
        'gap:.6rem;margin-top:.85rem}'
        '.hk-ready-factor{border-radius:14px;background:var(--hk-slate-50);padding:.65rem .75rem}'
        '.hk-ready-factor-name{font-size:.72rem;color:var(--hk-slate-400);margin-bottom:.2rem}'
        '.hk-ready-factor-value{font-size:.92rem;font-weight:700;color:var(--hk-slate-700)}'
        '.hk-ready-factor-note{font-size:.72rem;color:var(--hk-slate-400);margin-top:.2rem;line-height:1.35}'
        '</style>'
        '<div class="hk-section-label">申请准备度</div>'
        '<div class="hk-ready-card">'
        '<div class="hk-ready-top">'
        '<div>'
        f'<div class="hk-ready-score">{profile["score"]:.0f}</div>'
        f'<div class="hk-ready-label">{html.escape(profile["label"])}</div>'
        '</div>'
        f'<div class="hk-ready-action">{html.escape(profile["next_action"])}</div>'
        '</div>'
        f'<div class="hk-ready-factors">{factor_html}</div>'
        '</div>'
    )


def _all_results(prediction_results: Any) -> list[dict[str, Any]]:
    if not prediction_results:
        return []
    groups = [
        getattr(prediction_results, "similarity_results", None),
        getattr(prediction_results, "cross_major_results", None),
        getattr(prediction_results, "user_specified_results", None),
    ]
    return [r for group in groups for r in (group or []) if isinstance(r, dict)]


def _selection_factor(results: list[dict[str, Any]]) -> dict[str, Any]:
    probs = sorted((float(r.get("probability", 0) or 0) for r in results), reverse=True)
    top3 = probs[:3]
    score = round((sum(top3) / len(top3)) * 100, 1) if top3 else 0.0
    safety = sum(1 for p in probs if p >= 0.6)
    note = f"Top3 平均 {score:.0f}%，高概率项目 {safety} 个"
    return {"name": "选校风险", "score": score, "note": note, "weight": 0.45}


def _background_factor(
    input_data: dict[str, Any],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    penalty = 0
    if any(r.get("language_penalty_applied") for r in results):
        penalty += 22
    penalty += min(_trace_penalty_count(results) * 8, 34)
    if not input_data.get("experience_details"):
        penalty += 8
    score = max(0, 100 - penalty)
    note = "语言/跨申/经历调整已计入" if penalty else "暂无明显结构性扣分"
    return {"name": "背景风险", "score": round(score, 1), "note": note, "weight": 0.35}


def _trace_penalty_count(results: list[dict[str, Any]]) -> int:
    count = 0
    for result in results[:8]:
        trace = result.get("_adjustment_trace")
        if isinstance(trace, dict):
            count += sum(1 for key in trace if "penalty" in str(key).lower())
        elif isinstance(trace, list):
            count += sum(1 for item in trace if "penalty" in str(item).lower())
    return count


def _overall_label(score: float) -> str:
    if score >= 75:
        return "可以推进，优先做精修"
    if score >= 55:
        return "基本可用，仍有一处关键风险"
    return "建议先调整后再提交"


def _next_action(factor: dict[str, Any]) -> str:
    name = factor["name"]
    if name == "背景风险":
        return "下一步优先确认语言、经历和跨专业跨度，避免概率被结构性扣分。"
    return "下一步调整目标梯度，保留冲刺项，同时补足更稳的相似专业。"


def _factor_html(factor: dict[str, Any]) -> str:
    value = "待评估" if factor["score"] is None else f'{factor["score"]:.0f}'
    return (
        '<div class="hk-ready-factor">'
        f'<div class="hk-ready-factor-name">{html.escape(factor["name"])}</div>'
        f'<div class="hk-ready-factor-value">{value}</div>'
        f'<div class="hk-ready-factor-note">{html.escape(factor["note"])}</div>'
        '</div>'
    )
