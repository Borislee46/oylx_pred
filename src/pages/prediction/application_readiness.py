
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
        '<div class="hk-section-label">风险摘要</div>'
        '<div class="hk-ready-card">'
        '<div class="hk-ready-top">'
        '<div>'
        '<div class="hk-ready-kicker">当前判断</div>'
        f'<div class="hk-ready-title">{html.escape(profile["label"])}</div>'
        '</div>'
        f'<div class="hk-ready-action"><span>下一步</span>{html.escape(profile["next_action"])}</div>'
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
    note = f"Top3 平均 {score:.0f}% · 稳妥项 {safety}"
    return {"name": "目标梯度", "score": score, "note": note, "weight": 0.45}


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
    note = "存在语言/跨申/经历约束" if penalty else "暂无明显硬伤"
    return {"name": "背景约束", "score": round(score, 1), "note": note, "weight": 0.35}


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
        return "当前组合较稳，可以进入核验"
    if score >= 55:
        return "可以推进，但需先处理关键约束"
    return "建议先调整后再提交"


def _next_action(factor: dict[str, Any]) -> str:
    name = factor["name"]
    if name == "背景约束":
        return "先核验语言、经历和跨专业跨度。"
    return "先补足更稳的相似专业，再保留少量冲刺项。"


def _factor_html(factor: dict[str, Any]) -> str:
    value = _factor_label(factor.get("score"))
    return (
        '<div class="hk-ready-factor">'
        f'<div class="hk-ready-factor-name">{html.escape(factor["name"])}</div>'
        f'<div class="hk-ready-factor-value">{value}</div>'
        f'<div class="hk-ready-factor-note">{html.escape(factor["note"])}</div>'
        '</div>'
    )


def _factor_label(score: float | None) -> str:
    if score is None:
        return "待核验"
    if score >= 75:
        return "较稳"
    if score >= 55:
        return "需核验"
    return "需调整"
