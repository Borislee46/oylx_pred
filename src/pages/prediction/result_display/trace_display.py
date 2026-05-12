"""
trace_display: 概率调整链路的可视化组件。

核心设计——30 秒能讲完一个 case 的故事：
- header: 一句话 punchline + 4 个 metric chip（起点 / 终点 / Δ / 主导因子）
- waterfall: 真瀑布图（浮动柱），baseline 虚线作历史对比锚点，每条带设计意图 tooltip
- counterfactual: 4 个扰动场景下的预测概率，回答"如果背景调整..."
- calibration: 模型校准指标（Brier / AUC / Threshold / 阳性率偏差），证明 pipeline 必要性
- selector: 多 case 时顶部 radio 切换 top 3
"""

import hashlib
import html
import json
from pathlib import Path
from typing import Any

import streamlit as st

from src.pages.prediction.result_display.trace_assets import CSS_TEMPLATE, STEP_INTENT

_STEP_DISPLAY = {
    "GPA Penalty": "GPA 偏差惩罚",
    "Language Penalty": "语言偏差惩罚",
    "Cross Major Penalty": "跨专业惩罚",
    "Faculty Out of Scope Penalty": "学部超范围",
    "Professional Major Penalty": "职业学位惩罚",
    "NLP Text Boost": "文本背景提升",
}

_CF_LABEL = {
    "gpa_up": "GPA +0.2",
    "gpa_down": "GPA -0.2",
    "lang_up": "语言 +0.05",
    "intern_up": "实习 +1 段",
}


def _format_pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def _format_delta(v: float) -> str:
    sign = "+" if v > 0 else ""
    return f"{sign}{v * 100:.1f}%"


def _delta_class(v: float) -> str:
    if abs(v) < 0.0001:
        return "dir-zero"
    return "dir-pos" if v > 0 else "dir-neg"


def _info_icon(name: str) -> str:
    intent = STEP_INTENT.get(name)
    if not intent:
        return ""
    return f'<i class="trace-info">i<span class="trace-tooltip">{html.escape(intent)}</span></i>'


def _baseline_line(baseline: float | None, n: int = 0, with_cap: bool = False) -> str:
    if baseline is None:
        return ""
    pct = max(2.0, min(98.0, baseline * 100))
    n_str = f"（n={n}）" if n > 0 else ""
    cap = (
        f'<span class="trace-baseline-cap">历史平均 {_format_pct(baseline)}{n_str}</span>'
        if with_cap
        else ""
    )
    return f'<div class="trace-baseline" style="left:{pct:.2f}%;">{cap}</div>'


def _row(
    label: str,
    track_inner: str,
    pct: float,
    delta: float | None,
    desc: str = "",
    info: str = "",
    is_anchor: bool = False,
    dist_html: str = "",
) -> str:
    label_cls = "trace-row-label is-anchor" if is_anchor else "trace-row-label"
    pct_cls = "trace-pct is-anchor" if is_anchor else "trace-pct"

    if delta is None or abs(delta) < 0.0001:
        delta_html = '<span class="trace-delta dir-zero">—</span>'
    else:
        delta_html = (
            f'<span class="trace-delta {_delta_class(delta)}">{_format_delta(delta)}</span>'
        )

    desc_html = (
        f'<span class="trace-desc">{html.escape(desc)}</span>'
        if desc
        else '<span class="trace-desc"></span>'
    )

    row_html = (
        f'<div class="trace-row">'
        f'<span class="{label_cls}">{html.escape(label)}{info}</span>'
        f'<div class="trace-track">{track_inner}</div>'
        f'<span class="{pct_cls}">{_format_pct(pct)}</span>'
        f"{delta_html}{desc_html}"
        f"</div>"
    )
    if dist_html:
        row_html += f'<div class="trace-dist-row">{dist_html}</div>'
    return row_html


def _render_header(base_prob: float, final_prob: float, steps: list[dict]) -> str:
    nonzero = [s for s in steps if abs(s.get("delta", 0)) > 0.0001]
    dominant = max(nonzero, key=lambda s: abs(s["delta"])) if nonzero else None
    delta_total = final_prob - base_prob

    if dominant:
        dom_name = _STEP_DISPLAY.get(dominant["name"], dominant["name"])
        dom_clause = (
            f'（<span class="{_delta_class(dominant["delta"])}">{dom_name} '
            f'{_format_delta(dominant["delta"])}</span> 影响最大）'
        )
    else:
        dom_name = "无调整"
        dom_clause = "（XGBoost 直接透出）"

    punchline = (
        f'XGBoost 原始 <span class="num">{_format_pct(base_prob)}</span> '
        f"→ 经 {len(nonzero)} 层调整后 "
        f'<span class="num">{_format_pct(final_prob)}</span>'
        f"{dom_clause}"
    )

    chips = [
        ("起点", _format_pct(base_prob), ""),
        ("终点", _format_pct(final_prob), ""),
        ("Δ", _format_delta(delta_total), _delta_class(delta_total)),
        ("主导因子", dom_name, ""),
    ]
    chip_html = "".join(
        f'<div class="trace-chip">'
        f'<span class="trace-chip-label">{label}</span>'
        f'<span class="trace-chip-value {cls}">{html.escape(str(val))}</span>'
        f"</div>"
        for label, val, cls in chips
    )

    return (
        f'<div class="trace-header">'
        f'<div class="trace-punchline">{punchline}</div>'
        f'<div class="trace-chips">{chip_html}</div>'
        f"</div>"
    )


def _render_waterfall(
    steps: list[dict],
    base_prob: float,
    final_prob: float,
    baseline: float | None,
    baseline_n: int = 0,
) -> str:
    rows: list[str] = []

    base_bar = f'<div class="trace-bar-solid kind-base" style="width:{base_prob * 100:.2f}%"></div>'
    rows.append(
        _row(
            "XGBoost 原始",
            _baseline_line(baseline, n=baseline_n, with_cap=True) + base_bar,
            base_prob,
            None,
            desc="校准后概率",
            info=_info_icon("XGBoost 原始"),
            is_anchor=True,
        )
    )

    for step in steps:
        delta = float(step.get("delta", 0))
        if abs(delta) < 0.0001:
            continue
        before = float(step["before"])
        after = float(step["after"])
        left = min(before, after) * 100
        width = abs(delta) * 100
        dir_cls = "dir-pos" if delta > 0 else "dir-neg"

        track_inner = (
            _baseline_line(baseline) + f'<div class="trace-bar-floating {dir_cls}" '
            f'style="left:{left:.2f}%;width:{width:.2f}%"></div>'
            f'<div class="trace-anchor" style="left:{after * 100:.2f}%"></div>'
        )
        name = step.get("name", "")
        step_type = step.get("type", "")
        dist = _extract_dist_params(step) if step_type == "penalty" else ""
        rows.append(
            _row(
                f"└─ {_STEP_DISPLAY.get(name, name)}",
                track_inner,
                after,
                delta,
                desc=step.get("description", ""),
                info=_info_icon(name),
                dist_html=dist,
            )
        )

    rows.append('<hr class="trace-divider">')

    final_bar = (
        f'<div class="trace-bar-solid kind-final" style="width:{final_prob * 100:.2f}%"></div>'
    )
    rows.append(
        _row(
            "最终概率",
            _baseline_line(baseline) + final_bar,
            final_prob,
            final_prob - base_prob,
            desc="调整链聚合",
            info=_info_icon("最终概率"),
            is_anchor=True,
        )
    )

    return f'<div class="trace-waterfall">{"".join(rows)}</div>'


def _render_counterfactual(cf: dict) -> str:
    if not cf or "origin" not in cf:
        return ""
    origin = float(cf["origin"])
    cards: list[str] = []
    for key in ("gpa_up", "gpa_down", "lang_up", "intern_up"):
        if key not in cf:
            continue
        new_prob = float(cf[key])
        delta = new_prob - origin
        if abs(delta) < 0.0001:
            kind, delta_text, dcls = "kind-zero", "—", "dir-zero"
        else:
            kind = "kind-pos" if delta > 0 else "kind-neg"
            delta_text = _format_delta(delta)
            dcls = _delta_class(delta)
        cards.append(
            f'<div class="trace-cf-card {kind}">'
            f'<span class="trace-cf-scenario">{html.escape(_CF_LABEL[key])}</span>'
            f'<span class="trace-cf-prob">{_format_pct(new_prob)}</span>'
            f'<span class="trace-cf-delta {dcls}">{delta_text} vs origin</span>'
            f"</div>"
        )
    if not cards:
        return ""
    return (
        '<div class="trace-cf-section">'
        '<div class="trace-cf-title">如果背景调整… '
        '<span style="color:var(--hk-slate-300);font-weight:400">'
        "（约 0.5σ 典型变动，仅核心调整链）</span></div>"
        f'<div class="trace-cf-grid">{"".join(cards)}</div>'
        "</div>"
    )


@st.cache_data(ttl=300, show_spinner=False)
def _load_model_metrics() -> dict[str, Any]:
    """Load key calibration metrics from the latest evaluation JSON."""
    eval_dir = (
        Path(__file__).resolve().parents[3] / "machine_learning_models" / "evaluation_results"
    )
    if not eval_dir.exists():
        return {}
    jsons = sorted(eval_dir.glob("xgboost_evaluation_*.json"), reverse=True)
    if not jsons:
        return {}
    try:
        with open(jsons[0], encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    m = data.get("metrics", {})
    return {
        "brier": m.get("brier_score"),
        "auc": m.get("roc_auc"),
        "threshold": m.get("prediction_threshold"),
        "pred_pos": m.get("positive_rate_predicted"),
        "actual_pos": m.get("positive_rate_actual"),
    }


def _render_calibration() -> str:
    m = _load_model_metrics()
    if not m or m["brier"] is None:
        return ""

    brier = float(m["brier"])
    auc = float(m["auc"])
    threshold = float(m["threshold"])
    pred_pos = float(m["pred_pos"])
    actual_pos = float(m["actual_pos"])
    gap = pred_pos - actual_pos
    gap_text = f"+{gap * 100:.0f}pp" if gap > 0 else f"{gap * 100:.0f}pp"

    items = [
        ("Brier Score", f"{brier:.3f}", ""),
        ("AUC", f"{auc:.3f}", ""),
        ("Threshold", f"{threshold:.2f}", ""),
        ("阳性率偏差", f"{gap_text}", "dir-neg" if gap > 0 else "dir-pos"),
    ]
    chips = "".join(
        f'<div class="trace-cal-chip">'
        f'<span class="trace-cal-chip-label">{label}</span>'
        f'<span class="trace-cal-chip-value {cls}">{val}</span>'
        f"</div>"
        for label, val, cls in items
    )

    rationale = (
        f"预测阳性率 {pred_pos * 100:.0f}% vs 实际 {actual_pos * 100:.0f}%："
        f"模型偏乐观（{gap_text}），调整链将预测拉回更保守的估计。"
    )

    return (
        '<div class="trace-cal-section">'
        '<div class="trace-cal-title">模型校准</div>'
        f'<div class="trace-cal-chips">{chips}</div>'
        f'<div class="trace-cal-rationale">{html.escape(rationale)}</div>'
        "</div>"
    )


def _z_from_desc(desc: str) -> str:
    """Extract z-score for compact display inline in waterfall row."""
    import re

    m = re.search(r"z=([-\d.]+)", desc)
    if not m:
        return ""
    z = float(m.group(1))
    abs_z = abs(z)
    if abs_z < 0.3:
        level, cls = "≈平均", "kind-zero"
    elif abs_z < 0.7:
        level, cls = "略低" if z > 0 else "略高", "kind-neg" if z > 0 else "kind-pos"
    elif abs_z < 1.3:
        level, cls = "偏低" if z > 0 else "偏高", "kind-neg" if z > 0 else "kind-pos"
    else:
        level, cls = "显著低" if z > 0 else "显著高", "kind-neg" if z > 0 else "kind-pos"
    return f'<span class="trace-z-tag {cls}">z={z:.1f} {level}</span>'


def _extract_dist_params(step: dict) -> str:
    """Return compact distribution line from step description."""
    desc = str(step.get("description", ""))
    if not desc or "|" not in desc:
        return ""
    parts = desc.split("|")
    if len(parts) < 2:
        return ""
    z_tag = _z_from_desc(desc)
    dist_text = html.escape(parts[0].strip())
    return f'<span class="trace-dist">{dist_text}</span>{z_tag}'


def render_trace(result: dict[str, Any]) -> None:
    steps = result.get("_adjustment_steps") or []
    if not steps:
        return

    trace = result.get("_adjustment_trace", {}) or {}
    base_prob = float(trace.get("base", steps[0].get("before", 0) or 0) or 0)
    final_prob = float(result.get("probability", steps[-1].get("after", 0)) or 0)
    baseline = result.get("_baseline_admit_rate")
    baseline_n = int(result.get("_baseline_sample_count", 0) or 0)
    cf = result.get("_counterfactuals") or {}

    html_parts = [
        CSS_TEMPLATE,
        '<div class="trace-panel">',
        _render_header(base_prob, final_prob, steps),
        _render_waterfall(steps, base_prob, final_prob, baseline, baseline_n=baseline_n),
        _render_counterfactual(cf),
        _render_calibration(),
        "</div>",
    ]
    st.html("".join(html_parts))


def _selector_key(candidates: list[dict]) -> str:
    # Content-dependent key: radio resets to first when the candidate
    # pool changes, avoiding the edge case of a stale selection index.
    seed = ":".join(f"{c.get('university', '')}|{c.get('major', '')}" for c in candidates)
    digest = hashlib.md5(seed.encode("utf-8")).hexdigest()[:8]
    return f"trace_case_{digest}"


@st.fragment
def render_trace_for_results(results: list[dict[str, Any]] | None) -> None:
    """Render trace for up to 3 top candidates. Caller should pre-filter & sort."""
    if not results:
        return
    candidates = [r for r in results if r.get("_adjustment_steps")]
    if not candidates:
        return
    candidates = candidates[:3]

    if len(candidates) == 1:
        render_trace(candidates[0])
        return

    _rank_labels = ["🥇", "🥈", "🥉"]
    options = [
        f"{_rank_labels[i]} Top{i + 1}  {c.get('university', '?')} · {c.get('major', '?')} · "
        f"{_format_pct(float(c.get('probability', 0) or 0))}"
        for i, c in enumerate(candidates)
    ]
    st.html('<div class="trace-radio">')
    selected = st.radio(
        "调整链路 case",
        options=list(range(len(candidates))),
        format_func=lambda i: options[i],
        horizontal=True,
        key=_selector_key(candidates),
        label_visibility="collapsed",
    )
    st.html("</div>")
    render_trace(candidates[selected])
