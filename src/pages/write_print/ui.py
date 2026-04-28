"""WritePrint — AI text detection & LLM-powered rewriting, minimalist UI."""

import hashlib

import streamlit as st

from src.pages.write_print.engine import analyze
from src.utils import SUPPORT_EMAIL, SessionManager, log_interaction_event
from src.pages.write_print.model import get_model
from src.pages.write_print.features import read_input
from src.pages.write_print.profile import build_writing_profile
from src.pages.write_print.rewriter import generate_full_rewrite

STUB_FOOTER = (
    '<div class="hk-footer" style="display:flex;justify-content:flex-end;align-items:center;">'
    '<span style="font-size:0.78rem;color:var(--hk-slate-300)">'
    'WritePrint — 文书AI检测</span><div class="hk-footer-dot"></div>'
    f'<a href="mailto:{SUPPORT_EMAIL}" style="font-size:0.78rem;color:var(--hk-slate-300)">'
    f'技术支持：{SUPPORT_EMAIL}</a></div>'
)

WP_STYLE = """<style>
.wp-ring { width: 160px; height: 160px; margin: 0 auto; position: relative; }
.wp-ring svg { transform: rotate(-90deg); }
.wp-ring-bg { fill: none; stroke: var(--hk-slate-100); stroke-width: 6; }
.wp-ring-fill { fill: none; stroke-width: 6; stroke-linecap: round; transition: stroke-dashoffset 1.2s cubic-bezier(0.34, 1.56, 0.64, 1); }
.wp-ring-center { position: absolute; inset: 0; display: flex; flex-direction: column; align-items: center; justify-content: center; }
.wp-ring-score { font-family: var(--hk-font-display); font-size: 2.6rem; font-weight: 800; line-height: 1; margin: 0; color: var(--hk-slate-900); }
.wp-ring-label { font-size: 0.7rem; font-weight: 500; color: var(--hk-slate-400); letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px; }
.wp-verdict-text { text-align: center; font-size: 0.875rem; color: var(--hk-slate-500); max-width: 480px; margin: 0.75rem auto 0; line-height: 1.55; }
.wp-stats-row { display: flex; justify-content: center; gap: 1.5rem; margin-top: 0.6rem; }
.wp-stat { font-size: 0.72rem; color: var(--hk-slate-400); letter-spacing: 0.02em; }
.wp-bar-row { display: flex; align-items: center; gap: 0.6rem; margin: 0.25rem 0; }
.wp-bar-label { width: 130px; font-size: 0.78rem; font-weight: 500; color: var(--hk-slate-600); text-align: right; flex-shrink: 0; }
.wp-bar-track { flex: 1; height: 5px; background: var(--hk-slate-100); border-radius: 3px; overflow: hidden; }
.wp-bar-fill { height: 100%; border-radius: 3px; transition: width 1s cubic-bezier(0.34, 1.56, 0.64, 1); }
.wp-sent-block { padding: 0.6rem 0; }
.wp-sent-idx { font-size: 0.65rem; font-weight: 700; color: var(--hk-slate-400); letter-spacing: 0.06em; margin-bottom: 0.25rem; }
.wp-sent-orig { font-size: 0.85rem; color: var(--hk-slate-700); border-left: 2px solid var(--hk-slate-200); padding-left: 0.6rem; margin: 0.2rem 0; }
.wp-sent-rewrite { font-size: 0.85rem; padding-left: 0.6rem; margin: 0.3rem 0; border-radius: 2px; }
.wp-sent-rewrite-cons { border-left: 2px solid var(--hk-cyan); }
.wp-sent-rewrite-bold { border-left: 2px solid #8b5cf6; }
.wp-sent-tone { font-size: 0.6rem; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; margin-right: 0.5rem; padding: 1px 5px; border-radius: 2px; }
.wp-tone-cons { background: #ecfeff; color: #0891b2; }
.wp-tone-bold { background: #f5f3ff; color: #7c3aed; }
.wp-divider { height: 1px; background: var(--hk-slate-100); margin: 1rem 0; }
</style>"""


def _score_color(score: float) -> str:
    if score < 30:
        return "#10b981"
    elif score < 55:
        return "#f59e0b"
    return "#ef4444"


def _render_score_ring(score: float, verdict: dict, text_stats: dict) -> None:
    color = _score_color(score)
    circumference = 2 * 3.14159 * 52
    offset = circumference * (1 - score / 100)

    st.html(
        '<div class="wp-ring">'
        f'<svg width="160" height="160" viewBox="0 0 160 160">'
        f'<circle class="wp-ring-bg" cx="80" cy="80" r="52"/>'
        f'<circle class="wp-ring-fill" cx="80" cy="80" r="52"'
        f' style="stroke:{color};stroke-dasharray:{circumference:.1f};'
        f'stroke-dashoffset:{offset:.1f};" /></svg>'
        f'<div class="wp-ring-center">'
        f'<span class="wp-ring-score">{score:.0f}%</span>'
        f'<span class="wp-ring-label">AI Score</span></div></div>'
        f'<p class="wp-verdict-text"><strong>{verdict["label"]}</strong>'
        f' &mdash; {verdict["description"]}</p>'
        f'<div class="wp-stats-row">'
        f'<span class="wp-stat">{text_stats["words"]:,} words</span>'
        f'<span class="wp-stat">{text_stats["sentences"]} sentences</span>'
        f'<span class="wp-stat">{text_stats["paragraphs"]} paragraphs</span>'
        f'</div>'
    )


def _render_feature_bars(features: dict) -> None:
    contributions = [fb["contribution"] for fb in features.values()]
    max_abs = max(abs(c) for c in contributions) if contributions else 1

    for name, fb in features.items():
        pct = abs(fb["contribution"]) / max_abs * 100 if max_abs > 0 else 0
        bar_color = "var(--hk-cyan)" if fb["contribution"] > 0 else "var(--hk-slate-300)"
        direction = "+" if fb["contribution"] > 0 else ""
        st.html(
            '<div class="wp-bar-row">'
            f'<span class="wp-bar-label">{fb["label"]}</span>'
            '<div class="wp-bar-track">'
            f'<div class="wp-bar-fill" style="width:{pct:.0f}%;background:{bar_color};"></div>'
            '</div>'
            f'<span style="font-size:0.72rem;color:var(--hk-slate-400);">'
            f'{direction}{fb["contribution"]:.1f}</span></div>'
        )


@st.cache_resource(show_spinner=False)
def _load_model():
    return get_model()


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


def _render_footer() -> None:
    st.page_link("main.py", label="← 返回首页",
                 query_params={"scroll_to": "main-page-header-anchor"})
    st.html(STUB_FOOTER)


def _clear_wp_state():
    for k in ("wp_result", "wp_text_hash", "wp_rewritten", "wp_new_result",
              "wp_processing"):
        st.session_state.pop(k, None)


def _publish_writing_profile(
    text: str,
    result: dict,
    new_result: dict | None = None,
) -> None:
    profile = build_writing_profile(text, result, new_result)
    SessionManager().set(writing_profile=profile)
    st.session_state["writing_profile"] = profile
    log_interaction_event(
        "write_print_profile",
        {
            "text_hash": profile["text_hash"],
            "score": profile["score"],
            "risk_level": profile["risk_level"],
            "after_rewrite_score": profile.get("after_rewrite_score"),
            "top_feature_count": len(profile.get("top_features", [])),
            "top_fix_count": len(profile.get("top_fixes", [])),
        },
    )


def render() -> None:
    st.html(WP_STYLE)

    st.html(
        '<div class="hk-header"><div>'
        '<p class="hk-header-title">WritePrint</p>'
        '<p class="hk-header-subtitle">文书AI检测 & 智能改写</p>'
        '</div></div>'
    )
    st.caption("Paste a personal statement to detect AI patterns and get human-like rewrites.")

    input_method = st.radio("Input", ["Paste text", "Upload file"],
                            horizontal=True, label_visibility="collapsed")

    text = ""
    if input_method == "Upload file":
        uploaded = st.file_uploader("Drop a PDF or TXT file here", type=["pdf", "txt"])
        if uploaded:
            import tempfile
            import os
            suffix = ".pdf" if uploaded.name.endswith(".pdf") else ".txt"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name
            try:
                text = read_input(tmp_path)
            finally:
                os.unlink(tmp_path)
            if text:
                st.caption(f"Loaded: {uploaded.name} ({len(text):,} chars)")
    else:
        text = st.text_area(
            "Paste text", height=200,
            placeholder="Paste your personal statement here...",
            label_visibility="collapsed",
        )

    # Detect text change → invalidate old results
    text_changed = (_text_hash(text) != st.session_state.get("wp_text_hash"))
    if text_changed and "wp_result" in st.session_state:
        _clear_wp_state()

    if st.button("Analyze", type="primary", disabled=len(text) < 50,
                 use_container_width=True, key="submit_button_key"):
        model, scaler = _load_model()
        with st.spinner("Analyzing..."):
            result = analyze(text, scaler, model)

        if "error" in result:
            st.error(result["error"])
            return

        _clear_wp_state()
        st.session_state["wp_result"] = result
        st.session_state["wp_text_hash"] = _text_hash(text)
        _publish_writing_profile(text, result)

    # ── render analysis results ──
    result = st.session_state.get("wp_result")
    if not result:
        _render_footer()
        return

    _render_score_ring(result["score"], result["verdict"], result["text_stats"])
    st.html('<div class="wp-divider"></div>')
    _render_feature_bars(result["features"])

    # Full-text humanize button
    st.html('<div class="wp-divider"></div>')
    col_a, col_b = st.columns([2, 1])

    processing = st.session_state.get("wp_processing", False)
    with col_a:
        disabled = processing or st.session_state.get("wp_text_hash") is None
        if st.button("Humanize — Full Text Rewrite", type="secondary",
                     use_container_width=True, disabled=disabled):
            st.session_state["wp_processing"] = True
            st.rerun()

    # Execute rewrite outside button handler to avoid double-state
    if processing:
        with st.spinner("LLM rewriting full text with feature guidance..."):
            rewritten = generate_full_rewrite(
                text, result["score"], result["features"],
            )
        if rewritten:
            st.session_state["wp_rewritten"] = rewritten
            model, scaler = _load_model()
            new_result = analyze(rewritten, scaler, model)
            st.session_state["wp_new_result"] = new_result
            _publish_writing_profile(text, result, new_result)
        else:
            st.toast("LLM rewrite failed — please try again", icon="⚠️")
        st.session_state["wp_processing"] = False
        st.rerun()

    # Show rewritten result
    rewritten_text = st.session_state.get("wp_rewritten")
    if rewritten_text and "wp_new_result" in st.session_state:
        new_result = st.session_state["wp_new_result"]
        diff = result["score"] - new_result["score"]
        with col_b:
            color = "#10b981" if new_result["score"] <= 20 else "#f59e0b"
            st.html(
                f'<div style="text-align:center;padding:0.5rem;">'
                f'<span style="font-size:2rem;font-weight:800;color:{color};">'
                f'{new_result["score"]:.0f}%</span>'
                f'<span style="font-size:0.8rem;color:var(--hk-slate-400);">'
                f' (Δ{diff:+.0f}%)</span></div>'
            )
        st.text_area(
            "Rewritten text", value=rewritten_text,
            height=300, label_visibility="collapsed",
        )

    st.caption("Score ±18% confidence (LOO-CV MAE). 10-feature Ridge model.")
    _render_footer()
