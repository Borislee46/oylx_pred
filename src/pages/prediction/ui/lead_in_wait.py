from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st

from src.agent.lead_in.dispatch_constants import MAX_RETRIES
from src.pages.prediction.ui.lead_in_progress import render_progress_html
from src.pages.prediction.ui.lead_in_progress_copy import (
    PIPELINE,
    humanize_detail,
    stage_index,
    status_hint,
)
from src.pages.prediction.ui.lead_in_wait_profile import (
    extract_tags,
    infer_persona,
    parse_score_norms,
    profile_from_sources,
)

_DIST = Path(__file__).parent / "frontend" / "lead_in_wait" / "dist"

_component = st.components.v1.declare_component(
    "lead_in_wait",
    path=str(_DIST),
)

_IFRAME_TRANSPARENT_CSS = """
<style>
iframe[title*="lead_in_wait"] {
  background: transparent !important;
  color-scheme: normal;
}
div[data-testid="stCustomComponentV1"]:has(iframe[title*="lead_in_wait"]),
div[data-testid="element-container"]:has(iframe[title*="lead_in_wait"]) {
  background: transparent !important;
}
</style>
"""

_DEFAULT_BG_DARK = "#0b1220"
_DEFAULT_BG_LIGHT = "#ffffff"


def _ensure_iframe_transparent() -> None:
    if st.session_state.get("_lead_in_wait_iframe_css"):
        return
    st.html(_IFRAME_TRANSPARENT_CSS)
    st.session_state["_lead_in_wait_iframe_css"] = True


_STAGE_SHORT = {
    "解析学生背景": "读背景",
    "结构化提取": "提取",
    "写入表单": "填表",
    "完成": "完成",
}


def build_wait_props(
    steps: list[str],
    partial_text: str,
    *,
    elapsed: float = 0,
    retry: int = 0,
    details: list[str] | None = None,
    path_hint: str = "",
    ctx: Any | None = None,
    applied: dict[str, Any] | None = None,
    sse_port: int = 0,
    sse_run_id: str = "",
    sse_url: str = "",
    dark: bool = False,
    bg_color: str = "",
) -> dict[str, Any]:
    detail_lines = [str(x).strip() for x in (details or []) if str(x).strip()]
    idx = stage_index(steps)
    if idx < 0:
        idx = 0
    name = PIPELINE[min(idx, len(PIPELINE) - 1)]
    stage = _STAGE_SHORT.get(name, name)

    ph = (path_hint or "").strip()
    if ph:
        subtitle = ph
    elif detail_lines:
        subtitle = humanize_detail(detail_lines[-1])
    else:
        pt = (partial_text or "").strip()
        subtitle = humanize_detail(pt) if pt else "正在接通…"

    feed: list[str] = []
    for raw in detail_lines:
        line = humanize_detail(raw)
        if line and (not feed or feed[-1] != line):
            feed.append(line)
    feed = feed[-6:]

    llm_raw = (partial_text or "").strip() or (feed[-1] if feed else "")
    llm_text = humanize_detail(llm_raw) if llm_raw else subtitle

    profile = profile_from_sources(ctx, applied)
    text_gpa, text_lang = parse_score_norms(ph, llm_raw, subtitle, *detail_lines)
    gpa_norm = profile["gpa_norm"] if profile["gpa_norm"] is not None else text_gpa
    lang_norm = profile["lang_norm"] if profile["lang_norm"] is not None else text_lang

    persona = infer_persona(profile["major_blob"])
    if persona == "signal":
        persona = infer_persona(
            profile["raw_input"],
            ph,
            llm_raw,
            subtitle,
            *detail_lines,
        )
    tags = profile["tags"] or extract_tags(detail_lines, partial_text or "")
    if profile.get("raw_input") and len(tags) < 3:
        extra = extract_tags([], str(profile["raw_input"]))
        for t in extra:
            if t not in tags:
                tags.append(t)
            if len(tags) >= 6:
                break

    if ph and feed and subtitle == ph:
        pass
    elif feed and (not subtitle or subtitle == llm_text):
        subtitle = feed[-1]

    return {
        "title": "页面闪断，正在自动续跑" if retry > 0 else "Signals 正在解读",
        "stage": stage,
        "subtitle": subtitle,
        "hint": status_hint(elapsed, active=True, retry=retry),
        "elapsed": float(elapsed or 0),
        "retry": int(retry or 0),
        "stage_index": int(idx),
        "stage_count": len(PIPELINE),
        "persona": persona,
        "llm_text": llm_text,
        "details": feed,
        "tags": tags,
        "gpa_norm": float(gpa_norm),
        "lang_norm": float(lang_norm),
        "gpa_label": profile.get("gpa_label", ""),
        "lang_label": profile.get("lang_label", ""),
        "sse_url": str(sse_url or ""),
        "sse_port": int(sse_port or 0),
        "sse_run_id": str(sse_run_id or ""),
        "started_at": float(time.time() - max(elapsed, 0)) if elapsed and elapsed > 0 else 0,
        "retry_max": int(MAX_RETRIES),
        "dark": bool(dark),
        "bg_color": bg_color or (_DEFAULT_BG_DARK if dark else _DEFAULT_BG_LIGHT),
    }


def render_lead_in_wait(
    steps: list[str],
    partial_text: str,
    *,
    elapsed: float = 0,
    retry: int = 0,
    variant: str = "default",
    details: list[str] | None = None,
    path_hint: str = "",
    ctx: Any | None = None,
    applied: dict[str, Any] | None = None,
    sse_port: int = 0,
    sse_run_id: str = "",
    sse_url: str = "",
    key: str | None = "lead_in_wait",
) -> Any:
    detail_lines = [str(x).strip() for x in (details or []) if str(x).strip()]

    if variant == "intent_blocked":
        st.html(
            render_progress_html(
                steps,
                partial_text,
                elapsed=elapsed,
                retry=retry,
                variant=variant,
                details=detail_lines,
                active=True,
                path_hint=path_hint,
            )
        )
        return None

    if ctx is None:
        ctx = st.session_state.get("lead_in_ctx")
    if applied is None:
        applied = st.session_state.get("_lead_in_last_applied") or {}

    _ensure_iframe_transparent()

    try:
        dark = str(st.get_option("theme.base") or "").lower() == "dark"
        bg_color = str(st.get_option("theme.backgroundColor") or "").strip()
    except Exception:
        dark = False
        bg_color = ""

    props = build_wait_props(
        steps,
        partial_text,
        elapsed=elapsed,
        retry=retry,
        details=detail_lines,
        path_hint=path_hint,
        ctx=ctx,
        applied=applied if isinstance(applied, dict) else {},
        sse_port=sse_port,
        sse_run_id=sse_run_id,
        sse_url=sse_url,
        dark=dark,
        bg_color=bg_color,
    )
    return _component(**props, key=key, default=None)
