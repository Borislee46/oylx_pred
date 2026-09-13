from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from src.adjustment.gpa_scenario import (
    GPA_TIERS,
    build_gpa_payload,
    ensure_gpa_tier,
    gpa_tier_key,
    gpa_tier_label,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct

logger = setup_logger("page3", "prediction")

_DIST = Path(__file__).resolve().parent / "frontend" / "hk_whatif_axis" / "dist"
_HAS_FRONTEND = (_DIST / "index.html").is_file()

_ENABLED = False

_component = None
if _HAS_FRONTEND:
    _component = st.components.v1.declare_component(
        "hk_whatif_axis",
        path=str(_DIST),
    )

_IFRAME_TRANSPARENT_CSS = """
<style>
iframe[title*="hk_whatif_axis"] {
  background: transparent !important;
  color-scheme: normal;
}
div[data-testid="stCustomComponentV1"]:has(iframe[title*="hk_whatif_axis"]),
div[data-testid="element-container"]:has(iframe[title*="hk_whatif_axis"]) {
  background: transparent !important;
}
</style>
"""

_AXIS_SELECT_KEY = "_hk_whatif_axis_selected"
_AXIS_READY_KEYS = "_hk_whatif_axis_ready_keys"


def _ensure_iframe_transparent() -> None:
    if st.session_state.get("_hk_whatif_axis_iframe_css"):
        return
    st.html(_IFRAME_TRANSPARENT_CSS)
    st.session_state["_hk_whatif_axis_iframe_css"] = True


def render_hk_whatif_axis(
    page_state: Any,
    input_data: dict,
    unified: list[dict],
    *,
    key: str = "hk_whatif_axis",
) -> str | None:
    if not _ENABLED:
        logger.info("hk_whatif_axis 已临时废弃（_ENABLED=False），跳过渲染")
        return None

    if not _HAS_FRONTEND or _component is None:
        logger.warning("hk_whatif_axis dist 未构建，跳过推演卡片")
        return None

    _ensure_iframe_transparent()

    if not unified:
        return None

    base_gpa = input_data.get("gpa") or input_data.get("gpa_model")
    try:
        base_gpa = float(base_gpa)
    except (TypeError, ValueError):
        base_gpa = None
    base_key = gpa_tier_key(
        min(GPA_TIERS, key=lambda t: abs(t - (base_gpa or 3.4))) if base_gpa is not None else 3.4
    )

    base_best = max(
        (clip_probability_coerce(r.get("probability")) for r in unified),
        default=0.0,
    )
    base_entry = {
        "gpa": base_gpa if base_gpa is not None else 3.4,
        "key": base_key,
        "label": gpa_tier_label(base_gpa if base_gpa is not None else 3.4),
        "unified": unified,
        "best_prob": base_best,
        "bands": {},
        "best_pct": prob_to_pct(base_best),
    }

    ready = st.session_state.get(_AXIS_READY_KEYS) or {base_key}
    ready = {k for k in ready if k in {gpa_tier_key(t) for t in GPA_TIERS}}
    ready.add(base_key)

    valid_tiers = {gpa_tier_key(t) for t in GPA_TIERS}
    selected = st.session_state.get(_AXIS_SELECT_KEY)
    if selected not in valid_tiers:
        selected = base_key

    entries = {base_key: base_entry}
    if selected != base_key:
        with st.skeleton(height=760):
            tier_entry = ensure_gpa_tier(
                page_state,
                input_data,
                unified,
                float(selected.split("_")[1]),
            )
        entries[selected] = tier_entry
        ready.add(selected)

    st.session_state[_AXIS_READY_KEYS] = ready

    payload = build_gpa_payload(entries, base_input=input_data, all_tiers=GPA_TIERS)
    payload["selected"] = selected
    payload["ready_keys"] = sorted(ready)

    try:
        result = _component(**payload, default=selected, key=key)
    except Exception:
        logger.exception("hk_whatif_axis component failed")
        return None

    if isinstance(result, str) and result in valid_tiers:
        st.session_state[_AXIS_SELECT_KEY] = result
        if result not in ready:
            logger.info("推演仪 | 检测到新档位 %s，触发懒重算", result)
            st.rerun(scope="app")
        return result
    return selected
