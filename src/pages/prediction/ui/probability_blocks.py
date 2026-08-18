from __future__ import annotations

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import streamlit as st

from src.adjustment.sales_scenario import (
    ensure_pipeline_combos,
    pipeline_combos_pending,
)
from src.pages.prediction.result_display.portfolio_service import (
    ensure_portfolio_admit_cache,
    portfolio_admit_pending,
)
from src.pages.prediction.result_display.product_meta import (
    get_blocks_selection,
    get_portfolio_revealed,
    normalize_selection,
    set_blocks_selection,
    set_portfolio_revealed,
    split_selection,
)
from src.pages.prediction.result_display.product_registry import (
    ProductKind,
    build_block_configs,
    build_personalized_fixed_pp_map,
    contract_tier_from_selection,
    resolve_attribution,
    slot_defs_payload,
    synergy_config_payload,
)
from src.utils.session_manager import SessionManager

logger = logging.getLogger(__name__)

_FRONTEND_DIR = (Path(__file__).resolve().parent / "frontend" / "probability_blocks").resolve()
_HAS_FRONTEND = (_FRONTEND_DIR / "index.html").is_file()

_component = None
if _HAS_FRONTEND:
    _component = st.components.v1.declare_component(
        "hk_probability_blocks",
        path=str(_FRONTEND_DIR),
    )


def _render_pill_fallback(
    base_input: dict,
    *,
    key: str,
    contract_tier: str = "",
    session_manager: SessionManager | None = None,
) -> list[str]:
    blocks = build_block_configs(base_input, contract_tier=contract_tier)
    app_names = [b["name"] for b in blocks if b["kind"] == ProductKind.APPLICATION.value]
    upgrade_names = [b["name"] for b in blocks if b["kind"] == ProductKind.UPGRADE.value]
    stored = get_blocks_selection(session_manager)
    app_default = next((n for n in stored if n in app_names), None)
    upgrade_default = [n for n in stored if n in upgrade_names]

    app_picked = st.radio(
        "申请方案（仅一项）",
        options=["（不选）"] + app_names,
        index=(app_names.index(app_default) + 1 if app_default in app_names else 0),
        key=f"{key}_app_fallback",
    )
    upgrade_picked = st.multiselect(
        "升级产品（可多选）",
        options=upgrade_names,
        default=upgrade_default,
        key=f"{key}_upgrade_fallback",
    )
    picked = normalize_selection(
        ([] if app_picked == "（不选）" else [app_picked]) + upgrade_picked
    )
    return set_blocks_selection(picked, session_manager)


def _selection_from_widget(key: str) -> list[str] | None:
    raw = st.session_state.get(key)
    if isinstance(raw, dict) and "selected" in raw:
        return normalize_selection(raw.get("selected") or [])
    return None


def _apply_widget_meta(raw: Any, sm: SessionManager) -> None:
    if not isinstance(raw, dict):
        return
    if "portfolio_revealed" in raw:
        set_portfolio_revealed(bool(raw.get("portfolio_revealed")), sm)


def render_probability_blocks(
    base_input: dict,
    precomputed: dict[str, Any],
    *,
    page_state: Any = None,
    key: str = "hk_prob_blocks",
    contract_tier: str = "",
    session_manager: SessionManager | None = None,
) -> list[str]:
    sm = session_manager or SessionManager()
    widget_raw = st.session_state.get(key)
    stored = _selection_from_widget(key)
    if stored is None:
        stored = get_blocks_selection(sm)
    else:
        _apply_widget_meta(widget_raw, sm)

    contract_tier = contract_tier_from_selection(stored, contract_tier)
    if "personalized_fixed_pp" not in precomputed:
        precomputed["personalized_fixed_pp"] = build_personalized_fixed_pp_map(
            base_input, contract_tier
        )

    base_unified = precomputed.get("unified_by_combo", {}).get("") or []
    if page_state is not None and pipeline_combos_pending(precomputed, stored):
        with st.skeleton(height=120):
            ensure_pipeline_combos(precomputed, page_state, base_input, base_unified, stored)

    if not _HAS_FRONTEND or _component is None:
        return _render_pill_fallback(
            base_input, key=key, contract_tier=contract_tier, session_manager=sm
        )

    unified_by_combo = precomputed.get("unified_by_combo", {})
    needs_portfolio = portfolio_admit_pending(unified_by_combo, stored)
    loading_ctx = st.skeleton(height=120) if needs_portfolio else nullcontext()
    with loading_ctx:
        portfolio = ensure_portfolio_admit_cache(unified_by_combo, stored)

    attr = resolve_attribution(
        stored,
        base_pct=int(precomputed.get("base_pct", 0) or 0),
        combos=precomputed.get("combos", {}),
        personalized_fixed_pp=precomputed.get("personalized_fixed_pp"),
    )
    app, _ = split_selection(stored)
    portfolio_revealed = get_portfolio_revealed(sm) if app else False
    if not app and get_portfolio_revealed(sm):
        set_portfolio_revealed(False, sm)
        portfolio_revealed = False

    config: dict[str, Any] = {
        "slots": slot_defs_payload(),
        "blocks": build_block_configs(base_input, contract_tier=contract_tier),
        "base_pct": min(100, int(precomputed.get("base_pct", 0) or 0)),
        "combos": precomputed.get("combos", {}),
        "initial": stored,
        "portfolio": portfolio,
        "synergy": synergy_config_payload(),
        "attribution": {
            "base_pct": int(attr["base_pct"]),
            "final_pct": int(attr["final_pct"]),
            "contributions": {k: float(v) for k, v in (attr.get("contributions") or {}).items()},
        },
        "portfolio_revealed": portfolio_revealed,
    }
    try:
        result = _component(config=config, default=None, key=key)
    except Exception:
        logger.exception("hk_probability_blocks component failed, using fallback")
        return _render_pill_fallback(
            base_input, key=key, contract_tier=contract_tier, session_manager=sm
        )

    if isinstance(result, dict):
        selected = normalize_selection(result.get("selected") or [])
        if "portfolio_revealed" in result:
            set_portfolio_revealed(bool(result.get("portfolio_revealed")), sm)
        return set_blocks_selection(selected, sm)
    return stored
