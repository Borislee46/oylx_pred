from __future__ import annotations

from typing import TYPE_CHECKING, Any

import streamlit as st

from src.pages.prediction.result_display.product_registry import (
    PRODUCTS_DEF,
    ProductKind,
    normalize_selection,
    registry_by_name,
    selectable_product_names,
)
from src.pages.prediction.result_display.product_registry import (
    _diag_language as _diag_language,
)
from src.pages.prediction.result_display.product_registry import (
    _diag_tutoring as _diag_tutoring,
)
from src.utils.logger import setup_logger

if TYPE_CHECKING:
    from src.utils.session_manager import SessionManager

_logger = setup_logger("page3", "prediction")

__all__ = [
    "BLOCKS_INITIALIZED_KEY",
    "BLOCKS_STATE_KEY",
    "PORTFOLIO_REVEALED_KEY",
    "blocks_selection_initialized",
    "PRODUCT_META",
    "SCENARIO_STATE_KEY",
    "SIM_CACHE_KEY",
    "get_blocks_selection",
    "get_portfolio_revealed",
    "normalize_selection",
    "product_hint_tags",
    "registry_by_name",
    "selectable_product_names",
    "set_blocks_selection",
    "set_portfolio_revealed",
    "split_selection",
]

SCENARIO_STATE_KEY = "hk_sales_scenario"
SIM_CACHE_KEY = "hk_sales_sim_cache"
BLOCKS_STATE_KEY = "hk_sales_blocks_selection"
BLOCKS_INITIALIZED_KEY = "hk_sales_blocks_initialized"
PORTFOLIO_REVEALED_KEY = "hk_sales_portfolio_revealed"

PRODUCT_META: dict[str, dict[str, Any]] = {
    p.name: {
        "label": p.name,
        "tooltip": p.tooltip,
        "diagnose": p.diagnose or (lambda _d: None),
        "kind": p.kind.value,
    }
    for p in PRODUCTS_DEF
}


def blocks_selection_initialized(session_manager: SessionManager | None = None) -> bool:
    from src.utils.session_manager import SessionManager as _SM

    sm = session_manager or _SM()
    return bool(sm.get(BLOCKS_INITIALIZED_KEY))


def get_blocks_selection(session_manager: SessionManager | None = None) -> list[str]:
    from src.utils.session_manager import SessionManager as _SM

    sm = session_manager or _SM()
    if blocks_selection_initialized(sm):
        val = sm.get(BLOCKS_STATE_KEY)
        return normalize_selection(val) if isinstance(val, list) else []
    legacy = st.session_state.get(BLOCKS_STATE_KEY)
    if isinstance(legacy, list):
        picked = normalize_selection(legacy)
        sm.set(**{BLOCKS_STATE_KEY: picked, BLOCKS_INITIALIZED_KEY: True})
        st.session_state.pop(BLOCKS_STATE_KEY, None)
        return picked
    return []


def set_blocks_selection(
    selected: list[str],
    session_manager: SessionManager | None = None,
) -> list[str]:
    from src.pages.prediction.ui.explain_cache import invalidate_pathfinder_pdf
    from src.pages.prediction.ui.sales_explain_bridge import blocks_selection_fingerprint
    from src.utils.session_manager import SessionManager as _SM

    sm = session_manager or _SM()
    prev = get_blocks_selection(sm)
    prev_fp = blocks_selection_fingerprint(prev)
    picked = normalize_selection(selected)
    sm.set(**{BLOCKS_STATE_KEY: picked, BLOCKS_INITIALIZED_KEY: True})
    new_fp = blocks_selection_fingerprint(picked)
    if new_fp != prev_fp:
        if "explain_cache" in st.session_state:
            st.session_state["explain_cache"] = {}
        invalidate_pathfinder_pdf()
        app, _ = split_selection(picked)
        if not app:
            sm.set(**{PORTFOLIO_REVEALED_KEY: False})
    return picked


def get_portfolio_revealed(session_manager: SessionManager | None = None) -> bool:
    from src.utils.session_manager import SessionManager as _SM

    sm = session_manager or _SM()
    return bool(sm.get(PORTFOLIO_REVEALED_KEY))


def set_portfolio_revealed(revealed: bool, session_manager: SessionManager | None = None) -> bool:
    from src.utils.session_manager import SessionManager as _SM

    sm = session_manager or _SM()
    sm.set(**{PORTFOLIO_REVEALED_KEY: bool(revealed)})
    return bool(revealed)


def split_selection(selected: list[str]) -> tuple[str | None, list[str]]:
    app: str | None = None
    upgrades: list[str] = []
    for name in normalize_selection(selected):
        p = registry_by_name().get(name)
        if p and p.kind == ProductKind.APPLICATION:
            app = name
        else:
            upgrades.append(name)
    return app, upgrades


def product_hint_tags(selected: list[str], base_input: dict) -> list[str]:
    tags = []
    for name in normalize_selection(selected):
        p = registry_by_name().get(name)
        if not p:
            tags.append(name)
            continue
        diag = p.diagnose(base_input) if p.diagnose else None
        tags.append(f"{p.name}（{diag}）" if diag else p.name)
    return tags
