from __future__ import annotations

import streamlit as st

from src.portfolio.portfolio_contract import (
    PortfolioContract,
    load_contracts,
)
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

_logger = setup_logger("page3", "prediction")

CONTRACT_TIER_KEY = "hk_contract_tier"


def _default_tier(session_manager: SessionManager) -> str:
    info = session_manager.get("user_info", {}) or {}
    tier = info.get("contract_tier")
    if tier:
        return str(tier)
    from src.utils.contract_config import get_tier_for_email

    return get_tier_for_email(None)


def resolve_contract(session_manager: SessionManager) -> PortfolioContract:
    tier = session_manager.get(CONTRACT_TIER_KEY) or _default_tier(session_manager)
    _logger.info("resolve_contract: tier=%s", tier)
    return PortfolioContract.from_tier(tier)


def _tier_label(contract: PortfolioContract) -> str:
    return f"{contract.label}（{contract.max_schools}所 / 退费{contract.refund_ratio * 100:.0f}%）"


def _sales_tier_label(contract: PortfolioContract) -> str:
    return f"{contract.label}（可申 {contract.max_schools} 所）"


def render_contract_picker(
    session_manager: SessionManager,
    *,
    key: str,
    label: str = "合同套餐",
    simple: bool = False,
) -> PortfolioContract:
    contracts = load_contracts()
    if not contracts:
        return resolve_contract(session_manager)

    tier_ids = list(contracts.keys())
    current = session_manager.get(CONTRACT_TIER_KEY) or _default_tier(session_manager)
    idx = tier_ids.index(current) if current in tier_ids else 0

    fmt = _sales_tier_label if simple else _tier_label
    chosen = st.selectbox(
        label,
        tier_ids,
        index=idx,
        format_func=lambda k: fmt(contracts[k]),
        key=key,
    )
    if chosen != session_manager.get(CONTRACT_TIER_KEY):
        session_manager.set(**{CONTRACT_TIER_KEY: chosen})
    return contracts[chosen]
