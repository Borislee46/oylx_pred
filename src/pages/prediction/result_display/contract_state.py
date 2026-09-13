from __future__ import annotations

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
