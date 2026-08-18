from __future__ import annotations

import logging
from typing import Any

from src.utils.config_models import ContractConfig

_logger = logging.getLogger(__name__)

_cached: ContractConfig | None = None


def _load() -> ContractConfig:
    global _cached
    if _cached is not None:
        return _cached
    _cached = ContractConfig.load()
    return _cached


def get_tier_for_email(email: str | None) -> str:
    cfg = _load()
    if email and email in cfg.email_to_tier:
        return cfg.email_to_tier[email]
    return cfg.default_tier


def get_tier_config(tier: str) -> dict:
    cfg = _load()
    return cfg.applications.get(tier) or cfg.applications.get(cfg.default_tier, {})


def get_contract_context(tier: str, format_type: str = "short") -> str:
    cfg = _load()
    t = cfg.applications.get(tier) or cfg.applications.get(cfg.default_tier, {})
    template = cfg.prompt_context_template.get(format_type, "")
    if not template or not t:
        return ""
    return template.format(
        label=t.get("label", t.get("name", "")),
        price=f"{t.get('price_cny', 0):,}",
        max_schools=t.get("max_schools", ""),
        max_majors=t.get("max_majors", ""),
        team=t.get("team", ""),
        regions="、".join(t.get("regions", [])),
        note=t.get("note", "无"),
    )


def get_applications(*, include_disabled: bool = False) -> dict[str, dict[str, Any]]:
    apps = _load().applications
    if include_disabled:
        return dict(apps)
    return {k: v for k, v in apps.items() if v.get("_enabled", True)}


def get_upgrades(*, include_disabled: bool = False) -> dict[str, dict[str, Any]]:
    upgs = _load().upgrades
    if include_disabled:
        return dict(upgs)
    return {k: v for k, v in upgs.items() if v.get("_enabled", True)}


def get_application(tier_id: str) -> dict[str, Any]:
    return _load().applications.get(tier_id, {})


def get_upgrade(product_id: str) -> dict[str, Any]:
    return _load().upgrades.get(product_id, {})


def get_default_tier() -> str:
    return _load().default_tier
