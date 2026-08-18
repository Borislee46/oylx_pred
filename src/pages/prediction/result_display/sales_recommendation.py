from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.pages.prediction.result_display.product_registry import (
    ProductDef,
    ProductKind,
    UpliftMode,
    registry_by_name,
    selectable_product_names,
)
from src.utils.contract_config import get_application, get_upgrade

_POLICY_PATH = Path(__file__).resolve().parents[4] / "config" / "sales_blocks_policy.json"


def _cfg_to_product(cfg: dict) -> dict:
    variant = cfg.get("narrative", "")
    if not variant:
        services = cfg.get("services", [])
        variant = services[0] if services else ""
    return {
        "name": cfg.get("name", ""),
        "variant": variant,
        "scale": f"{cfg.get('max_schools', '')}所" if cfg.get("max_schools") else "",
        "price": (
            f"¥{cfg.get('price_cny', 0):,}"
            if isinstance(cfg.get("price_cny"), int)
            else cfg.get("price_cny", "")
        ),
        "dot": cfg.get("dot", "#94a3b8"),
    }


def load_sales_blocks_policy() -> dict[str, Any]:
    with open(_POLICY_PATH, encoding="utf-8") as f:
        return json.load(f)


_COMPETITIVE_UNIS: frozenset[str] = frozenset(
    {
        "香港大学",
        "香港中文大学",
        "香港科技大学",
        "香港城市大学",
        "香港理工大学",
        "香港浸会大学",
        "新加坡国立大学",
        "南洋理工大学",
        "新加坡管理大学",
        "澳门大学",
    }
)


def _pick_application_tier(input_data: dict, policy: dict) -> str:
    target_unis = input_data.get("target_universities", []) or []
    n = len(target_unis)
    competitive_n = sum(1 for u in target_unis if str(u) in _COMPETITIVE_UNIS)
    adjusted = n + competitive_n * 0.5

    apps = policy.get("application_by_target_n") or {}
    if adjusted >= float(apps.get("premium_c_min", 8)):
        return "premium_c"
    if adjusted >= float(apps.get("premium_b_min", 4)):
        return "premium_b"
    return str(apps.get("fallback", "joint_hkmo_sg"))


def _pick_application(input_data: dict, policy: dict) -> str:
    return get_application(_pick_application_tier(input_data, policy))["name"]


def build_default_blocks_selection(input_data: dict) -> list[str]:
    policy = load_sales_blocks_policy()
    by_name = registry_by_name()
    max_up = int(policy.get("max_default_upgrades", 2))
    exclude = set(policy.get("exclude_from_default", []))

    picked: list[str] = [_pick_application(input_data, policy)]
    already: set[str] = set(picked)
    candidates = selectable_product_names(input_data)

    scored: list[tuple[str, float, int]] = []
    for name in candidates:
        p = by_name.get(name)
        if not p or p.kind != ProductKind.UPGRADE or p.catalog_id in exclude:
            continue
        if name in already:
            continue
        diag = p.diagnose(input_data) if p.diagnose else None
        if diag is None:
            continue
        sev = p.severity(input_data) if p.severity else 0.0
        mode_boost = 0.10 if p.uplift_mode == UpliftMode.PIPELINE else 0.0
        scored.append((name, sev + mode_boost, p.sort_order))

    scored.sort(key=lambda x: (-x[1], x[2]))
    upgrades: list[str] = []
    for name, _score, _order in scored:
        if len(upgrades) >= max_up:
            break
        upgrades.append(name)
        already.add(name)

    if not upgrades:
        _maybe_catalog_fallback(picked, candidates, by_name, exclude, already, input_data, policy)

    return picked + upgrades


def _maybe_catalog_fallback(
    picked: list[str],
    candidates: list[str],
    by_name: dict[str, ProductDef],
    exclude: set[str],
    already: set[str],
    input_data: dict,
    policy: dict,
) -> None:
    gpa = float(input_data.get("gpa") or 0)
    lang = input_data.get("language_score_raw") or input_data.get("language_score")
    has_lang = lang is not None
    if not (gpa > 2.0 or has_lang):
        return

    for cat in policy.get("catalog_fallback", []):
        if cat in exclude:
            continue
        cfg = get_upgrade(cat)
        name = cfg.get("name", "")
        if name and name in candidates and name not in already:
            picked.append(name)
            return


def blocks_selection_to_product_dicts(names: list[str], input_data: dict) -> list[dict]:
    by_name = registry_by_name()
    result: list[dict] = []
    for name in names:
        p = by_name.get(name)
        if not p:
            continue
        if p.kind == ProductKind.APPLICATION:
            result.append(_cfg_to_product(get_application(p.catalog_id)))
        else:
            result.append(_cfg_to_product(get_upgrade(p.catalog_id)))
    return result


def build_matched_products(input_data: dict, has_cross: bool = False) -> list[dict]:
    return _build_products(input_data, has_cross)


def _build_products(input_data: dict, has_cross: bool) -> list[dict]:
    exp = input_data.get("experience_details") or {}
    try:
        gpa = float(input_data.get("gpa", 0) or 0)
    except (TypeError, ValueError):
        gpa = 0
    has_research = bool(exp.get("research"))
    has_internship = bool(exp.get("internship"))

    policy = load_sales_blocks_policy()
    exclude = set(policy.get("exclude_from_default", []))
    products: list[dict] = []

    app_tier = _pick_application_tier(input_data, policy)
    products.append(_cfg_to_product(get_application(app_tier)))

    if not has_research:
        if gpa >= 3.5:
            _try_append_upgrade(products, "research_r_plan", exclude)
        else:
            _try_append_upgrade(products, "bg_research", exclude)
    elif has_cross or gpa < 3.3:
        _try_append_upgrade(products, "academic_tutoring", exclude)

    if not has_internship:
        _try_append_upgrade(products, "bg_intern", exclude)

    _try_append_upgrade(products, "butler_plan", set())

    return products


def _try_append_upgrade(products: list[dict], catalog_id: str, exclude: set[str]) -> None:
    if catalog_id in exclude:
        return
    try:
        cfg = get_upgrade(catalog_id)
        if cfg:
            products.append(_cfg_to_product(cfg))
    except Exception:
        pass
