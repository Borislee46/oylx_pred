from __future__ import annotations

import hashlib
from dataclasses import dataclass

from src.pages.prediction.result_display.product_meta import split_selection
from src.pages.prediction.result_display.product_registry import resolve_attribution
from src.pages.prediction.result_display.sales_recommendation import load_sales_blocks_policy


@dataclass(frozen=True)
class SalesSnapshot:
    blocks_selection: tuple[str, ...]
    base_pct: int
    display_pct: int
    upgrade_contribs: dict[str, float]
    application_name: str | None
    upgrade_names: tuple[str, ...]
    positioning: str


def blocks_selection_fingerprint(names: list[str]) -> str:
    return hashlib.md5("|".join(sorted(names)).encode()).hexdigest()


def build_sales_snapshot(
    input_data: dict,
    selected: list[str],
    precomputed: dict,
    *,
    contract_tier: str = "",
) -> SalesSnapshot:
    _ = input_data, contract_tier
    app, upgrades = split_selection(selected)
    base_pct = int(precomputed.get("base_pct", 0) or 0)
    attr = resolve_attribution(
        selected,
        base_pct=base_pct,
        combos=precomputed.get("combos", {}),
        personalized_fixed_pp=precomputed.get("personalized_fixed_pp"),
    )
    contribs = attr.get("contributions") or attr.get("contrib") or {}
    policy = load_sales_blocks_policy()
    return SalesSnapshot(
        blocks_selection=tuple(selected),
        base_pct=base_pct,
        display_pct=int(attr["final_pct"]),
        upgrade_contribs={k: float(v) for k, v in contribs.items()},
        application_name=app,
        upgrade_names=tuple(upgrades),
        positioning=str(policy.get("positioning", "application_assist")),
    )
