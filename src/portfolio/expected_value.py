from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio.config import PortfolioMode
from src.portfolio.portfolio_contract import PortfolioContract
from src.portfolio.probability import get_probability_backend

_EXPORTS = frozenset(
    {
        "calculate_prestige_score",
        "ev_select",
        "FrontierPoint",
        "company_utility",
        "frontier_and_nash",
        "student_utility",
    }
)


@dataclass
class EVDecomposition:
    value: float | None
    ev_best_prestige: float
    p_all_reject: float
    refund_liability: float
    add_cost: float
    k: int


def combo_metrics(
    selected: list[dict[str, Any]],
    *,
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
    compute_ev: bool = True,
    mode: PortfolioMode | None = None,
) -> tuple[float, float]:
    backend = get_probability_backend(mode)
    result = backend.combo_metrics(
        selected,
        correlation_matrix=correlation_matrix,
        pair_weight_matrix=pair_weight_matrix,
        compute_ev=compute_ev,
    )
    return result.ev_best_prestige, result.p_all_reject


def expected_best_prestige(
    selected: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
    *,
    mode: PortfolioMode | None = None,
) -> float:
    ev, _ = combo_metrics(
        selected,
        correlation_matrix=correlation_matrix,
        pair_weight_matrix=pair_weight_matrix,
        mode=mode,
    )
    return ev


def all_reject_probability(
    selected: list[dict[str, Any]],
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
    *,
    mode: PortfolioMode | None = None,
) -> float:
    _, p_reject = combo_metrics(
        selected,
        correlation_matrix=correlation_matrix,
        pair_weight_matrix=pair_weight_matrix,
        mode=mode,
    )
    return p_reject


def decompose(
    selected: list[dict[str, Any]],
    contract: PortfolioContract,
    *,
    risk_weight: float,
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
    compute_ev: bool = True,
    mode: PortfolioMode | None = None,
) -> EVDecomposition:
    if not selected:
        return EVDecomposition(0.0, 0.0, 1.0, float(contract.reject_deduction), 0.0, 0)

    ev_prestige, p_reject = combo_metrics(
        selected,
        correlation_matrix=correlation_matrix,
        pair_weight_matrix=pair_weight_matrix,
        compute_ev=compute_ev,
        mode=mode,
    )
    refund_liability = p_reject * contract.reject_deduction

    extra = max(0, len(selected) - contract.min_schools)
    add_cost_ratio = contract.add_school_cost / contract.price_cny if contract.price_cny else 0.0
    add_cost = add_cost_ratio * extra

    if compute_ev:
        value = round(ev_prestige - risk_weight * p_reject * contract.refund_ratio - add_cost, 4)
    else:
        value = None
    return EVDecomposition(
        value=value,
        ev_best_prestige=round(ev_prestige, 4),
        p_all_reject=round(p_reject, 4),
        refund_liability=round(refund_liability, 1),
        add_cost=round(add_cost, 4),
        k=len(selected),
    )


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name == "calculate_prestige_score":
        from src.portfolio.prestige import calculate_prestige_score

        return calculate_prestige_score
    if name == "ev_select":
        from src.portfolio.optimizer import ev_select

        return ev_select
    from src.portfolio import nash as _nash

    return getattr(_nash, name)
