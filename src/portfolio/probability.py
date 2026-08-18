from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Protocol

import pandas as pd

from src.portfolio.config import (
    DEFAULT_PORTFOLIO_MODE,
    PortfolioMode,
)
from src.portfolio.prestige import school_prestige


def _prob(s: dict[str, Any]) -> float:
    p = float(s.get("probability", 0.0))
    return p if isfinite(p) else 0.0


@dataclass(frozen=True)
class ComboMetrics:
    ev_best_prestige: float
    p_all_reject: float


class ProbabilityBackend(Protocol):
    def combo_metrics(
        self,
        selected: list[dict[str, Any]],
        *,
        correlation_matrix: pd.DataFrame | None,
        pair_weight_matrix: pd.DataFrame | None,
        compute_ev: bool,
    ) -> ComboMetrics: ...


def _independent_metrics(
    selected: list[dict[str, Any]],
    *,
    compute_ev: bool,
) -> ComboMetrics:
    p_reject = 1.0
    for s in selected:
        p_reject *= 1.0 - _prob(s)

    if not compute_ev:
        return ComboMetrics(0.0, p_reject)

    ranked = sorted(selected, key=school_prestige, reverse=True)
    acc_reject = 1.0
    ev = 0.0
    for s in ranked:
        p = _prob(s)
        ev += school_prestige(s) * p * acc_reject
        acc_reject *= 1.0 - p
    return ComboMetrics(ev, p_reject)


class IndependentBackend:
    def combo_metrics(
        self,
        selected: list[dict[str, Any]],
        *,
        correlation_matrix: pd.DataFrame | None,
        pair_weight_matrix: pd.DataFrame | None,
        compute_ev: bool,
    ) -> ComboMetrics:
        if not selected:
            return ComboMetrics(0.0, 1.0)
        if len(selected) == 1:
            p = _prob(selected[0])
            ev = school_prestige(selected[0]) * p if compute_ev else 0.0
            return ComboMetrics(ev, 1.0 - p)
        return _independent_metrics(selected, compute_ev=compute_ev)


class CopulaMcBackend:
    def combo_metrics(
        self,
        selected: list[dict[str, Any]],
        *,
        correlation_matrix: pd.DataFrame | None,
        pair_weight_matrix: pd.DataFrame | None,
        compute_ev: bool,
    ) -> ComboMetrics:
        if not selected:
            return ComboMetrics(0.0, 1.0)
        if len(selected) == 1:
            p = _prob(selected[0])
            ev = school_prestige(selected[0]) * p if compute_ev else 0.0
            return ComboMetrics(ev, 1.0 - p)

        if correlation_matrix is None or correlation_matrix.empty:
            return _independent_metrics(selected, compute_ev=compute_ev)

        from src.portfolio.monte_carlo import run_monte_carlo_simulation

        prestige_scores = [school_prestige(s) for s in selected]
        p_reject, _, ev = run_monte_carlo_simulation(
            selected,
            correlation_matrix,
            pair_weight_matrix,
            school_prestige_scores=prestige_scores,
            compute_ev=compute_ev,
        )
        return ComboMetrics(float(ev or 0.0), float(p_reject))


_BACKENDS: dict[PortfolioMode, ProbabilityBackend] = {
    PortfolioMode.INDEPENDENT: IndependentBackend(),
    PortfolioMode.COPULA_MC: CopulaMcBackend(),
}


def get_probability_backend(mode: PortfolioMode | None = None) -> ProbabilityBackend:
    return _BACKENDS[mode or DEFAULT_PORTFOLIO_MODE]
