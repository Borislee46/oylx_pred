from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio.config import RISK_SWEEP
from src.portfolio.expected_value import EVDecomposition
from src.portfolio.optimizer import ev_select
from src.portfolio.portfolio_contract import PortfolioContract


def student_utility(d: EVDecomposition) -> float:
    return d.ev_best_prestige


def company_utility(d: EVDecomposition, contract: PortfolioContract) -> float:
    return 1.0 - d.p_all_reject * contract.refund_ratio


@dataclass
class FrontierPoint:
    risk_weight: float
    combo: list[dict[str, Any]]
    decomp: EVDecomposition
    u_student: float
    u_company: float


def frontier_and_nash(
    pool: list[dict[str, Any]],
    contract: PortfolioContract,
    *,
    similarity_floor: float = 0.0,
    correlation_matrix: pd.DataFrame | None = None,
    pair_weight_matrix: pd.DataFrame | None = None,
) -> tuple[list[FrontierPoint], int]:
    pts: list[FrontierPoint] = []
    for rw in RISK_SWEEP:
        combo, d = ev_select(
            pool,
            contract,
            risk_weight=rw,
            similarity_floor=similarity_floor,
            correlation_matrix=correlation_matrix,
            pair_weight_matrix=pair_weight_matrix,
            return_decomposition=True,
        )
        if not combo or d is None:
            continue
        pts.append(FrontierPoint(rw, combo, d, student_utility(d), company_utility(d, contract)))
    if not pts:
        return [], -1

    us_min, uc_min = 0.0, 0.0
    nash_idx = max(
        range(len(pts)),
        key=lambda i: (
            (pts[i].u_student - us_min) * (pts[i].u_company - uc_min),
            pts[i].u_student,
        ),
    )
    return pts, nash_idx
