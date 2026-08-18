from __future__ import annotations

from enum import Enum
from types import MappingProxyType

MONTE_CARLO_DEFAULTS = MappingProxyType(
    {
        "n_simulations": 5000,
        "min_simulations": 1000,
        "max_simulations": 10000,
        "convergence_threshold": 0.01,
        "batch_size": 500,
    }
)

RISK_SWEEP = [round(i * 0.25, 2) for i in range(17)]


class PortfolioMode(str, Enum):
    INDEPENDENT = "independent"
    COPULA_MC = "copula_mc"


PRODUCTION_MODE = PortfolioMode.INDEPENDENT
DEFAULT_PORTFOLIO_MODE = PRODUCTION_MODE

_MULTI_PROGRAM_SCHOOLS: dict[str, int] = {
    "澳门大学": 2,
}


def max_programs_per_school(university: str) -> int:
    return _MULTI_PROGRAM_SCHOOLS.get(university, 1)
