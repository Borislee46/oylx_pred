"""Portfolio — school combination evaluation, EV optimization, and Nash bargaining."""

from src.portfolio.config import (
    DEFAULT_PORTFOLIO_MODE,
    PRODUCTION_MODE,
    RISK_SWEEP,
    PortfolioMode,
)
from src.portfolio.expected_value import (
    EVDecomposition,
    all_reject_probability,
    calculate_prestige_score,
    decompose,
    ev_select,
    expected_best_prestige,
)
from src.portfolio.nash import (
    FrontierPoint,
    company_utility,
    frontier_and_nash,
    student_utility,
)
from src.portfolio.pool_builder import prediction_results_to_schools
from src.portfolio.portfolio_contract import (
    PortfolioContract,
    filter_pool_by_contract,
    load_contracts,
    portfolio_k_for_contract,
    target_universities_for_contract,
    university_region,
)
from src.portfolio.probability import (
    ComboMetrics,
    CopulaMcBackend,
    IndependentBackend,
    get_probability_backend,
)

__all__ = [
    "ComboMetrics",
    "CopulaMcBackend",
    "DEFAULT_PORTFOLIO_MODE",
    "EVDecomposition",
    "FrontierPoint",
    "IndependentBackend",
    "PRODUCTION_MODE",
    "PortfolioContract",
    "PortfolioMode",
    "RISK_SWEEP",
    "all_reject_probability",
    "calculate_prestige_score",
    "company_utility",
    "decompose",
    "ev_select",
    "expected_best_prestige",
    "filter_pool_by_contract",
    "frontier_and_nash",
    "get_probability_backend",
    "load_contracts",
    "portfolio_k_for_contract",
    "prediction_results_to_schools",
    "student_utility",
    "target_universities_for_contract",
    "university_region",
]
