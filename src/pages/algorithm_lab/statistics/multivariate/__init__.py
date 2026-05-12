"""多变量分析模块 — 降维与测量评估"""

from .factor_analysis import (
    FactorResult, factor_analysis, kmo_test, bartlett_test,
    parallel_analysis, factor_report,
)
from .reliability import (
    AlphaResult, ICCResult, FleissKappaResult,
    cronbach_alpha, icc, fleiss_kappa,
    alpha_report, icc_report, fleiss_report,
)
from .discriminant import (
    LDAResult, lda,
)

__all__ = [
    "FactorResult", "factor_analysis", "kmo_test", "bartlett_test",
    "parallel_analysis", "factor_report",
    "AlphaResult", "ICCResult", "FleissKappaResult",
    "cronbach_alpha", "icc", "fleiss_kappa",
    "alpha_report", "icc_report", "fleiss_report",
    "LDAResult", "lda",
]
