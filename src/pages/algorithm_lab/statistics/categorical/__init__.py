"""分类数据分析模块"""

from .crosstabs import (
    CrosstabResult,
    KappaResult,
    GofResult,
    crosstab,
    chi_square_test,
    chi_square_gof,
    cohens_kappa,
    mantel_haenszel,
    chi_square_report,
    gof_report,
)

__all__ = [
    "CrosstabResult", "KappaResult", "GofResult",
    "crosstab", "chi_square_test", "chi_square_gof",
    "cohens_kappa", "mantel_haenszel",
    "chi_square_report", "gof_report",
]
