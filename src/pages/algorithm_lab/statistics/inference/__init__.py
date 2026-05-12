"""推断统计模块 — 假设检验与参数估计

复刻 SPSS 推断统计菜单:
- t_test: 独立样本/配对/单样本 t 检验 (Welch + Student) + Cohen's d + BF
- anova: 单因素 ANOVA (Classic + Welch) + 事后检验 (Tukey/Bonferroni/Games-Howell/Dunnett)
- nonparametric: 非参数检验 (M-W U / K-W / Wilcoxon / Friedman / J-T / McNemar / Cochran's Q)
- bootstrap: Bootstrap 自助法 (CI, BCa, 分层)
- regression_ols: 线性回归 (OLS) + VIF + 残差诊断
- regression_logistic: 二元 Logistic 回归 (MLE/Newton-Raphson, OR, AUC, Hosmer-Lemeshow)
- proportion: 比例检验 (单样本/独立/配对, Wilson/Agresti-Coull/Clopper-Pearson)
- power_analysis: 统计效力分析 (t/ANOVA/χ²/相关, 样本量↔效力)
- regression_count: 计数回归 (Poisson/NB, IRR, 过离散检验)
"""

from .t_test import (
    TTestResult,
    independent_t_test,
    paired_t_test,
    one_sample_t_test,
    cohens_d,
    hedges_g,
    glass_delta,
    t_test_report,
)
from .anova import (
    AnovaResult,
    PostHocResult,
    PostHocComparison,
    RMAnovaResult,
    oneway_anova,
    welch_anova,
    repeated_measures_anova,
    tukey_hsd,
    bonferroni,
    games_howell,
    lsd,
    sidak,
    scheffe,
    dunnett,
    posthoc,
    linear_trend_test,
    anova_report,
    posthoc_report,
    rm_anova_report,
    POSTHOC_GUIDE,
)
from .nonparametric import (
    NonparametricResult,
    mann_whitney,
    kruskal_wallis,
    wilcoxon_signed_rank,
    sign_test,
    friedman,
    jonckheere_terpstra,
    ks_two_sample,
    mcnemar,
    cochran_q,
)
from .bootstrap import (
    BootstrapResult,
    bootstrap,
    bootstrap_mean,
    bootstrap_median,
    bootstrap_correlation,
    bootstrap_stratified,
    bootstrap_report,
)
from .regression_ols import (
    OLSCoefficient,
    OLSResult,
    HierarchicalStep,
    ols,
    hierarchical_regression,
    ols_report,
)
from .proportion import (
    ProportionCI,
    OneSampleProportionResult,
    TwoSampleProportionResult,
    one_sample_proportion,
    two_sample_proportion,
    paired_proportion_ci,
    one_sample_report,
    two_sample_report,
    CI_METHODS,
)
from .regression_logistic import (
    LogisticCoefficient,
    LogisticResult,
    logistic_regression,
    logistic_regression_report,
)
from .power_analysis import (
    PowerResult,
    power_t_test,
    power_t_test_paired,
    power_anova,
    power_chi_square,
    power_correlation,
    power_report,
)
from .regression_count import (
    CountCoefficient,
    CountRegressionResult,
    poisson_regression,
    negative_binomial_regression,
    count_regression_report,
)

__all__ = [
    # t_test
    "TTestResult",
    "independent_t_test",
    "paired_t_test",
    "one_sample_t_test",
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "t_test_report",
    # anova
    "AnovaResult",
    "PostHocResult",
    "PostHocComparison",
    "RMAnovaResult",
    "oneway_anova",
    "welch_anova",
    "repeated_measures_anova",
    "tukey_hsd",
    "bonferroni",
    "games_howell",
    "lsd",
    "sidak",
    "scheffe",
    "dunnett",
    "posthoc",
    "linear_trend_test",
    "anova_report",
    "posthoc_report",
    "rm_anova_report",
    "POSTHOC_GUIDE",
    # nonparametric
    "NonparametricResult",
    "mann_whitney",
    "kruskal_wallis",
    "wilcoxon_signed_rank",
    "sign_test",
    "friedman",
    "jonckheere_terpstra",
    "ks_two_sample",
    "mcnemar",
    "cochran_q",
    # bootstrap
    "BootstrapResult",
    "bootstrap",
    "bootstrap_mean",
    "bootstrap_median",
    "bootstrap_correlation",
    "bootstrap_stratified",
    "bootstrap_report",
    # regression_ols
    "OLSCoefficient",
    "OLSResult",
    "HierarchicalStep",
    "ols",
    "hierarchical_regression",
    "ols_report",
    # proportion
    "ProportionCI",
    "OneSampleProportionResult",
    "TwoSampleProportionResult",
    "one_sample_proportion",
    "two_sample_proportion",
    "paired_proportion_ci",
    "one_sample_report",
    "two_sample_report",
    "CI_METHODS",
    # regression_logistic
    "LogisticCoefficient",
    "LogisticResult",
    "logistic_regression",
    "logistic_regression_report",
    # power_analysis
    "PowerResult",
    "power_t_test",
    "power_t_test_paired",
    "power_anova",
    "power_chi_square",
    "power_correlation",
    "power_report",
    # regression_count
    "CountCoefficient",
    "CountRegressionResult",
    "poisson_regression",
    "negative_binomial_regression",
    "count_regression_report",
]
