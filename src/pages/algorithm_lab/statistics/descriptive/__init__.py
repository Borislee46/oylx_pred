"""描述统计模块 — 描述性统计分析

复刻 SPSS 描述统计菜单:
- Frequencies (频率分析)
- Explore (探索性数据分析)
- P-P Plot (概率-概率图)
- Q-Q Plot (分位数-分位数图)
- Ratio Statistics (比率统计, IAAO 标准)
- Correlation (相关分析: Pearson/Spearman/Kendall, 多重比较校正, Fisher CI)
"""

from .frequencies import (
    FreqTable,
    DescriptiveStats,
    frequency_table,
    descriptive_stats,
    analyze,
)
from .explore import (
    NormalityResult,
    LeveneResult,
    BoxplotStats,
    GroupExploreResult,
    SpreadLevelResult,
    normality_tests,
    levene_test,
    boxplot_stats,
    stem_and_leaf,
    spread_level_analysis,
    explore_by_group,
    explore_summary_table,
)
from .pp_plot import (
    PPData,
    pp_plot,
    pp_plot_diagnose,
    PROPORTION_FORMULAS,
    SUPPORTED_DISTRIBUTIONS,
)
from .qq_plot import (
    QQData,
    qq_plot,
    qq_plot_diagnose,
)
from .ratio_stats import (
    RatioStatsResult,
    RatioConcentration,
    ratio_statistics,
    concentration_between,
    concentration_within_median_pct,
    ratio_report,
    iaao_cod_grade,
    iaao_prd_grade,
)
from .correlation import (
    CorrelationPairResult,
    CorrelationResult,
    correlation,
    correlation_matrix,
    correlation_report,
    correlation_pair_report,
)

__all__ = [
    # Frequencies
    "FreqTable",
    "DescriptiveStats",
    "frequency_table",
    "descriptive_stats",
    "analyze",
    # Explore
    "NormalityResult",
    "LeveneResult",
    "BoxplotStats",
    "GroupExploreResult",
    "SpreadLevelResult",
    "normality_tests",
    "levene_test",
    "boxplot_stats",
    "stem_and_leaf",
    "spread_level_analysis",
    "explore_by_group",
    "explore_summary_table",
    # P-P Plot
    "PPData",
    "pp_plot",
    "pp_plot_diagnose",
    "PROPORTION_FORMULAS",
    "SUPPORTED_DISTRIBUTIONS",
    # Q-Q Plot
    "QQData",
    "qq_plot",
    "qq_plot_diagnose",
    # Ratio Statistics
    "RatioStatsResult",
    "RatioConcentration",
    "ratio_statistics",
    "concentration_between",
    "concentration_within_median_pct",
    "ratio_report",
    "iaao_cod_grade",
    "iaao_prd_grade",
    # correlation
    "CorrelationPairResult",
    "CorrelationResult",
    "correlation",
    "correlation_matrix",
    "correlation_report",
    "correlation_pair_report",
]
