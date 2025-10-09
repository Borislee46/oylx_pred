SCHOOL_CATEGORY_THRESHOLDS = {
    "safety": 0.8,
    "target_lower": 0.6,
}

BALANCE_RATIOS = {
    "safety": 0.3,
    "target": 0.4,
    "reach": 0.3,
}

BALANCE_RATIOS_HIGH_BG = {
    "safety": 0.2,
    "target": 0.4,
    "reach": 0.4,
}

ADAPTIVE_THRESHOLD_PERCENTILES = {
    "reach_percentile_val": 10,
    "safety_percentile_val": 70,
}

ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG = {
    "reach_percentile_val": 15,
    "safety_percentile_val": 85,
}

MONTE_CARLO_DEFAULTS = {
    "n_simulations": 5000,
    "min_simulations": 1000,
    "max_simulations": 10000,
    "convergence_threshold": 0.01,
    "batch_size": 500,
}

TOP3_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]

TOP5_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港理工大学",
    "香港城市大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]

TOP8_SCHOOLS = [
    "香港大学",
    "香港中文大学",
    "香港科技大学",
    "香港城市大学",
    "香港理工大学",
    "香港浸会大学",
    "香港教育大学",
    "香港岭南大学",
    "澳门大学",
    "新加坡南洋理工大学",
    "新加坡国立大学",
]


CROSS_MAJOR_RECALL_FILTER = {
    "same_group_similarity_min": 0.75,
    "cross_group_similarity_min": 0.93,
    "strict_extra": 0.05,
    "global_min_similarity": 0.85,
}


CROSS_MAJOR_CALIBRATION = {
    "prior_p": 0.1,
    "shrinkage_alpha": 0.3,
    "clip_quantile": 0.9,
    "same_group_multiplier": 0.85,
    "cross_group_multiplier": 0.6,
    "strict_multiplier": 0.7,
}

MAJOR_SIMILARITY_WEIGHT = 2.5

SAME_GROUP_MIN_RATIO = 0.7

CONSTRAINT_FLEXIBILITY: dict[str, bool | float | int] = {}

PRESTIGE_WEIGHT = 3.0

MIN_TOP3_COUNT_FOR_HIGH_BG = 2
MIN_TOP5_COUNT_FOR_HIGH_BG = 3

OBJECTIVE_WEIGHTS = {
    "rejection_probability": 1.0,
    "diversity": 2.5,
    "balance_score": 2.0,
    "major_similarity": 1.0,
    "new_major_ratio": 0.5,
    "major_category_score": 1.0,
}

TOP_BG_LEVELS_SET = {"985", "211", "1-50", "51-100"}

PRIORITY_THRESHOLD_TOP_BG_GPA_GE_3_2 = 5
PRIORITY_THRESHOLD_TOP_BG_GPA_GE_2_8 = 6
PRIORITY_THRESHOLD_TOP_BG_DEFAULT = 7

PRIORITY_THRESHOLD_NORMAL_BG_GPA_GE_3_0 = 5
PRIORITY_THRESHOLD_NORMAL_BG_DEFAULT = 7
