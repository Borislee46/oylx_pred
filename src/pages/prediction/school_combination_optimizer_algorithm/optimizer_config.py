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
    "safety": 0.25,
    "target": 0.4,
    "reach": 0.35,
}

ADAPTIVE_THRESHOLD_PERCENTILES = {
    "reach_percentile_val": 10,
    "safety_percentile_val": 70,
}

ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG = {
    "reach_percentile_val": 20,
    "safety_percentile_val": 70,
}

SAFETY_SCHOOL_THRESHOLD = 0.7
MIN_SAFETY_SCHOOL_COUNT_DEFAULT = 3
MIN_SAFETY_SCHOOL_COUNT_HIGH_BG = 1

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

MACAU_UNIVERSITIES = {
    "澳门大学",
    "澳门科技大学",
    "澳门城市大学",
    "澳门理工大学",
}

GLOBAL_MIN_SIMILARITY = 0.8888


MAJOR_SIMILARITY_WEIGHT = 2.5

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
