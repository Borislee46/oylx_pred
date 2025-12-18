GPA_SCALES = {
    "4.0": {"max": 4.0, "step": 0.1, "format": "%.2f"},
    "4.3": {"max": 4.3, "step": 0.1, "format": "%.2f"},
    "4.5": {"max": 4.5, "step": 0.1, "format": "%.2f"},
    "5.0": {"max": 5.0, "step": 0.1, "format": "%.2f"},
    "10": {"max": 10.0, "step": 0.5, "format": "%.1f"},
    "20": {"max": 20.0, "step": 0.5, "format": "%.1f"},
    "100": {"max": 100.0, "step": 1.0, "format": "%.0f"},
}

DEFAULT_GPA_SCALE = "4.0"

LANGUAGE_TYPES = ["雅思", "托福"]

LANGUAGE_SCORE_RANGES = {
    "雅思": {"min": 0.0, "max": 9.0, "step": 0.5, "format": "%.1f"},
    "托福": {"min": 0, "max": 120, "step": 1, "format": "%d"},
}

TARGET_COUNTRY_UNIVERSITY_MAP = {
    "中国香港": [
        "香港大学",
        "香港中文大学",
        "香港科技大学",
        "香港理工大学",
        "香港城市大学",
        "香港中文大学 (深圳校区)",
        "香港浸会大学",
        "香港岭南大学",
        "香港教育大学",
        "香港都会大学",
        "香港恒生大学",
        "香港珠海学院",
    ],
    "新加坡": ["新加坡南洋理工大学", "新加坡国立大学", "新加坡管理大学"],
    "中国澳门": ["澳门大学", "澳门科技大学", "澳门理工大学", "澳门城市大学"],
    "马来西亚": [
        "马来亚大学",
        "马来西亚博特拉大学",
        "马来西亚理科大学",
        "马来西亚国立大学",
    ],
}

UNIVERSITY_SORT_ORDER = [
    uni
    for country in TARGET_COUNTRY_UNIVERSITY_MAP
    for uni in TARGET_COUNTRY_UNIVERSITY_MAP[country]
]

TARGET_COUNTRIES = list(TARGET_COUNTRY_UNIVERSITY_MAP.keys())

GPA_WARNING_THRESHOLDS = {
    "4.0": 2,
    "4.3": 2.15,
    "4.5": 2.25,
    "5.0": 2.5,
    "10": 5.0,
    "20": 10.0,
    "100": 50.0,
}

LANGUAGE_WARNING_THRESHOLDS = {"雅思": 5.5, "托福": 72}

DEFAULT_LANGUAGE_SCORES = {
    "雅思": 6.5,
    "托福": 90,
}

STANDARDIZED_TEST_TYPES = ["GRE", "GMAT"]

GRE_SCORE_RANGE = {"min": 260, "max": 340, "step": 1, "format": "%d"}
GMAT_SCORE_RANGE = {"min": 200, "max": 800, "step": 10, "format": "%d"}

GRE_BONUS_THRESHOLD = 311
GRE_SIGMOID_MIDPOINT = 325
GRE_SIGMOID_STEEPNESS = 0.5
GRE_MAX_BONUS = 0.5

GMAT_BONUS_THRESHOLD = 611
GMAT_SIGMOID_MIDPOINT = 700
GMAT_SIGMOID_STEEPNESS = 0.05
GMAT_MAX_BONUS = 0.5
