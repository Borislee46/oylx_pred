IRRELEVANT_COLUMNS = [
    "research_detail",
    "paper_detail",
    "internship_detail",
    "award_detail",
    "activity_detail",
    "activity_count",
    "background_major_original",
    "faculty",
]

CATEGORICAL_COLUMNS = [
    "target_university",
    "target_major",
    "background_university",
    "background_major",
]

TEXT_COLUMNS = [
    "research_detail",
    "paper_detail",
    "internship_detail",
    "award_detail",
    "activity_detail",
]

COUNT_COLUMNS_FOR_LOG_TRANSFORM = [
    "research_count",
    "award_count",
    "internship_count",
    "paper_count",
]

TARGET_COLUMN = "admitted"
TEST_SIZE = 0.2

HELD_OUT_FEATHER_PATH = "data/held_out_test.feather"
HELD_OUT_META_PATH = "data/held_out_test_meta.json"
HELD_OUT_SPLIT_YEAR = 2024
USE_HELD_OUT_IF_AVAILABLE = True
CALIBRATION_METHOD = "isotonic"
N_ITER = 100
DEFAULT_PREDICTION_THRESHOLD = 0.24
THRESHOLD_SCAN_STEPS = 101
TEXT_EMPTY_SAMPLE_WEIGHT = 0.85
RECENT_SAMPLE_BOOST_COUNT = 10000
RECENT_SAMPLE_BOOST_WEIGHT = 1.1

MONOTONE_INCREASING_WHITELIST = [
    "gpa",
    "language_score",
    "research_count",
    "award_count",
    "internship_count",
    "paper_count",
]
MONOTONE_DECREASING_WHITELIST: list[str] = []
