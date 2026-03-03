import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _get_project_root() -> Path:
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[2]


PROJECT_ROOT: Path = _get_project_root()

DEFAULT_TEXT_BOOST_CONFIG: dict = {
    "enabled": True,
    "max_total_boost": 0.15,
    "sim_gate_sum_min": 0.10,
    "sim_gate_max_min": 0.08,
    "smoothing": 0.7,
    "cap_min_factor": 0.10,
    "cap_quality_gamma": 1.2,
    "high_signal": {
        "enabled": True,
        "lexicon_path": str(PROJECT_ROOT / "config" / "text_high_signal_terms.json"),
        "lexicon_weight": 1.0,
        "novelty_weight": 0.12,
        "novelty_min_chars": 12,
        "bonus_cap_per_field": 0.6,
        "max_reasons": 3,
    },
    "model_paths": {
        "tfidf_vectorizer": str(
            PROJECT_ROOT
            / "src"
            / "machine_learning_models"
            / "pre-trained_models"
            / "tfidf_vectorizer.joblib"
        ),
        "tfidf_centroids": str(
            PROJECT_ROOT
            / "src"
            / "machine_learning_models"
            / "pre-trained_models"
            / "tfidf_centroids.npz"
        ),
        "text_uplift_weights": str(
            PROJECT_ROOT
            / "src"
            / "machine_learning_models"
            / "pre-trained_models"
            / "text_uplift_weights.json"
        ),
    },
}

LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN: float = 0.25
LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN: float = 0.22
LOGIT_UPLIFT_DEFAULT_SMOOTHING: float = 0.5
LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR: float = 0.4
LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA: float = 1.0
GPA_MINIMUM: float = 2.0
GPA_PENALTY_SEVERE_THRESHOLD: float = 0.95
GPA_PENALTY_MAX_COEFFICIENT: float = 0.8
GPA_PENALTY_QUADRATIC_COEFFICIENT: float = 0.15
LANGUAGE_MINIMUM: float = 0.6
LANGUAGE_PENALTY_SEVERE_THRESHOLD: float = 0.95
LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER: float = 0.5
LANGUAGE_PENALTY_LEVEL_1_THRESHOLD: float = 0.85
LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER: float = 1.5
LANGUAGE_PENALTY_LEVEL_2_THRESHOLD: float = 0.7
LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER: float = 1.0
LANGUAGE_PENALTY_LEVEL_3_THRESHOLD: float = 0.4
PROBABILITY_MIN_VALUE: float = 0.001
PROBABILITY_ADJUSTMENT_THRESHOLD: float = 0.01
PROBABILITY_EXTREME_STD_MULTIPLIER: float = 2.0
CROSS_MAJOR_PENALTY_FACTOR: float = 0.5
FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR: float = 0.3
COMPREHENSIVE_SCORE_BOOST_THRESHOLD: float = 0.6
SELECTION_SCORE_BOOST_FACTOR: float = 0.3
PROBABILITY_SCALE_FACTOR: float = 2.0
PREDICTION_RULES_PATH: Path = PROJECT_ROOT / "config" / "prediction_rules.json"
UNIVERSITY_DIFFICULTY_CONFIG_PATH: Path = PREDICTION_RULES_PATH


def load_prediction_rules() -> dict:
    if PREDICTION_RULES_PATH.exists():
        try:
            with open(PREDICTION_RULES_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_rules = load_prediction_rules()

PROFESSIONAL_MAJORS: list[str] = _rules.get(
    "PROFESSIONAL_MAJORS", ["Business Administration", "MBA"]
)
PROFESSIONAL_MAJORS_LOWER: list[str] = [m.lower() for m in PROFESSIONAL_MAJORS]
PROFESSIONAL_REDUCTION_FACTOR: float = 0.30
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR: float = 0.50
MIN_SIMILARITY_THRESHOLD: float = 0.89
HIGHER_SIMILARITY_THRESHOLD: float = 0.92
FUZZY_BIAS_THRESHOLD_HIGH: float = 92.0
FUZZY_BIAS_THRESHOLD_MID: float = 82.0
FUZZY_BIAS_THRESHOLD_LOW: float = 72.0
FUZZY_BIAS_MULTIPLIER_HIGH: float = 1.25
FUZZY_BIAS_MULTIPLIER_MID: float = 1.15
FUZZY_BIAS_MULTIPLIER_LOW: float = 1.05
UNIVERSITY_COUNT_THRESHOLD: int = 2
CROSS_MAJOR_SIMILARITY_MIN: float = 0.8
TOP_N_RECOMMENDATIONS: int = 30
USER_SPECIFIED_SMALL_RANGE_THRESHOLD: int = 20
USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD: int = 100
USER_SPECIFIED_MEDIUM_RANGE_TOP_N: int = 50
USER_SPECIFIED_LARGE_RANGE_TOP_N: int = 100
PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50
SIMILARITY_ADJUSTMENT_RULES_PATH: Path = (
    PROJECT_ROOT / "config" / "similarity_adjustment_rules.json"
)
QUALITY_SCORE_MAX_WEIGHT: float = 0.7
QUALITY_SCORE_MEAN_WEIGHT: float = 0.3
QUALITY_SCORE_THRESHOLD: float = 0.15
PROBABILITY_BOOST_MIN: float = 0.1
PROBABILITY_BOOST_MAX: float = 0.9
PROBABILITY_SCALE_CENTER: float = 0.5

DEFAULT_UNIVERSITY_DIFFICULTY_ORDER: tuple[str, ...] = tuple(
    _rules.get(
        "UNIVERSITY_DIFFICULTY_ORDER",
        (
            "新加坡国立大学",
            "新加坡南洋理工大学",
            "香港大学",
            "香港中文大学",
            "香港科技大学",
            "新加坡管理大学",
            "马来亚大学",
            "香港理工大学",
            "香港城市大学",
            "马来西亚理科大学",
            "马来西亚博特拉大学",
            "香港浸会大学",
            "马来西亚国立大学",
            "澳门大学",
            "香港中文大学 (深圳校区)",
            "澳门科技大学",
            "澳门城市大学",
            "澳门理工大学",
            "香港教育大学",
            "香港岭南大学",
            "香港都会大学",
            "香港恒生大学",
            "香港珠海学院",
        ),
    )
)
AGENT_MIN_SAFE_RELAX_THRESHOLD: float = 0.87
AGENT_BOUNDARY_SIMILARITY_RANGE: float = 0.03
AGENT_MAX_BOUNDARY_CASES: int = 20
AGENT_TAIL_PERCENTAGE: float = 0.2
AGENT_MIN_BALANCE_DIFF_MIN: int = 3
AGENT_MIN_BALANCE_DIFF_RATIO: float = 0.15
AGENT_NO_CHANGE_THRESHOLD: int = 3
LANGUAGE_REQUIREMENT_PENALTY_STEEPNESS: float = 7.0
LANGUAGE_REQUIREMENT_PENALTY_MIDPOINT: float = 0.5
MAX_TOTAL_PENALTY_RATIO: float = 0.7
MAX_TOTAL_BOOST_RATIO: float = 0.3
PENALTY_DECAY_FACTOR: float = 0.85
BOOST_DECAY_FACTOR: float = 0.8
ARBITRATION_MIN_PROBABILITY: float = 0.005
