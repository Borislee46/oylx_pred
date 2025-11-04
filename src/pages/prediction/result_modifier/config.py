DEFAULT_TEXT_BOOST_CONFIG: dict = {
    "enabled": True,
    "max_total_boost": 0.15,
    "sim_gate_sum_min": 0.10,
    "sim_gate_max_min": 0.08,
    "smoothing": 0.7,
    "cap_min_factor": 0.10,
    "cap_quality_gamma": 1.2,
    "model_paths": {
        "tfidf_vectorizer": "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib",
        "tfidf_centroids": "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz",
        "text_uplift_weights": "src/machine_learning_models/pre-trained_models/text_uplift_weights.json",
    },
}

GPA_MINIMUM: float = 2.0
LANGUAGE_MINIMUM: float = 0.6

CROSS_MAJOR_PENALTY_FACTOR: float = 0.5

PROFESSIONAL_MAJORS: list[str] = ["Business Administration", "MBA"]
PROFESSIONAL_REDUCTION_FACTOR: float = 0.30
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR: float = 0.50

MIN_SIMILARITY_THRESHOLD: float = 0.89
HIGHER_SIMILARITY_THRESHOLD: float = 0.91
UNIVERSITY_COUNT_THRESHOLD: int = 2

TOP_N_RECOMMENDATIONS: int = 50

USER_SPECIFIED_SMALL_RANGE_THRESHOLD: int = 20
USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD: int = 100
USER_SPECIFIED_MEDIUM_RANGE_TOP_N: int = 50
USER_SPECIFIED_LARGE_RANGE_TOP_N: int = 100

PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50

UNIVERSITY_DIFFICULTY_ORDER = [
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
]
