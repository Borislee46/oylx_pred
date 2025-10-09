DEFAULT_TEXT_BOOST_CONFIG: dict = {
    "enabled": True,
    "max_total_boost": 0.05,
    "timeout_ms": 100,
    "similarity_thresholds": [[0.15, 0.05]],
    "model_paths": {
        "tfidf_vectorizer": "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib",
        "tfidf_centroids": "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz",
    },
}

GPA_MINIMUM: float = 2.0
LANGUAGE_MINIMUM: float = 0.6

CROSS_MAJOR_PENALTY_FACTOR: float = 0.5

PROFESSIONAL_MAJORS: list[str] = ["Business Administration", "MBA"]
PROFESSIONAL_REDUCTION_FACTOR: float = 0.70
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR: float = 0.85

MIN_SIMILARITY_THRESHOLD: float = 0.89
HIGHER_SIMILARITY_THRESHOLD: float = 0.92
UNIVERSITY_COUNT_THRESHOLD: int = 5

TOP_N_RECOMMENDATIONS: int = 50

PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50
KEYWORD_BOOSTER_CACHE_SIZE: int = 1000
