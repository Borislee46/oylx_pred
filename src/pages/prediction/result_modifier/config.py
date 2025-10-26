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
HIGHER_SIMILARITY_THRESHOLD: float = 0.92
UNIVERSITY_COUNT_THRESHOLD: int = 5

TOP_N_RECOMMENDATIONS: int = 50

PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50
