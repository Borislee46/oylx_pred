from pathlib import Path

# 文本加成配置
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

# LogitUpliftProvider 默认值
LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN: float = 0.25
LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN: float = 0.22
LOGIT_UPLIFT_DEFAULT_SMOOTHING: float = 0.5
LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR: float = 0.4
LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA: float = 1.0

# 概率调整器配置
GPA_MINIMUM: float = 2.0
LANGUAGE_MINIMUM: float = 0.6

# GPA 惩罚配置
GPA_PENALTY_SEVERE_THRESHOLD: float = 0.95  # GPA低于最小值时的惩罚系数
GPA_PENALTY_MAX_COEFFICIENT: float = 0.8  # GPA惩罚的最大值
GPA_PENALTY_QUADRATIC_COEFFICIENT: float = 0.15  # GPA惩罚的二次项系数

# 语言分数惩罚配置
LANGUAGE_PENALTY_SEVERE_THRESHOLD: float = 0.95  # 语言分数低于最小值时的惩罚系数
LANGUAGE_PENALTY_PASS_LINE_MULTIPLIER: float = 0.5  # 语言分数及格线的标准差倍数
LANGUAGE_PENALTY_LEVEL_1_THRESHOLD: float = 0.85  # 语言分数惩罚级别1
LANGUAGE_PENALTY_LEVEL_1_MULTIPLIER: float = 1.5  # 语言分数惩罚级别1的标准差倍数
LANGUAGE_PENALTY_LEVEL_2_THRESHOLD: float = 0.7  # 语言分数惩罚级别2
LANGUAGE_PENALTY_LEVEL_2_MULTIPLIER: float = 1.0  # 语言分数惩罚级别2的标准差倍数
LANGUAGE_PENALTY_LEVEL_3_THRESHOLD: float = 0.4  # 语言分数惩罚级别3

# 概率调整配置
PROBABILITY_MIN_VALUE: float = 0.001  # 概率的最小值
PROBABILITY_ADJUSTMENT_THRESHOLD: float = 0.01  # 调整后的概率阈值
PROBABILITY_EXTREME_STD_MULTIPLIER: float = 2.0  # 极端值的标准差倍数

# 跨专业惩罚配置
CROSS_MAJOR_PENALTY_FACTOR: float = 0.5

# 职业型专业配置
PROFESSIONAL_MAJORS: list[str] = ["Business Administration", "MBA"]
PROFESSIONAL_REDUCTION_FACTOR: float = 0.30
PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR: float = 0.50

# 相似度阈值配置
MIN_SIMILARITY_THRESHOLD: float = 0.89
HIGHER_SIMILARITY_THRESHOLD: float = 0.91
UNIVERSITY_COUNT_THRESHOLD: int = 2
CROSS_MAJOR_SIMILARITY_MIN: float = 0.8  # 跨专业推荐的最小相似度阈值

# 推荐配置
TOP_N_RECOMMENDATIONS: int = 50

USER_SPECIFIED_SMALL_RANGE_THRESHOLD: int = 20
USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD: int = 100
USER_SPECIFIED_MEDIUM_RANGE_TOP_N: int = 50
USER_SPECIFIED_LARGE_RANGE_TOP_N: int = 100

# 缓存配置
PROBABILITY_ADJUSTER_CACHE_SIZE: int = 50

# 相似度调整规则配置文件路径
SIMILARITY_ADJUSTMENT_RULES_PATH: Path = Path("config/similarity_adjustment_rules.json")

# LogitUpliftProvider 质量分数配置
QUALITY_SCORE_MAX_WEIGHT: float = 0.7  # 质量分数中最大值的权重
QUALITY_SCORE_MEAN_WEIGHT: float = 0.3  # 质量分数中均值的权重
QUALITY_SCORE_THRESHOLD: float = 0.15  # 质量分数显示阈值

# LogitUpliftProvider 概率范围配置
PROBABILITY_BOOST_MIN: float = 0.1  # 应用boost的最小概率
PROBABILITY_BOOST_MAX: float = 0.9  # 应用boost的最大概率
PROBABILITY_SCALE_CENTER: float = 0.5  # 概率缩放的中心点
PROBABILITY_SCALE_FACTOR: float = 2.0  # 概率缩放因子

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
