# =============================================================================
# 配置文件：训练常量与业务约束
# ─────────────────────────────────────────────────────────────────────────────
# 本文件定义了机器学习训练流水线的所有硬编码常量和配置项。
# 每个配置项包含"为什么选这个值"的业务/技术理由。
#
# 面试要点：这些数字不是拍脑袋的。每个选择都有可辩护的理由。
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
# 特征工程配置
# ─────────────────────────────────────────────────────────────────────────────

# 训练时排除的列 — 文本详情列含未结构化信息，不直接作为 XGBoost 输入。
# 它们会在推理阶段通过 TF-IDF 文本提升模块单独处理（避免树模型对稀疏特征的次优表现）。
# faculty 排除因为它被用作后处理过滤规则而非模型特征。
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

# 用 log1p 变换的计数列 — 经历计数（0-10+）分布高度右偏：大部分学生0-2段，
# 极少数有5+段。log1p 将右偏分布拉近正态，减小极值对模型的影响。
# log1p (而非 log) 因为 log(0)= -inf，且计数常有0值。
COUNT_COLUMNS_FOR_LOG_TRANSFORM = [
    "research_count",
    "award_count",
    "internship_count",
    "paper_count",
]

TARGET_COLUMN = "admitted"
TEST_SIZE = 0.2
# 校准方法：sigmoid (Platt scaling)
# 为什么不是 isotonic？数据量不够大时 isotonic 容易过拟合（对每个区间独立拟合），
# sigmoid 只有两个参数 a, b，更稳健。录取场景校准集通常 ~数百样本，sigmoid 是安全选择。
CALIBRATION_METHOD = "sigmoid"
N_ITER = 100
# 预测阈值 0.24 — 不是默认的 0.5。
# 为什么？录取是偏态分布（正例 ~20-25%），阈值需匹配先验概率。
# 0.24 来自 threshold scan（在 evaluate_model 中做）找到的 F1 最优点。
# 面试要点：偏态分布下 threshold=0.5 是最常见的 DS 错误之一。
# 实际应用中，阈值还取决于业务偏好 — 宁可多申（高recall）还是精确推荐（高precision）。
DEFAULT_PREDICTION_THRESHOLD = 0.24
THRESHOLD_SCAN_STEPS = 101

# 文本为空的样本权重 — 背提文本缺失学生的信息量不如有完整文本的，
# 样本权重降为 0.85 防止模型过度关注信息不完整的样本。
TEXT_EMPTY_SAMPLE_WEIGHT = 0.85

# 时间衰减：最新 10000 例权重 ×1.1，缓解 concept drift（录取标准随时间变化）。
# 为什么不用指数衰减？录取年份不可靠（部分案例年份标注缺失），
# 用最近样本数做硬阈值更稳健。
RECENT_SAMPLE_BOOST_COUNT = 10000
RECENT_SAMPLE_BOOST_WEIGHT = 1.1

# 单调递增特征白名单 — 这些特征在录取场景下有明确的业务单调性：
# GPA、语言成绩、四段经历（科研/获奖/实习/论文）数量越多→录取概率越高。
# XGBoost monotone_constraints=1 确保模型输出不会违反这个先验知识。
# 面试要点：data-driven 的 ML 用 domain knowledge 做约束，这是 DS 的核心能力。
MONOTONE_INCREASING_WHITELIST = [
    "gpa",
    "language_score",
    "research_count",
    "award_count",
    "internship_count",
    "paper_count",
]
# 当前没有单调递减特征（如果能证明某项指标与录取明确负相关，可在此加入）。
MONOTONE_DECREASING_WHITELIST: list[str] = []
