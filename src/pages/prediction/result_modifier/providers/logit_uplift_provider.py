"""
背提文本"含金量"核心算法实现, 对count类进行加成

Q: "含金量"该怎么定义？
A: 这个概念很抽象，很难具象化成一个固定的公式。单纯的关键词匹配过于单薄，比如："一区SCI论文第一作者"这种背景固然很厉害，但是实际对申请成功率的影响可能并不大。
所以需要结合文本的丰富度、新颖度、质量等多个维度来综合评估，且要结合真正的历史数据来训练模型，而不是仅仅依赖于关键词匹配。

Q: 为啥不用LLM？
A: 因为LLM的计算速度太慢了，而且计算成本太高了，而且计算结果的准确性也不高（1是幻觉，2是本升硕并没有绝对约定熟成的背题要求不像是申请博士那样需要有特定的背题要求），所以不用LLM。

Q: 为啥不塞到XGB一起训练？
A: 树模型训文本光特征工程就够我吃一壶的了，还有当前无文本的训练权重30mb加文本恐怕压不住，且短期来看用户使用频率较低，先这么着吧。
"""

from __future__ import annotations

from typing import Any

from src.pages.prediction.result_modifier.config import (
    LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR,
    LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA,
    LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN,
    LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN,
    LOGIT_UPLIFT_DEFAULT_SMOOTHING,
)
from src.pages.prediction.result_modifier.providers.logit_uplift import (
    DeltaCalculator,
    ProbabilityApplier,
    SimilarityComputer,
    TextProcessor,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    get_model_loader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.signal_scorer import (
    SignalScorer,
)
from src.pages.prediction.result_modifier.text_boost_provider import TextBoostProvider
from src.pages.prediction.result_modifier.utils import has_any_experience
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class LogitUpliftProvider(TextBoostProvider):
    """
    文本质量加成处理器v2.6 - 通过Logit空间转换实现概率提升

    组件流水线：
    TextProcessor → 特征提取 → 相似度计算 → 线性模型 → ProbabilityApplier

    处理流程：
    1. 文本解析: 提取背景描述特征和经历数量
    2. 模型加载: 惰性加载TF-IDF向量器、聚类中心点、回归权重
    3. 相似度评估: 计算与高质量案例的向量距离 + 关键词信号融合
    4. 增量计算: 基于相似度得分，通过线性模型计算Logit空间偏移量(Δ)
    5. 概率转换: 将Δ通过sigmoid函数映射为概率提升，并应用质量自适应封顶

    关键技术：
    - Logit空间操作: 在log-odds空间进行线性加成，避免概率值越界[0,1]
    - 质量敏感封顶: 基于文本质量和原始概率动态限制最大提升幅度
    - 惰性加载: 大型模型资源按需加载，降低内存开销

    输出特性：
    - 保序性: 高质量文本获得更大加成
    - 边界安全: 输出概率严格在(0,1)区间
    - 平滑响应: 相似度变化引起连续的概率变化
    """

    def __init__(
        self,
        vectorizer_path: str,
        centroids_path: str,
        weights_path: str,
        max_total_boost: float = 0.05,
        sim_gate_sum_min: float | None = None,
        sim_gate_max_min: float | None = None,
        smoothing: float | None = None,
        cap_min_factor: float | None = None,
        cap_quality_gamma: float | None = None,
        high_signal: dict[str, Any] | None = None,
    ) -> None:
        """
        初始化 LogitUpliftProvider。

        Args:
            vectorizer_path: TF-IDF 向量器文件路径 (.joblib)。
            centroids_path: 各维度高质量聚类中心点路径 (.npz)。
            weights_path: 线性回归权重路径 (.json)。
            max_total_boost: 允许的最大概率提升上限（如 0.05 表示 5%）。
            sim_gate_sum_min: 总相似度门槛，低于此值不予加成。
            sim_gate_max_min: 单项最大相似度门槛，低于此值不予加成。
            smoothing: Logit 空间的平滑因子，用于控制加成的灵敏度。
            cap_min_factor: 最小封顶系数，即使文本质量极高，也会受此系数约束。
            cap_quality_gamma: 质量敏感度指数，用于调节质量得分对封顶上限的影响。
            high_signal: 信号增强配置（词库路径、权重等）。
        """
        self._max_total_boost = float(max_total_boost)
        self._sim_gate_sum_min = (
            LOGIT_UPLIFT_DEFAULT_SIM_GATE_SUM_MIN
            if sim_gate_sum_min is None
            else float(sim_gate_sum_min)
        )
        self._sim_gate_max_min = (
            LOGIT_UPLIFT_DEFAULT_SIM_GATE_MAX_MIN
            if sim_gate_max_min is None
            else float(sim_gate_max_min)
        )
        self._smoothing = LOGIT_UPLIFT_DEFAULT_SMOOTHING if smoothing is None else float(smoothing)
        self._cap_min_factor = (
            LOGIT_UPLIFT_DEFAULT_CAP_MIN_FACTOR if cap_min_factor is None else float(cap_min_factor)
        )
        self._cap_quality_gamma = (
            LOGIT_UPLIFT_DEFAULT_CAP_QUALITY_GAMMA
            if cap_quality_gamma is None
            else float(cap_quality_gamma)
        )

        text_keys = (
            "research_details",
            "award_details",
            "internship_details",
            "paper_details",
        )
        count_keys = ("research_count", "award_count", "internship_count", "paper_count")

        hs = high_signal or {}
        hs_enabled = bool(hs.get("enabled", False))
        signal_scorer: SignalScorer | None = None
        novelty_weight = 0.0
        novelty_min_chars = 12
        if hs_enabled:
            enabled_fields_raw = hs.get("enabled_fields")
            enabled_fields = (
                tuple(enabled_fields_raw)
                if isinstance(enabled_fields_raw, list)
                and all(isinstance(x, str) and x.strip() for x in enabled_fields_raw)
                else None
            )
            lexicon_path = hs.get("lexicon_path")
            signal_scorer = SignalScorer(
                lexicon_path=lexicon_path if isinstance(lexicon_path, str) else None,
                enabled_fields=enabled_fields,
                per_field_cap=float(hs.get("bonus_cap_per_field", 0.6)),
                lexicon_weight=float(hs.get("lexicon_weight", 1.0)),
            )
            novelty_weight = float(hs.get("novelty_weight", 0.12))
            novelty_min_chars = int(hs.get("novelty_min_chars", 12))

        self._text_processor = TextProcessor(text_keys=text_keys, count_keys=count_keys)
        self._model_loader = get_model_loader(
            vectorizer_path=vectorizer_path,
            centroids_path=centroids_path,
            weights_path=weights_path,
        )
        self._similarity_computer = SimilarityComputer(
            model_loader=self._model_loader,
            text_processor=self._text_processor,
            signal_scorer=signal_scorer,
            novelty_weight=novelty_weight,
            novelty_min_chars=novelty_min_chars,
        )
        self._delta_calculator = DeltaCalculator(
            model_loader=self._model_loader,
            similarity_computer=self._similarity_computer,
            text_processor=self._text_processor,
            sim_gate_sum_min=self._sim_gate_sum_min,
            sim_gate_max_min=self._sim_gate_max_min,
        )
        self._probability_applier = ProbabilityApplier(
            text_processor=self._text_processor,
            max_total_boost=self._max_total_boost,
            smoothing=self._smoothing,
            cap_min_factor=self._cap_min_factor,
            cap_quality_gamma=self._cap_quality_gamma,
        )

    def apply(self, probabilities: list[float], experience_details: dict[str, Any]) -> list[float]:
        if not probabilities:
            return probabilities
        if not has_any_experience(experience_details):
            return probabilities

        sig = self._text_processor.make_signature(experience_details)
        try:
            delta_logit, sims, remarks = self._delta_calculator.cached_delta_logit(sig)
        except (
            FileNotFoundError,
            OSError,
            EOFError,
            ValueError,
            TypeError,
            RuntimeError,
            ImportError,
            AttributeError,
        ) as e:
            logger.error(f"LogitUpliftProvider 计算 delta_logit 失败: {str(e)}", exc_info=True)
            return probabilities

        if delta_logit <= 0:
            return probabilities

        if remarks:
            flat_remarks = []
            for field, tags in remarks.items():
                field_cn = {
                    "research_details": "科研",
                    "award_details": "奖项",
                    "internship_details": "实习",
                    "paper_details": "论文",
                }.get(field, field)
                flat_remarks.append(f"{field_cn}: {', '.join(tags)}")

            if flat_remarks:
                logger.info(f"文本加成生效 [Logit+{delta_logit:.3f}]: {'; '.join(flat_remarks)}")

        updated = self._probability_applier.apply_probability_boost(
            probabilities=probabilities,
            delta_logit=delta_logit,
            sims=sims,
        )

        return updated
