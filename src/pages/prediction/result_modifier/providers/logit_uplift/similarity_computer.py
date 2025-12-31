from __future__ import annotations

from typing import Any

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.signal_scorer import (
    SignalScorer,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)
from src.pages.prediction.result_modifier.utils import clip_probability


class SimilarityComputer:
    """
    语义相似度计算器v2.6。

    将用户背景描述与高质量案例特征中心进行多维度的相似度评估，
    综合使用统计、词法和新颖性特征。

    计算流程：
    1. 基础相似度 (s0)
       - 使用 TF-IDF 将文本向量化
       - 计算与高质量案例质心（Centroids）的余弦相似度

    2. 关键词加成 (lexicon_bonus)
       - 硬匹配特定成就词汇（如"一等奖"、"Nature"）
       - 弥补 TF-IDF 对精确术语的敏感性不足，同时提高泛化性（当前欧亚的案例并没有所谓的大众认知上的关键词signal）

    3. 新颖性探测 (novelty_bonus)
       - 检测 TF-IDF 向量中的极端权重词
       - 高权重罕见词通常表示独特专业背景或特殊成就

    融合策略（Bounded Fuse）：
    S_final = 1 - (1 - S_base) × (1 - S_bonus)

    特性：
    - 输出范围固定：[0, 1]
    - 边际效用递减：高基础分时，加成效果减弱
    - 防止分数膨胀：无论多少加成，总分不超过1

    设计理念：
    1. 统计为主：TF-IDF 捕捉常规语义
    2. 规则补充：关键词匹配确保重要信号不丢失
    3. 异常奖励：对独特背景给予额外认可
    4. 稳健融合：非线性组合防止分数失真
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        text_processor: TextProcessor,
        signal_scorer: SignalScorer | None = None,
        novelty_weight: float = 0.0,
        novelty_min_chars: int = 12,
    ) -> None:
        self._model_loader = model_loader
        self._text_processor = text_processor
        self._signal_scorer = signal_scorer
        self._novelty_weight = float(novelty_weight)
        self._novelty_min_chars = int(novelty_min_chars)

    @staticmethod
    def _bounded_fuse(base: float, bonus: float) -> float:
        """
        执行非线性分值融合。确保结果在 [0, 1] 范围内。
        """
        base = clip_probability(base)
        bonus = clip_probability(bonus)
        return 1.0 - (1.0 - base) * (1.0 - bonus)

    def _compute_novelty_bonus(self, text: str, row: Any) -> float:
        """
        计算文本新颖性加成。

        逻辑：
        利用 TF-IDF 的特性，如果一个稀有词出现在文本中，它在向量中的值会很大。
        我们以此作为"高含金量/独特性"的代理指标。
        """
        if self._novelty_weight <= 0:
            return 0.0
        if not isinstance(text, str) or len(text.strip()) < self._novelty_min_chars:
            return 0.0
        if row is None or getattr(row, "data", None) is None or row.data.size == 0:
            return 0.0

        # 提取向量中的最大分量作为新颖性原始值
        max_val = float(np.max(row.data))
        # 映射公式：(max_val - 阈值) / 缩放系数。0.18 和 0.35 为经验参数，还在灰测中。
        raw = clip_probability((max_val - 0.18) / 0.35)
        return clip_probability(raw * self._novelty_weight)

    def compute_similarities(
        self, details: dict[str, Any]
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """
        计算所有维度的综合相似度。

        Returns:
            tuple: (相似度字典, 详细原因/标签记录)
        """
        vectorizer = self._model_loader.vectorizer
        centroids = self._model_loader.centroids
        text_keys = self._text_processor.text_keys

        # 文本预处理与批量向量化
        texts = [self._text_processor.prep_text(details.get(k, "")) for k in text_keys]
        if all(not t for t in texts):
            return dict.fromkeys(text_keys, 0.0), {}

        # TF-IDF 转化 (CSR Sparse Matrix)
        X = vectorizer.transform(texts)

        # 关键词词库评分
        lex_bonuses: dict[str, float] = {}
        lex_tags: dict[str, list[str]] = {}
        if self._signal_scorer is not None:
            lex_bonuses, lex_tags = self._signal_scorer.score(
                {k: texts[idx] for idx, k in enumerate(text_keys)}
            )

        sims: dict[str, float] = {}
        remarks: dict[str, list[str]] = {}

        # 遍历各维度计算余弦相似度并融合
        for idx, k in enumerate(text_keys):
            row = X.getrow(idx)
            current_remarks = []

            if row.nnz == 0:
                sims[k] = 0.0
                continue

            centroid = centroids.get(k)
            if centroid is None or centroid.size == 0:
                sims[k] = 0.0
                continue

            # 余弦相似度计算：dot(v1, v2) / (||v1|| * ||v2||)
            # 由于质心和 TF-IDF 向量均已 L2 归一化，直接点积即为余弦值
            dot_val = row.dot(centroid)
            dot_scalar = float(np.asarray(dot_val).flat[0])
            s0 = clip_probability(dot_scalar)

            # 记录加成理由
            if k in lex_tags:
                current_remarks.extend(lex_tags[k])

            novelty_bonus = self._compute_novelty_bonus(texts[idx], row)
            if novelty_bonus > 0.001:
                current_remarks.append("content_novelty")  # 内容独特性标签

            # 融合基础分与加成分
            bonus = float(lex_bonuses.get(k, 0.0)) + novelty_bonus
            sims[k] = self._bounded_fuse(s0, bonus)

            if current_remarks:
                remarks[k] = current_remarks

        return sims, remarks
