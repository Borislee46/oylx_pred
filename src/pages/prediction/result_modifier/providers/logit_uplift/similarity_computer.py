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
    相似度计算核心类。
    
    该类负责量化用户输入的背景描述（非结构化文本）与高质量录取案例（特征中心）之间的"语义距离"。
    
    算法演进与融合逻辑：
    1. **基础得分 (s0)**: 使用预训练的 TF-IDF 向量器将文本转为高维向量，并计算其与对应维度（如科研、实习）高质量案例质心 (Centroids) 的余弦相似度。
    2. **信号加成 (lexicon_bonus)**: 针对 TF-IDF 无法捕捉的精确关键词（如"一等奖"、"Nature"、"核心期刊"）进行硬匹配加成。
    3. **新颖性探测 (novelty_bonus)**: 检查 TF-IDF 向量中的最大权重。如果某个词的权重极高，说明该词在语料库中极为罕见但被用户提及，通常暗示了极其细分的专业背景或独特的成就。
    
    数学融合公式（Bounded Fuse）：
    $S_{final} = 1 - (1 - S_{base}) \times (1 - S_{bonus})$
    这种非线性融合方式确保了：
    - 结果始终在 [0, 1] 闭区间内。
    - 具有"边际效用递减"特性：如果基础得分已经很高，额外的加成带来的提升会逐渐减小，避免过度膨胀。
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

    def compute_similarities(self, details: dict[str, Any]) -> tuple[dict[str, float], dict[str, list[str]]]:
        """
        计算所有维度的综合相似度。

        Returns:
            tuple: (相似度字典, 详细原因/标签记录)
        """
        vectorizer = self._model_loader.vectorizer
        centroids = self._model_loader.centroids
        text_keys = self._text_processor.text_keys

        # 1. 文本预处理与批量向量化
        texts = [self._text_processor.prep_text(details.get(k, "")) for k in text_keys]
        if all(not t for t in texts):
            return dict.fromkeys(text_keys, 0.0), {}

        # 核心：TF-IDF 转化 (CSR Sparse Matrix)
        X = vectorizer.transform(texts)

        # 2. 关键词词库评分
        lex_bonuses: dict[str, float] = {}
        lex_tags: dict[str, list[str]] = {}
        if self._signal_scorer is not None:
            lex_bonuses, lex_tags = self._signal_scorer.score(
                {k: texts[idx] for idx, k in enumerate(text_keys)}
            )

        sims: dict[str, float] = {}
        remarks: dict[str, list[str]] = {}
        
        # 3. 遍历各维度计算余弦相似度并融合
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
                current_remarks.append("content_novelty") # 内容独特性标签
            
            # 融合基础分与加成分
            bonus = float(lex_bonuses.get(k, 0.0)) + novelty_bonus
            sims[k] = self._bounded_fuse(s0, bonus)
            
            if current_remarks:
                remarks[k] = current_remarks
                
        return sims, remarks
