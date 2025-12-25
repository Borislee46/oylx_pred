from __future__ import annotations

import json
import math
from functools import lru_cache

import numpy as np

from src.pages.prediction.result_modifier.providers.logit_uplift.model_loader import (
    ModelLoader,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.similarity_computer import (
    SimilarityComputer,
)
from src.pages.prediction.result_modifier.providers.logit_uplift.text_processor import (
    TextProcessor,
)


class DeltaCalculator:
    """
    Logit 增量计算器。

    该类实现了加成模型的核心数学逻辑：将多维度的文本质量指标转换为一个统一的 Logit 空间偏移量。

    数学模型：
    $\Delta Logit = \beta_0 + \sum_{i} w_i \cdot S'_{i} + \sum_{i} u_i \cdot S'_{i} \cdot \ln(1 + C_i \cdot R_i)$

    其中：
    - $\beta_0$: 偏置项 (Bias)，模型的基础调节量。
    - $w_i$: 第 $i$ 个维度的相似度权重。
    - $S'_i$: 经过内容丰富度修正后的相似度得分 ($S_i \times R_i$)。
    - $u_i$: 交互项权重，用于衡量"质量"与"数量"共同作用带来的额外加成。
    - $C_i$: 经历的数量（如发表了几篇论文）。
    - $R_i$: 文本的内容丰富度 (Entropy-based Richness)，用于抑制"充数"的经历。

    门槛控制：
    只有当总相似度 $\sum S_i$ 或最大单项相似度 $\max S_i$ 达到预设阈值时，才会计算加成。这保证了加成只赋予有实质性背景的用户。
    """

    def __init__(
        self,
        model_loader: ModelLoader,
        similarity_computer: SimilarityComputer,
        text_processor: TextProcessor,
        sim_gate_sum_min: float,
        sim_gate_max_min: float,
    ) -> None:
        self._model_loader = model_loader
        self._similarity_computer = similarity_computer
        self._text_processor = text_processor
        self._sim_gate_sum_min = sim_gate_sum_min
        self._sim_gate_max_min = sim_gate_max_min
        # 使用 LRU 缓存避免在用户多次点击或刷新时重复进行昂贵的文本计算
        self._get_delta_logit_cached = lru_cache(maxsize=512)(self._compute_delta_logit_raw)

    def _compute_delta_logit_raw(
        self, sig: str
    ) -> tuple[float, tuple[tuple[str, float], ...], tuple[tuple[str, tuple[str, ...]], ...]]:
        """
        内部逻辑执行：从签名反序列化 -> 计算相似度 -> 应用线性模型。
        """
        weights = self._model_loader.weights_array
        text_keys = self._text_processor.text_keys
        count_keys = self._text_processor.count_keys
        compute_sims = self._similarity_computer.compute_similarities
        sum_min = self._sim_gate_sum_min
        max_min = self._sim_gate_max_min
        _log1p = math.log1p

        details = {}
        if sig and sig.startswith("{"):
            try:
                details = json.loads(sig)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # 1. 计算各维度的相似度融合分
        sims, remarks = compute_sims(details)
        if not sims:
            return 0.0, (), ()

        n_text = len(text_keys)
        s_values = [0.0] * n_text
        ssum = 0.0
        smax = 0.0
        sims_get = sims.get

        # 2. 统计相似度指标
        for i in range(n_text):
            val = sims_get(text_keys[i], 0.0)
            if val:
                s_values[i] = val
                ssum += val
                if val > smax:
                    smax = val

        # 3. 门槛检查：避免对弱相关文本进行加成
        if ssum < sum_min or smax < max_min:
            return 0.0, tuple(sims.items()), tuple((k, tuple(v)) for k, v in remarks.items())

        # 4. 执行线性回归预测
        delta = float(weights[0])  # weights[0] 为偏置项 b
        tw_start = 1

        text_w = weights[tw_start : tw_start + n_text]
        n_counts = len(count_keys)
        # 检查是否存在交互项权重
        has_inter = n_counts == n_text and len(weights) >= tw_start + 2 * n_text
        inter_w = weights[tw_start + n_text : tw_start + 2 * n_text] if has_inter else None
        details_get = details.get

        sims_adj = {}
        for i in range(n_text):
            s = s_values[i]
            if s <= 0:
                continue

            txt = details_get(text_keys[i], "")
            # 计算丰富度修正因子 (0~1)
            richness = _fast_entropy(txt)

            # 修正相似度：如果内容空洞（低熵），则认为其质量分无效
            s_adj = float(s * richness)
            sims_adj[text_keys[i]] = s_adj
            delta += text_w[i] * s_adj

            # 5. 计算交互项：$\text{weight} \cdot \text{Quality} \cdot \ln(1 + \text{Count} \cdot \text{Richness})$
            if has_inter:
                v = details_get(count_keys[i])
                if v:
                    try:
                        fv = float(v)
                        if fv > 0:
                            # 数量也要受 richness 抑制，防止虚报 count
                            delta += inter_w[i] * s_adj * _log1p(fv * richness)
                    except (TypeError, ValueError):
                        pass

        # 返回 Logit 增量（非负）、修正后的相似度和记录
        return (
            (delta if delta > 0.0 else 0.0),
            tuple(sims_adj.items()),
            tuple((k, tuple(v)) for k, v in remarks.items()),
        )

    def cached_delta_logit(self, sig: str) -> tuple[float, dict[str, float], dict[str, list[str]]]:
        """
        对外接口，提供自动反序列化的缓存支持。
        """
        delta, sims_tuple, remarks_tuple = self._get_delta_logit_cached(sig)
        return delta, dict(sims_tuple), {k: list(v) for k, v in remarks_tuple}


def _fast_entropy(text: str) -> float:
    """
    基于信息熵的内容丰富度检测。

    原理：
    通过计算文本字节流的香农熵 (Shannon Entropy) 来判断文本的信息量。
    - 随机或重复的文本（如 "aaaaa..."）熵值极低。
    - 内容充实的描述（中英文混排、专业术语）熵值较高。

    实现：
    1. 在字节层级进行频率统计。
    2. 计算 $H(x) = -\sum p_i \log_2(p_i)$。
    3. 归一化：通常自然语言熵在 3~5 左右，我们以 5.0 为基准映射到 [0, 1]。
    """
    if not text:
        return 0.0
    try:
        b = text.encode("utf-8")
    except UnicodeEncodeError:
        return 0.0
    if len(b) < 10:
        return 0.0
    # 使用 np.bincount 高效统计字节分布 (0-255)
    counts = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    probs = counts[counts > 0] / len(b)
    entropy = -np.sum(probs * np.log2(probs))
    # 映射到 [0, 1] 空间，5.0 是信息丰富度的饱和阈值
    return float(np.clip(entropy / 5.0, 0.0, 1.0))
