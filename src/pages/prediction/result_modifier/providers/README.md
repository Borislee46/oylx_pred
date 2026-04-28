# Result Modifier Providers 技术文档

## 1. 模块概述

`providers` 模块为预测结果修饰器（Result Modifier）提供文本加成（Text Boost）的实现。当前主要实现为 **LogitUpliftProvider**，基于 TF-IDF 向量化、质心相似度与 logit 空间线性模型，对用户背景提升（背提）文本进行量化评估，并据此提升录取概率预测值。

## 2. 目录结构

```
providers/
├── logit_uplift_provider.py    # 主入口：LogitUpliftProvider
└── logit_uplift/               # 核心算法子模块
    ├── __init__.py
    ├── model_loader.py         # 模型加载
    ├── text_processor.py       # 文本预处理与签名
    ├── similarity_computer.py # 相似度计算
    ├── signal_scorer.py       # 高信号词表评分
    ├── delta_calculator.py    # delta_logit 计算
    ├── probability_applier.py # 概率加成应用
    └── utils.py               # 工具函数
```

## 3. 核心流程

```
experience_details (用户背提数据)
        │
        ▼
   TextProcessor.make_signature()  →  JSON 签名
        │
        ▼
   DeltaCalculator.cached_delta_logit()
        │
        ├── SimilarityComputer.compute_similarities()
        │       ├── TF-IDF 向量化
        │       ├── 与质心点积 → 基础相似度
        │       ├── SignalScorer 词表匹配 → 词表加成
        │       └── 内容新颖度加成
        │
        ├── 相似度门控 (sim_gate_sum_min, sim_gate_max_min)
        ├── 线性模型: delta = b + Σ(w_r·s_r + ...) + Σ(u_r·s_r·log1p(count_r)·richness + ...)
        │
        └── 输出: (delta_logit, sims, remarks)
        │
        ▼
   ProbabilityApplier.apply_probability_boost()
        │
        ├── 质量分 → cap_factor
        ├── effective_delta = delta_logit * smoothing
        ├── logit 空间: new_p = sigmoid(logit(p) + effective_delta)
        └── 上限约束: min(new_p, cap)
        │
        ▼
   updated probabilities
```

## 4. 组件说明

### 4.1 LogitUpliftProvider

继承 `TextBoostProvider`，实现 `apply(probabilities, experience_details) -> list[float]`。

**初始化参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| vectorizer_path | str | TF-IDF 向量器路径 (.joblib) |
| centroids_path | str | 质心矩阵路径 (.npz) |
| weights_path | str | 线性模型权重路径 (.json) |
| max_total_boost | float | 最大总加成比例，默认 0.05 |
| sim_gate_sum_min | float | 相似度之和门控下限 |
| sim_gate_max_min | float | 最大相似度门控下限 |
| smoothing | float | delta 平滑系数 |
| cap_min_factor | float | 加成上限最小因子 |
| cap_quality_gamma | float | 质量分幂次 |
| high_signal | dict | 高信号词表配置 |

**背提字段：**

- 文本：`research_details`, `award_details`, `internship_details`, `paper_details`
- 计数：`research_count`, `award_count`, `internship_count`, `paper_count`

### 4.2 ModelLoader

延迟加载 TF-IDF 向量器、质心、权重，线程安全。

- `vectorizer`: joblib 加载的 TF-IDF 模型
- `centroids`: 各字段质心向量（已 L2 归一化）
- `weights_array`: `(b, w_r, w_a, w_i, w_p, u_r, u_a, u_i, u_p)` 元组

`get_model_loader()` 通过 `cache_resource` 缓存，避免重复加载。

### 4.3 TextProcessor

- `prep_text(s)`: 文本清洗（strip）
- `make_signature(details)`: 将 experience_details 转为 JSON 字符串，用于缓存 key

### 4.4 SimilarityComputer

计算各背提字段与质心的相似度，并融合词表加成与新颖度加成。

- **基础相似度**：TF-IDF 向量与质心点积，clip 到 [0,1]
- **词表加成**：`SignalScorer` 按规则匹配，`_bounded_fuse(base, bonus)` 融合
- **新颖度加成**：基于 TF-IDF 行最大值的简单变换，需 `novelty_min_chars` 以上文本

### 4.5 SignalScorer

从 JSON 词表加载规则，格式：

```json
{
  "rules": [
    {"pattern": "关键词", "score": 0.5, "tag": "标签", "fields": ["research_details"]}
  ]
}
```

- `fields: null` 表示全局规则
- 子串匹配（`pattern in text`），取最高 score，按 `lexicon_weight` 和 `per_field_cap` 缩放

### 4.6 DeltaCalculator

计算 logit 空间增量 `delta_logit`。

- **门控**：`sum(sims) < sim_gate_sum_min` 或 `max(sims) < sim_gate_max_min` 时返回 0
- **线性模型**：
  - 截距 `b`
  - 文本项：`w_r * s_r * richness` 等
  - 交互项（可选）：`u_r * s_r * log1p(count_r) * richness` 等
- **richness**：`_fast_entropy(text)`，基于字节熵的文本丰富度，clip 到 [0,1]
- 使用 `lru_cache(maxsize=512)` 按签名缓存

### 4.7 ProbabilityApplier

将 `delta_logit` 应用到概率列表。

- **质量分**：`q = 0.7 * max(sims) + 0.3 * mean(sims)`，`cap_factor = min(1, max(cap_min_factor, q^gamma))`
- **有效 delta**：`effective_delta = delta_logit * smoothing`
- **logit 变换**：`new_p = sigmoid(logit(p) + effective_delta)`
- **适用范围**：仅对 `p ∈ [PROBABILITY_BOOST_MIN, PROBABILITY_BOOST_MAX]` 的项加成
- **上限**：`cap = p * (1 + max_total_boost * cap_factor * scale)`，`scale` 与 `|p - 0.5|` 相关

### 4.8 utils

- `safe_float(x, default)`: 安全转 float
- `logit(p)`: numba JIT，`log(p/(1-p))`
- `sigmoid(z)`: numba JIT，`1/(1+exp(-z))`

## 5. 配置与依赖

- 模型文件：`tfidf_vectorizer.joblib`, `tfidf_centroids.npz`, `text_uplift_weights.json`
- 高信号词表：`config/text_high_signal_terms.json`
- 依赖：`joblib`, `numpy`, `numba`

## 6. 扩展

新增 Provider 需继承 `TextBoostProvider` 并实现 `apply()`。`get_text_boost_provider(config)` 当前固定返回 `GatedTextBoostProvider(LogitUpliftProvider(...))`，扩展时可基于 `config` 选择不同实现。
