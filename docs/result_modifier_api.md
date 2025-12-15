# 结果修正模块 API

**路径**: `src/pages/prediction/result_modifier`

本模块用于在主模型输出基础概率后，做可控的后处理：GPA/语言惩罚、职业型专业降权、跨专业无录取惩罚、以及基于 TF‑IDF 的 Logit 文本加成。

## 1. 结构概览

- **配置**：`config.py`
- **调整管线（新）**：`adjustment_pipeline.py`
- **概率调整**：`probability_adjuster.py`（统计 + 分段惩罚）
- **文本加成入口**：`text_boost_provider.py`
- **文本加成实现**：`providers/logit_uplift_provider.py`
- **推荐筛选**：`filters.py`
- **相似度规则微调**：`similarity_adjuster.py`
- **Agent 排序调整**：`ranker.py` / `engine.py` / `strategies.py`
- **录取组合缓存**：`admission_cache.py`
- **工具函数**：`utils.py`

## 2. 配置 (`config.py`)

### 文本加成配置 (`DEFAULT_TEXT_BOOST_CONFIG`)

- `enabled: bool`
- `max_total_boost`: 概率上限的最大相对提升幅度
- `sim_gate_*`: 相似度门控阈值
- `cap_min_factor / cap_quality_gamma`: 封顶因子
- `high_signal`: 可选的高信号词典/新颖度加分配置
- `model_paths`: 指向 TF-IDF 相关模型文件

### 常用阈值/系数

- **GPA/语言极低阈值**：`GPA_MINIMUM=2.0`, `LANGUAGE_MINIMUM=0.6`
- **最小概率截断**：`PROBABILITY_MIN_VALUE=0.001`
- **跨专业惩罚**：`CROSS_MAJOR_PENALTY_FACTOR=0.5`
- **职业型专业**：`PROFESSIONAL_MAJORS=["Business Administration","MBA"]`
- **相似度阈值**：`MIN_SIMILARITY_THRESHOLD=0.89`

## 3. 新版调整管线 (`adjustment_pipeline.py`)

**类**: `ProbabilityAdjustmentPipeline`

`adjust_batch(results, ctx)` 会按顺序做：
1.  （可选）职业型专业降权
2.  （可选）GPA/语言惩罚
3.  （可选）跨专业无录取惩罚
4.  （可选）文本加成

输出会把 `probability` 夹到 [0,1]。

## 4. 概率调整 (`probability_adjuster.py`)

**类**: `ProbabilityAdjuster`

- **adjust_probability(p, gpa, language_score, ...)**
  - 若 `gpa` 或 `language_score` 低于极低阈值，直接返回 `PROBABILITY_MIN_VALUE`。
  - 否则按 GPA/语言分段惩罚乘到概率上，并对极端低分做截断。

- **penalize_cross_major_without_cases(...)**
  - 仅对“用户指定结果列表”中的跨专业项生效（相似度低且无历史录取）。
  - 对概率乘 `CROSS_MAJOR_PENALTY_FACTOR`。

## 5. 文本加成 (`text_boost_provider.py`)

### 触发条件
- `has_valid_experience_details(experience_details)` 为 True：至少一项经历详情不是空/无意义文本。

### 计算逻辑
1.  **相似度计算**：TF-IDF 向量与质心点积，可叠加 High Signal 加分。
2.  **门控**：总相似度或最大相似度需超过阈值。
3.  **Delta Logit**：`delta = max(0, b + Σ w_k*s_k + Σ u_k*(s_k*log1p(count_k)))`。
4.  **概率应用**：
    - 仅对概率在 [0.1, 0.9] 范围内的结果生效。
    - 使用 Sigmoid 函数应用 Logit 增量。
    - 应用动态封顶逻辑，防止加成过度。

## 6. 推荐筛选 (`filters.py`)

- `get_similar_major_recommendations(...)`
  - 相似度阈值：院校数少时用 `0.92`，否则用 `0.89`。
  - 取 `TOP_N_RECOMMENDATIONS=30`，按概率降序。

- `get_cross_major_recommendations(...)`
  - 仅在“历史存在录取组合”的跨专业范围内选：`0.8 <= similarity < 0.89`。

## 7. Agent 排序调整 (`ranker.py`)

- `prediction_processor._apply_agent_balance_adjustment`
- 计算跨专业与相似专业的数量差 `balance_diff`。
- 若差异过大，触发 `BoundaryCaseAgent` 进行探索补充。
