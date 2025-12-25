# 结果修正模块 API

**路径**: `src/pages/prediction/result_modifier`

本模块用于在主模型输出基础概率后，做可控的后处理：GPA/语言惩罚、职业型专业降权、跨专业无录取惩罚、以及基于文本质量的 Logit 增量加成。

## 1. 结构概览

- **配置**：`config.py`
- **调整管线**：`adjustment_pipeline.py`
- **概率调整**：`probability_adjuster.py`（统计 + 分段惩罚）
- **文本加成入口**：`text_boost_provider.py`
- **文本加成实现**：详见 [文本加成 API (Text Uplift)](text_uplift_api.md)
- **推荐筛选**：`filters.py`
- **学院过滤器**：`faculty_filters.py` (处理跨学院惩罚)
- **语言惩罚辅助**：`language_penalty.py`
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

## 3. 调整管线 (`adjustment_pipeline.py`)

**类**: `ProbabilityAdjustmentPipeline`

`adjust_batch(results, ctx)` 会按顺序做：
1.  **GPA/语言惩罚**：调用 `ProbabilityAdjuster.adjust_probability`。
2.  **跨专业无录取惩罚**：基于 `MIN_SIMILARITY_THRESHOLD` 和历史录取数据（`admitted_combinations`）。
3.  **跨学院惩罚**：由 `faculty_filters.apply_out_of_scope_faculty_penalty` 执行。
4.  **职业型专业降权**：若 `internship_count <= 0`，对职业型专业（如 MBA/BA）进行降权。
5.  **新增专业标记**：根据 `is_new_major_cache` 注入 `is_new_major` 字段。
6.  **文本加成**：调用 `TextBoostProvider.apply`。

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

该功能已解耦至独立文档，请参阅：[**文本加成 API 文档**](text_uplift_api.md)。

### 快速要点：
- **核心逻辑**：基于增量建模（Uplift Modeling）计算 Logit 偏移。
- **防作弊**：内置香农熵（Shannon Entropy）检测，自动压制重复、注水文本的加成。
- **可解释性**：日志会自动输出加成原因（如命中的关键词标签）。
- **性能**：极致优化的字节级计算，处理速度比调用 LLM 快数万倍。

## 6. 推荐筛选 (`filters.py`)

- `get_similar_major_recommendations(...)`
  - 相似度阈值：院校数少时用 `0.92`，否则用 `0.89`。
  - 取 `TOP_N_RECOMMENDATIONS=30`，按概率降序。

- `get_cross_major_recommendations(...)`
  - 仅在“历史存在录取组合”的跨专业范围内选：`0.8 <= similarity < 0.89`。
