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
- **最小概率截断**：`PROBABILITY_MIN_VALUE=0.001` (仲裁后强制最低 `0.005`)
- **衰减系数**：惩罚项 `0.85`，加成项 `0.8`
- **安全封顶**：总惩罚率上限 `0.7`，总加成率上限 `0.3`
- **跨专业惩罚**：`CROSS_MAJOR_PENALTY_FACTOR=0.5`
- **跨学院惩罚**：`FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR=0.3`
- **职业型专业**：`["Business Administration","MBA"]`
- **相似度阈值**：`MIN_SIMILARITY_THRESHOLD=0.89`, `HIGHER_SIMILARITY_THRESHOLD=0.92`

## 3. 调整流程

整个调整分为两个阶段：

### 第一阶段：平准与平衡 (`flow/processor.py`)
在生成推荐列表时，系统会执行以下逻辑：
1.  **相似度偏置**：调用 `SimilarityAdjuster` 根据背景专业微调原始相似度。
    - **模糊偏移 (Fuzzy Bias)**：基于 `fuzz.token_sort_ratio` 计算背景与目标专业的字符重合度，若得分超过 92/82/72，分别施加 1.5/1.2/1.1 倍的乘法增益。
    - **人工规则**：从 `similarity_adjustment_rules.json` 加载关键词匹配规则，对满足条件的组合进行加法/减法修正。
2.  **槽位保障 (Identity Slot)**：在 `get_similar_major_recommendations` 中，保留至少 40% (`IDENTITY_MIN_SLOT_RATIO`) 的名额给强匹配专业（Token Sort Ratio > 92）。
3.  **推荐分值计算 (Selection Score)**：推荐列表的排序并非纯相似度，而是综合了 `similarity * (1 + boost)`。
    - **Boost 条件**：当用户的综合背景分（Comprehensive Score） > 0.6 且目标院校难度已知时触发。
    - **综合背景分**：由 40% GPA 得分 + 30% 语言得分 + 30% 背景院校得分构成。
4.  **Agent 平衡**：若相似推荐与跨专业推荐数量差异过大（超过 `AGENT_MIN_BALANCE_DIFF_RATIO`），调用 `BoundaryCaseAgent` 进行边界探索，动态增加或减少推荐项。

### 第二阶段：修正管线 (`adjustment_pipeline.py`)
**类**: `src/pages/prediction/result_modifier/adjustment_pipeline.py::ProbabilityAdjustmentPipeline`

`adjust_batch(results, ctx)` 会对每个结果按顺序执行：
1.  **目标特定语言要求惩罚**：
    - 从专业详情中提取 IELTS/TOEFL 录取门槛。
    - 若用户分数低于门槛，应用基于 Sigmoid 函数的惩罚。
2.  **通用 GPA/语言惩罚**：
    - **GPA 惩罚**：若 GPA < `GPA_MINIMUM` (2.0)，截断概率；否则根据其与历史录取均值的偏差施加**二次项惩罚**（Quadratic Penalty）。
    - **语言惩罚**：若语言分低于门槛或均值偏差过大，按分段步长施加惩罚。
3.  **仲裁器处理 (Arbitrator)**：所有修正因子通过 `AdjustmentArbitrator` 进行融合。
    - **衰减机制**：修正因子按强度排序，随后的因子会乘以衰减系数（惩罚项 0.85，加成项 0.8）。
    - **安全约束**：系统强制执行 `MAX_TOTAL_PENALTY_RATIO` (0.7) 和 `MAX_TOTAL_BOOST_RATIO` (0.3) 的总偏移约束。
4.  **跨专业惩罚 (`CrossMajorPenalty`)**: 若相似度 < `MIN_SIMILARITY_THRESHOLD` (0.89) 且无历史录取记录，施加惩罚。
5.  **跨学院惩罚**：
    - 基于内置的 `CROSS_FACULTY_RULES` 学部兼容矩阵判定。
    - 若目标专业不在背景专业允许的学部范围内，乘以 `FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR` (0.3)。
6.  **职业型专业降权**：若 `internship_count <= 0` 且目标为职业型专业（如 MBA/BA），乘以降权因子。
7.  **文本加成 (`TextBoostProvider`)**：调用 Logit Uplift 模型计算概率增量。
8.  **归一化层**：最终概率保证 $\ge$ `ARBITRATION_MIN_PROBABILITY` (0.005) 且 $\le 1.0$。

## 4. 核心配置项 (`config.py`)

- **相似度阈值**：院校数 <= 2 时用 `0.92`，否则用 `0.89`。
- **跨专业范围**：`0.8 <= similarity < 0.89`。
- **职业型专业**：`["Business Administration", "MBA"]`。
- **惩罚因子**：
    - 跨专业：`CROSS_MAJOR_PENALTY_FACTOR = 0.5`
    - 跨学院：`FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR = 0.3`
    - 职业型（普通）：`PROFESSIONAL_REDUCTION_FACTOR = 0.3`
    - 职业型（用户指定）：`PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR = 0.5`

## 5. 文本内容预核验 (`experience_text_validator.py`)

在触发文本加成模型前，系统会进行**有效性检查**：
1.  **LLM 校验**：若配置了 OpenAI 接口，调用 `TextPreprocessingAgent` 识别输入文本是否为真实的科研、实习、获奖描述（防止乱填或凑字数）。
2.  **本地兜底**：若 LLM 未启用或调用失败，通过正则表达式清洗非中英文字符，并校验清洗后的有效字符长度是否 $\ge 3$。
3.  **门控效应**：只有核验通过的字段才会参与后续的 Logit Uplift 计算。

## 6. 文本加成 (`text_boost_provider.py`)

该功能已解耦至独立文档，请参阅：[**文本加成 API 文档**](text_uplift_api.md)。

### 快速要点：
- **核心逻辑**：基于增量建模（Uplift Modeling）计算 Logit 偏移。
- **防作弊**：内置香农熵（Shannon Entropy）检测，自动压制重复、注水文本的加成。
- **可解释性**：日志会自动输出加成原因（如命中的关键词标签）。
- **性能**：极致优化的字节级计算，处理速度比调用 LLM 快数万倍。

## 6. 学院兼容矩阵 (`faculty_filters.py`)

系统内置了 `CROSS_FACULTY_RULES` 矩阵，定义了各学部之间的申请兼容性。若目标专业所属学部不在背景学部的兼容列表中，将触发惩罚。部分典型规则：
- **商学院**：兼容【社会科学院、文学院】。
- **理学院/工程学院**：兼容【商学院、计算机学院、建筑/设计学院】。
- **法学院/医学院**：通常仅兼容自身。

## 7. 推荐筛选与排序 (`filters.py`)

- `get_similar_major_recommendations(...)`
  - 相似度阈值：院校数少时用 `0.92`，否则用 `0.89`。
  - 取 `TOP_N_RECOMMENDATIONS=30`，按概率降序。

- `get_cross_major_recommendations(...)`
  - 仅在“历史存在录取组合”的跨专业范围内选：`0.8 <= similarity < 0.89`。
