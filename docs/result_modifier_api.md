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

### 第一阶段：单项处理与筛选 (`flow/processor.py` & `filters.py`)
在生成推荐列表时，系统会执行以下逻辑：
1.  **单项处理 (SingleResultProcessor)**：
    - **目标特定语言要求惩罚**：从专业详情中提取 IELTS/TOEFL 录取门槛，若用户分数低于门槛，应用惩罚。
    - **元数据注入**：附加学部 (`faculty`)、专业中文名等。
2.  **相似度偏置修正**：调用 `SimilarityAdjuster` 根据背景专业微调原始相似度。
    - **模糊偏移 (Fuzzy Bias)**：基于字符重合度施加乘法增益。
    - **规则修正**：基于 `similarity_adjustment_rules.json` 进行微调。
3.  **槽位保障 (Identity Slot)**：在推荐相似专业时，保留至少 40% (`IDENTITY_MIN_SLOT_RATIO`) 的名额给强匹配专业。
4.  **推荐分值计算 (Selection Score)**：推荐列表的排序并非纯相似度，而是综合了 `similarity * (1 + boost)`。
    - **Boost 条件**：当用户的综合背景分 (GPA+语言+背景院校) > 0.6 且目标院校难度已知时触发。
5.  **Agent 平衡**：调用 `BoundaryCaseAgent` 进行边界探索，动态平衡推荐项数量。

### 第二阶段：批量修正流水线 (`adjustment_pipeline.py`)
**类**: `src/pages/prediction/result_modifier/adjustment_pipeline.py::ProbabilityAdjustmentPipeline`

`adjust_batch(results, ctx)` 会对每个结果按顺序执行：
1.  **通用 GPA/语言惩罚**：根据用户 GPA 和语言分与历史录取的偏差施加惩罚（含二次项惩罚）。
2.  **动态惩罚项**：
    - **跨专业惩罚 (`CrossMajorPenalty`)**: 若相似度低且无历史录取，施加惩罚。
    - **跨学部惩罚**：若目标专业不在背景学部的兼容矩阵范围内，施加惩罚。
    - **职业型专业降权**：若缺乏实习背景且目标为职业型专业，乘以降权因子。
3.  **文本加成 (`TextBoostProvider`)**：基于 Logit Uplift 模型计算 NLP 经验带来的概率增量。
4.  **仲裁与归一化**：
    - 所有因子通过 `AdjustmentArbitrator` 融合，并执行**因子衰减**和**总偏移约束**。
    - 最终通过 `NormalizationLayer` 保证概率在 `[0.005, 1.0]` 区间。

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
