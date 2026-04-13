# 结果修正模块（Result Modifier）

| 项 | 说明 |
|----|------|
| 源码路径 | `src/pages/prediction/result_modifier/` |
| 作用域 | 主模型输出的概率在交付前的可控后处理（统计惩罚、业务规则、文本 Logit uplift） |
| 文本 uplift 细节 | [text_uplift_api.md](text_uplift_api.md) |

`AdjustmentContext` 字段表、仲裁公式、`RelaxStrategy`/`TightenStrategy` 行为及测试路径等更细的实现说明见 [src/pages/prediction/result_modifier/README.md](../src/pages/prediction/result_modifier/README.md)。

## 1. 组件索引

| 模块 | 职责 |
|------|------|
| `config.py` | 阈值、惩罚系数、文本加成默认配置与路径 |
| `adjustment_pipeline.py` | `ProbabilityAdjustmentPipeline` 批量修正入口 |
| `probability_adjuster.py` | GPA/语言等分段惩罚 |
| `text_boost_provider.py` | 文本加成提供者装配 |
| `filters.py` | 相似/跨专业推荐筛选与排序辅助 |
| `faculty_filters.py` | 跨学部兼容与惩罚 |
| `language_penalty.py` | 目标专业语言要求相关惩罚 |
| `similarity_adjuster.py` | 相似度规则与模糊偏置 |
| `ranker.py` / `engine.py` / `strategies.py` | Agent 侧排序策略 |
| `admission_cache.py` | 历史录取组合缓存 |
| `experience_text_validator.py` | 文本参与 uplift 前的有效性检查（可选 LLM，失败则本地规则） |
| `utils.py` | 共用工具函数 |

## 2. 配置要点（`config.py`）

默认值以源码为准；文档仅列概念：

- **文本加成**：`DEFAULT_TEXT_BOOST_CONFIG`（`enabled`、`max_total_boost`、相似度门控、`model_paths` 指向 TF-IDF 产物等）。
- **硬阈值示例**：`GPA_MINIMUM`、`LANGUAGE_MINIMUM`、`PROBABILITY_MIN_VALUE` 及仲裁后下限。
- **惩罚/加成上限**：总惩罚率上限、总加成率上限（防止数值发散）。
- **跨专业/学部**：`CROSS_MAJOR_PENALTY_FACTOR`、`FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR` 等。
- **相似度**：`MIN_SIMILARITY_THRESHOLD`、`HIGHER_SIMILARITY_THRESHOLD`（院校数量较少时可用更高阈值，见 `filters.py`）。

## 3. 处理阶段

### 3.1 列表生成阶段（与 `flow/processor.py`、筛选器协同）

- **单项处理**：专业详情中的 IELTS/TOEFL 门槛 vs 用户分数 → 目标相关语言惩罚；注入 `faculty`、中文名等元数据。
- **相似度**：`SimilarityAdjuster` 结合规则文件 `config/similarity_adjustment_rules.json` 与模糊偏置。
- **槽位**：`filters.py` 中 `IDENTITY_MIN_SLOT_RATIO`（如 0.4）保障强匹配比例下限。
- **排序**：综合相似度与 boost（具体公式见代码内注释与 `Selection Score` 实现）。
- **边界探索**：`BoundaryCaseAgent` 调节推荐条数平衡。

### 3.2 批量修正（`ProbabilityAdjustmentPipeline.adjust_batch`）

对每个结果项顺序执行：

1. 通用 GPA/语言惩罚（`ProbabilityAdjuster`）。
2. 动态项：`CrossMajorPenalty`、跨学部惩罚、职业型专业降权（配置中的专业名单与实习条件）。
3. 文本加成：`TextBoostProvider`（Logit 空间增量，见 [text_uplift_api.md](text_uplift_api.md)）。
4. 仲裁与归一化：`AdjustmentArbitrator` 融合与衰减；`NormalizationLayer` 将概率限制在配置区间内（如 `[0.005, 1.0]`）。

### 3.3 文本预检（`experience_text_validator.py`）

在参与 Logit uplift 前：

1. 若配置 OpenAI，可调用 `TextPreprocessingAgent` 做语义有效性判断。
2. 否则本地清洗非中英字符并校验有效长度下限。
3. 未通过预检的字段不参与 uplift 计算。

## 4. 学院兼容（`faculty_filters.py`）

`CROSS_FACULTY_RULES` 定义背景学部与目标学部的允许组合；目标学部不在允许集合时施加惩罚。具体矩阵以源码为准。

## 5. 推荐筛选（`filters.py`）

- **相似专业**：相似度阈值随候选院校数量切换（高/低两档）；`TOP_N_RECOMMENDATIONS` 限制条数；按概率排序。
- **跨专业**：相似度落在配置区间内且历史存在录取组合时入选；跨专业结果需带 `admitted` 标记方可进入 `unified_results`（与 [prediction_api.md](prediction_api.md) 一致）。

---

维护：与 `src/pages/prediction/result_modifier/` 同步更新。
