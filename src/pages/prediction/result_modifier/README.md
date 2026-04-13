# Result Modifier 技术文档

对外模块说明与组件索引见 [result_modifier_api.md](../../../../docs/result_modifier_api.md)；文本 Logit uplift 理论与训练产物见 [text_uplift_api.md](../../../../docs/text_uplift_api.md)。本文保留目录内类型、仲裁公式、`AdjustmentContext` 等与代码逐行对照的细节；预测全链路见 [../README.md](../README.md)。

## 1. 模块概述

`result_modifier` 是预测页面的结果修饰模块，负责对模型输出的录取概率进行多维度调整，包括 GPA/语言惩罚、跨专业惩罚、学部过滤、背提文本加成、相似度调整等，并支持基于 Agent 的边界案例微调。

## 2. 目录结构

```
result_modifier/
├── __init__.py
├── config.py                 # 配置常量
├── types.py                  # 类型定义
├── utils.py                  # 工具函数
├── streamlit_cache.py        # 缓存装饰器
├── adjustment_pipeline.py    # 概率调整流水线（主入口）
├── probability_adjuster.py  # GPA/语言惩罚
├── arbitrator.py             # 多因子仲裁
├── similarity_adjuster.py   # 相似度调整（模糊匹配、规则）
├── language_penalty.py       # 语言要求惩罚
├── filters.py                # 推荐过滤（同专业/跨专业）
├── faculty_filters.py        # 学部过滤规则
├── admission_cache.py        # 录取组合缓存
├── text_boost_provider.py    # 背提文本加成接口
├── experience_text_validator.py  # 背提文本 LLM 校验
├── engine.py                 # Agent 调整引擎
├── strategies.py             # 排名策略（Relax/Tighten）
├── ranker.py                 # Agent 排名调整入口
├── ui_handler.py             # 加载提示与 UI 交互
└── providers/                # 背提加成实现
    ├── README.md
    ├── logit_uplift_provider.py
    └── logit_uplift/
```

## 3. 核心流程

```
原始预测结果 (university, major, similarity, probability, ...)
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. 相似度调整 (similarity_adjuster)                               │
│    - 规则匹配: background_keywords + target_keywords → adjustment  │
│    - 模糊匹配: rapidfuzz → bias multiplier                       │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. 过滤与排序 (filters)                                           │
│    - 同专业: get_similar_major_recommendations (相似度阈值)        │
│    - 跨专业: get_cross_major_recommendations (录取组合 + 学部)     │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 概率调整流水线 (ProbabilityAdjustmentPipeline.adjust_batch)    │
│    a) 静态惩罚: GPA Penalty, Language Penalty (ProbabilityAdjuster)│
│    b) 单条调整 (adjust_single):                                    │
│       - Cross Major Penalty (相似度 < 阈值)                        │
│       - Faculty Out of Scope Penalty (学部跨度过大)                 │
│       - Professional Major Penalty (商科等缺实习)                  │
│    c) AdjustmentArbitrator: 多因子合并 (惩罚衰减 + 加成衰减)        │
│    d) 背提文本加成: TextBoostProvider.apply (LogitUpliftProvider)  │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. Agent 边界微调 (可选, ranker.adjust_similarity_results_with_agent)│
│    - RelaxStrategy: 放宽阈值，补充边界案例                         │
│    - TightenStrategy: 收紧阈值，移除尾部案例                       │
│    - AgentAdjustmentEngine: 调用 Agent 决策是否调整                │
└─────────────────────────────────────────────────────────────────┘
        │
        ▼
最终推荐结果 (按 probability 排序)
```

## 4. 组件说明

### 4.1 ProbabilityAdjustmentPipeline

主入口，负责批量概率调整。

**AdjustmentContext** 上下文字段：

| 字段 | 类型 | 说明 |
|------|------|------|
| gpa | float | GPA |
| language_score | float | 语言成绩（归一化） |
| background_university | str | 本科院校 |
| background_major | str | 本科专业 |
| background_faculty | str | 本科学部 |
| internship_count | int | 实习数量 |
| user_specified_majors | list | 用户指定专业 |
| experience_details | dict | 背提文本 |
| cases_df | DataFrame | 历史案例 |
| admitted_combinations | set | 录取组合 (univ, major) |
| is_new_major_cache | dict | 新专业标记缓存 |

**流程**：`adjust_batch` → 静态惩罚 → `adjust_single`（逐条）→ 背提加成。

### 4.2 ProbabilityAdjuster

基于历史案例统计计算 GPA、语言惩罚。

- **GPA 惩罚**：低于 minimum 直接严重惩罚；在 mean 以下用二次函数
- **语言惩罚**：低于 pass_line 分档惩罚（L1/L2/L3）
- **综合分**：`0.4*gpa_score + 0.3*lang_score + 0.3*school_score`
- **选择分**：`similarity * (1 + boost)`，综合分高且目标院校难时加成

### 4.3 AdjustmentArbitrator

多因子仲裁，将多个 Penalty/Boost 合并为最终概率。

- **惩罚**：按 value 降序，逐项乘以衰减 `PENALTY_DECAY_FACTOR`，总惩罚不超过 `MAX_TOTAL_PENALTY_RATIO`
- **加成**：同理，总加成不超过 `MAX_TOTAL_BOOST_RATIO`
- **公式**：`final = base * (1 - total_penalty) * (1 + total_boost)`
- **trace**：记录各因子贡献，用于调试

### 4.4 SimilarityAdjuster

- **规则**：从 `similarity_adjustment_rules.json` 加载，`background_keywords` + `target_keywords` 匹配时加 `adjustment`
- **模糊偏置**：`rapidfuzz.token_sort_ratio` 计算专业名相似度；阈值与倍率由 `config.FUZZY_BIAS_THRESHOLD_HIGH/MID/LOW` 与 `FUZZY_BIAS_MULTIPLIER_*` 控制（默认约 >92/82/72 对应 1.25/1.15/1.05）

### 4.5 Filters

- **get_similar_major_recommendations**：同专业推荐，按相似度阈值过滤，结合 `selection_score` 排序；强匹配由 `_strong_match_score > FUZZY_BIAS_THRESHOLD_HIGH` 判定（与 `strategies.check_fuzzy_bypass` 一致）
- **get_cross_major_recommendations**：跨专业推荐，需历史录取组合，且相似度在 `[CROSS_MAJOR_SIMILARITY_MIN, MIN_SIMILARITY_THRESHOLD)`，学部过滤

### 4.6 FacultyFilters

`CROSS_FACULTY_RULES` 定义学部可跨范围，如文学院可跨社会科学院、教育学院等。`is_faculty_out_of_scope` 判断目标学部是否超出范围。

### 4.7 TextBoostProvider / LogitUpliftProvider

背提文本加成，详见 `providers/README.md`。

### 4.8 Agent 调整 (engine / strategies / ranker)

- **RelaxStrategy**：当前结果不足时，从边界区补充案例，相似度在 `[lower_bound, threshold)` 的案例交由 Agent 决策是否加入
- **TightenStrategy**：结果过多时，从尾部移除案例，相似度低于 `HIGHER_SIMILARITY_THRESHOLD` 的案例交由 Agent 决策是否移除
- **AgentAdjustmentEngine**：选取候选 → 展示 UI → 调用 Agent 评估 → 按决策更新结果（当前为单轮 `run`；`RankerStrategy.update_boundary_cases` / `get_exploration_candidates` 预留多轮扩展）

### 4.9 ExperienceTextValidator

背提文本有效性校验，调用 `TextPreprocessingAgent.validate_field`，支持 LLM 与本地规则，校验通过后合并文本长度 ≥ 3 才视为有效。

### 4.10 StreamlitCache

`cache_data` / `cache_resource`：在 Streamlit 环境下委托 `st.cache_data` / `st.cache_resource`，非 Streamlit 时退化为无缓存。

## 5. 类型定义 (types.py)

- **CaseKey**：`tuple[str, str]`，即 `(university, major)`
- **is_case_with_key**：判断 `dict` 是否同时含 `str` 类型的 `university`、`major`
- **case_key**：合法时返回 `CaseKey`，否则返回 `None`（不抛 `KeyError`）；集合与去重逻辑应对 `None` 显式判断
- **AdjustmentFactor**：`name`, `value`, `factor_type` (PENALTY/BOOST), `description`, `weight`
- **AdjustmentDecision**：`DEFER_TO_AGENT`, `ADJUST`, `NO_ADJUST`

## 6. 配置要点 (config.py)

| 常量 | 说明 |
|------|------|
| MIN_SIMILARITY_THRESHOLD | 同专业相似度下限 |
| HIGHER_SIMILARITY_THRESHOLD | 高相似度阈值 |
| CROSS_MAJOR_SIMILARITY_MIN | 跨专业最低相似度 |
| FACULTY_OUT_OF_SCOPE_PENALTY_FACTOR | 学部超范围惩罚系数 |
| PROFESSIONAL_REDUCTION_FACTOR | 商科等缺实习惩罚 |
| MAX_TOTAL_PENALTY_RATIO / MAX_TOTAL_BOOST_RATIO | 仲裁总惩罚/加成上限 |
| PENALTY_DECAY_FACTOR / BOOST_DECAY_FACTOR | 因子衰减系数 |
| FUZZY_BIAS_THRESHOLD_HIGH / MID / LOW | 模糊匹配与强匹配门控（与 filters、strategies 共用 HIGH） |

- **`prediction_rules.json`**：由 `load_prediction_rules()` 读取；文件存在但损坏或不可读时会打 **warning** 日志并回退内置默认（如 `PROFESSIONAL_MAJORS` 列表）。

## 7. 测试

仓库根目录 `tests/test_result_modifier.py`：`case_key` / `deduplicate_results`、`AdjustmentArbitrator` 惩罚上限、`RelaxStrategy` / `TightenStrategy` 的 triage 与 fuzzy bypass。

## 8. 依赖

- `pandas`, `numpy`, `numba`：数值计算
- `rapidfuzz`：模糊匹配
- `streamlit`：UI 与缓存（可选）
- `joblib`：模型加载（providers）

