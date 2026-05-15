# Result Modifier 技术文档

## 1. 模块概述

`result_modifier` 是预测页面的结果修饰模块，负责对模型输出的录取概率进行多维度调整，包括 GPA/语言惩罚、跨专业惩罚、学部过滤、背提文本加成、相似度调整等，并支持基于 Agent 的边界案例微调。

## DS 视角

这是整个预测系统**校准问题的核心**——5层乘法惩罚链。每层在 `DECISIONS.md` 有独立的设计决策（DEC-002 GPA惩罚、DEC-003 语言惩罚、DEC-004 跨专业/学部），但**联合效应从未被系统设计过**。DEC-007 加了衰减仲裁（0.85/layer）是事后补救。整条链把一个 Platt 校准过的概率（XGBoost + sigmoid calibration）变成未校准的概率（ECE=0.1155），方向反了。

关键数据事实：99.4%的组合≤4个样本，模型在极端稀疏区域做外推，调整链的参数（0.15二次系数、0.89相似度阈值、0.5σ pass_line）全部在充足数据的假设下调出来的，在小样本区域的行为未经验证——外部数据-67pp就是证据。

## 2. 目录结构

```
result_modifier/
├── __init__.py
├── config.py                 # 配置常量
├── types.py                  # 类型定义
├── utils.py                  # 工具函数
├── streamlit_cache.py        # 缓存装饰器
├── fallback.py               # 🆕 预测失败兜底 (cascading population stats + Wilson CI, TODO-1 Item5)
├── counterfactual.py          # 反事实扰动分析（"如果背景调整..."）
├── adjustment_pipeline.py    # 概率调整流水线（主入口）
├── probability_adjuster.py  # GPA/语言惩罚（L1-L5 六档 + GPA 二次函数）
├── arbitrator.py             # 多因子仲裁（0.85 衰减 + 70% 上限）
├── similarity_adjuster.py   # 相似度调整（模糊匹配、规则）
├── language_penalty.py       # 语言要求惩罚
├── filters.py                # 推荐过滤（同专业/跨专业）
├── faculty_filters.py        # 学部过滤规则（CROSS_FACULTY_RULES）
├── admission_cache.py        # 录取组合缓存
├── text_boost_provider.py    # 背提文本加成接口
├── experience_text_validator.py  # 背提文本 LLM 校验（含批量模式）
├── engine.py                 # Agent 调整引擎
├── strategies.py             # 排名策略（Relax/Tighten）
├── ranker.py                 # Agent 排名调整入口
├── ui_handler.py             # 加载提示与 UI 交互
└── providers/                # 背提加成实现（LogitUpliftProvider）
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
- **语言惩罚**：低于 pass_line 离散阶梯惩罚（L1/L2/L3/L3.5/L4/L5），基于历史 `mean - 0.5*std` 为 pass_line。2026-05 新增 L3.5（0.20），将 6.0-6.5 雅思密集区拆分为两档，区分 6.0（L3, 0.40）和 6.3+（L3.5, 0.20）。用户行为分析显示 89% 的提交集中在 6.0-6.5，此改动让该区间从 1 个 bucket 变为 2 个，提升区分度。完整档位：L1(0.95) / L2(0.80) / L3(0.40) / L3.5(0.20) / L4(0.10) / L5(0.05)
- **综合分**：`0.4*gpa_score + 0.3*lang_score + 0.3*school_score`
- **选择分**：`similarity * (1 + boost)`，综合分高且目标院校难时加成

#### 4.2.1 批判性问题

- **GPA 用二次函数（0.15×z²），语言用阶梯函数——为什么不同？** 函数形式选择有 DS 层面的统一逻辑吗？还是纯粹历史遗留（GPA 惩罚的开发者偏好连续、语言惩罚的开发者偏好分档）？二次函数在 z=2 时惩罚 60%、在 z=1 时只有 15%，这个非线性程度的依据是什么？
- **pass_line = mean - 0.5σ：0.5 从哪来的？** 是 grid search 最优值还是经验拍板？有没有跑过 0.3σ 或 0.8σ 的敏感性测试？
- **GPA max penalty 0.8，语言 severe threshold 0.95——量级差距合理吗？** 语言极端低（<5.4 雅思）直接惩罚 95%，GPA 极端低（<2.0）只惩罚 95%——两者对录取的影响真的有接近的量级吗？教育学研究一般显示 GPA 的预测力高于语言成绩。
- **中位数填充缺失值：缺失 GPA 本身可能是信号。** 不填 GPA 的学生很可能 GPA 偏低，用中位数反而给他们一个"正常"起点。有没有对比过缺失 GPA 学生和已知低 GPA 学生的实际录取率？

### 4.3 AdjustmentArbitrator

多因子仲裁，将多个 Penalty/Boost 合并为最终概率。

- **惩罚**：按 value 降序，逐项乘以衰减 `PENALTY_DECAY_FACTOR`，总惩罚不超过 `MAX_TOTAL_PENALTY_RATIO`
- **加成**：同理，总加成不超过 `MAX_TOTAL_BOOST_RATIO`
- **公式**：`final = base * (1 - total_penalty) * (1 + total_boost)`
- **trace**：记录各因子贡献，用于调试

#### 4.3.1 设计问题

- **0.85 衰减系数**：2 个惩罚 ≈ 0.85 效果，3 个 ≈ 0.72 效果。但如果触发全部 5 层惩罚（GPA 低 + 语言低 + 跨专业 + 跨学部 + 无实习申 MBA）——第五层的边际贡献只有 0.85⁴ ≈ 52%。这是有意的"次要问题该弱化"还是意外的参数行为？
- **MAX_TOTAL_PENALTY_RATIO=0.70 制造了天花板**：所有输出 ≤ base_prob × 0.7 + boost。这等于系统说"最大把握是 72%"。清华 GPA 4.0 申珠海学院本专业也拿不到 >0.72——这个设计选择在业务上是否合理？
- **按 severity 排序后衰减**：GPA 惩罚可能排第一（因为值最大），但业务上跨学部跨度可能是更根本的障碍。severity 排序反映的是惩罚值的量级，不是因果重要性。

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

### 4.9 Fallback (🆕 TODO-1 Item 5)

当 XGBoost 因数据缺失无法运行时（缺少 GPA + 语言成绩），提供基于人口统计的兜底估算。

- **DataCompleteness**：四级分类 — `complete` / `degraded` / `minimal` / `insufficient`
- **Cascading Fallback**：从精确匹配逐级降级到全局录取率，每层 n≥5 阈值
  - Level 0: (bg_uni, bg_major, target_uni, target_major)
  - Level 1: (bg_uni, target_uni, target_major)
  - Level 2: (target_uni, target_major)
  - Level 3: (target_uni)
  - Level 4: 全局录取率
- **Wilson Score Interval**：95% CI，小样本下不溢出 [0,1]
- **可用调整层照跑**：跨学部、跨专业、职业学位、文本提升 — 不依赖 GPA/Language
- **UI 标识**：`_is_fallback=True` 标记，展示 ⚠️ warning 明确区分"历史统计估算"与"模型预测"

### 4.10 ExperienceTextValidator

背提文本有效性校验，调用 `TextPreprocessingAgent.validate_field`，支持 LLM 与本地规则，校验通过后合并文本长度 ≥ 3 才视为有效。

### 4.11 StreamlitCache

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

## DS Known Issues

- **ECE 从校准态退化到未校准态**：Platt scaling 校准后 ECE≈0.026，经过 5 层调整链变成 0.1155。调整链把校准过的概率变得不再校准——方向反了。
- **外部数据 -67pp**：ApplySquare 上的偏差是训练集的 7 倍，因为相似度匹配质量下降 → 更多跨专业惩罚被触发 → 乘法衰减放大。调整链对外部数据的脆弱性没有在设计时考虑。
- **不对称影响**：C9 被低估 18pp、双非被低估 6pp——乘法惩罚对不同基础概率区间的影响天然不对称，这不是 bug 而是乘法运算的数学特性。
- **静态惩罚批处理**：`adjustment_pipeline.py` 中 GPA/Language 惩罚以 `is_static=True` 加入——所有目标学校共享同一个 GPA 惩罚值。但 GPA 3.0 对港大 CS 和对珠海学院 CS 的录取影响应该是不同的——惩罚值不应完全相同。

## 7. 测试

仓库根目录 `tests/test_result_modifier.py`：`case_key` / `deduplicate_results`、`AdjustmentArbitrator` 惩罚上限、`RelaxStrategy` / `TightenStrategy` 的 triage 与 fuzzy bypass。

## 8. 依赖

- `pandas`, `numpy`, `numba`：数值计算
- `rapidfuzz`：模糊匹配
- `streamlit`：UI 与缓存（可选）
- `joblib`：模型加载（providers）

