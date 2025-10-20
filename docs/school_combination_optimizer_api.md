## 学校组合优化模块 API 文档（src/pages/prediction/school_combination_optimizer_algorithm）

本模块用于基于多目标优化（NSGA-III）和约束规则，在给定候选院校专业列表和用户背景下，自动生成若干套“平衡-多样-风险可控”的申请组合方案。

### 模块结构
- **核心优化**：`optimizer.py`（主优化器）、`problem.py`（问题定义）
- **指标与选择**：`metrics_calculator.py`（指标计算）、`school_selector.py`（平衡选择）
- **过滤器**：`filters.py`（含预处理去重、背景分层优先级、跨学院规则过滤）
- **工具与配置**：`config.py`（配置常量与规则）、`utils.py`（公共工具/缓存/自适应阈值/跨专业轻校准）
- **可视化**：`visualizer.py`
- **聚合导出**：`__init__.py`

### 依赖
- 算法与工具：`pymoo`, `numpy`, `pandas`, `scipy`, `concurrent.futures`（并行计算）
- 业务数据：目标专业与原专业相似度缓存、（可选）学校-专业的相关性矩阵（蒙特卡洛使用）

### 输入数据结构（schools 列表项）
- 必需字段：
  - `university: str`
  - `major: str`
  - `probability: float` （已归一化 0–1 的录取概率）
- 可选字段：
  - `is_new_major: bool`（是否新专业）

### 核心类与函数

#### 1) SchoolSelectionOptimizer（optimizer.py）
- 构造：
  - `SchoolSelectionOptimizer(population_size: int = 50, n_generations: int = 50, correlation_matrix: pd.DataFrame | None = None, plan_configs: Iterable[PlanConfig] | None = None, cache_capacity: int = 256)`
  - `population_size`：基础种群大小（内部会按候选规模自适应调整，且不小于参考方向数）
  - `correlation_matrix`：可选的"(大学-专业) × (大学-专业)"相关性矩阵，键名格式：`"{university} - {major}"`
  - `plan_configs`：自定义申请策略列表（可选），不传则使用 `DEFAULT_PLAN_CONFIGS`
  - `cache_capacity`：内部 LRU 缓存容量（指标/仿真缓存）。
- 方法：
  - `optimize(all_schools_data: list[dict[str, Any]], background_major: str, background_faculty: str | None = None, school_level: str | None = None, gpa: float | None = None, major_category_cache: dict | None = None, bg_target_similarity_cache: dict | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]`
    - `background_faculty`：用于跨学院规则过滤与轻度概率校准。
    - 返回：`(final_recommendations, adaptive_thresholds)`
    - `final_recommendations` 为若干方案（默认 3 种“申请策略X”，来源于 `config.py` 的 `PlanConfig`），每个方案包含：
      - `type: str`（策略名）
      - `schools: List[Dict[str, Any]]`（入选院校专业，源自输入并可能被裁剪/替换）
      - `metrics: Dict[str, Any]`（指标见下）
      - `objective_values: List[float]`（5 维目标的数值）
    - `adaptive_thresholds`：动态阈值 `{ 'safety': float, 'target_lower': float }`

稳定性说明：
- 过滤链依次执行（预处理去重 → 澳门院校“一校一专业” → 全局最低相似度 → 跨学院概率轻校准）。一旦某一步改变了候选规模，将停止后续过滤，并在该结果上构造 `SchoolSelectionProblem` 进行优化。
- 当前 NSGA-III 的参考方向、种群与迭代参数为固定配置；若优化失败，将记录日志并回退到启发式平衡方案（`_get_fallback_recommendation`）。

功能增强（2025-10-12）：
- 引入澳门院校“一校一专业”预处理：对于 `config.MACAU_UNIVERSITIES` 中的学校，在同一所大学内仅保留一个专业（见 `filters.deduplicate_universities_by_similarity`），选择准则为先比较条目自带的 `similarity`，若相同再比较 `probability`。

可选：自定义申请策略（plan_config）

```python
from src.pages.prediction.school_combination_optimizer_algorithm.config import PlanConfig
from src.pages.prediction.school_combination_optimizer_algorithm import SchoolSelectionOptimizer

custom_plans = [
    PlanConfig(name="普通", min_schools=5, max_schools=5),
    PlanConfig(name="联申", min_schools=6, max_schools=6),
    PlanConfig(name="高端", min_schools=10, max_schools=10),
]

optimizer = SchoolSelectionOptimizer(
    population_size=48,
    n_generations=60,
    correlation_matrix=your_corr_matrix,
    plan_configs=custom_plans,
)
```
- 指标（metrics，来自 `metrics_calculator.calculate_metrics`，并在需要时叠加仿真结果）：
  - `rejection_probability`：独立假设下全拒概率
  - `admission_probability`：独立假设下至少一所录取概率
  - `diversity`：院校多样性（不同大学计数）
  - `safety_count` / `target_count` / `reach_count`：按阈值分类的数量
  - `balance_score`：与理想配比的接近度（含 prestige 加权）
  - `major_similarity`：与原专业的平均相似度
  - `new_major_ratio` / `new_major_count`：新专业占比 / 数量
  - `simulated_rejection_probability` / `simulated_admission_probability`：相关性仿真估计（如提供相关性矩阵）
- 可视化：
  - `visualize_recommendations(recommendations, adaptive_thresholds)`：快速图形化展示（内部调用 `visualizer.py`）
  - 也可使用实例方法 `SchoolSelectionOptimizer.visualize_recommendations(recommendations, adaptive_thresholds)`，与包级函数等价。
  - 性能：参考方向使用 `reference_direction_cache.py` 做 LRU 缓存；评估阶段复用线程池并发计算；学校优先级查询与指标/蒙特卡洛结果使用 LRU 缓存；首次仅注入一次 CSS。

- 维护：
  - `clear_cache()`：清理内部结果/指标/仿真缓存。

#### 2) SchoolSelectionProblem（problem.py）
- 作用：定义 NSGA-III 的多目标与约束问题。
- 目标（最小化）：
  1. `f1` = 全拒概率
  2. `f2` = -多样性
  3. `f3` = -平衡度
  4. `f4` = -专业相似度
  5. `f5` = 新专业占比（权重为正，目标函数为最小化）
- 约束：
  - `g1`：所选数量 ≤ `max_schools`
  - `g2`：冲刺（reach）数量下限
  - `g3`：目标（target）数量下限
  - `g4`：保底（safety）数量下限
  - `g5`：所选数量 ≥ `min_schools`
  - `g6`：TOPN 学校约束（特定背景需包含指定港校）
  - `g7`：TOP3 学校最小数量（高背景高 GPA）
  - `g8`：TOP5 学校最小数量（高背景高 GPA）
- 其他：
  - 动态阈值 `adaptive_thresholds` 用于 reach/safety 分类
  - 使用相似度缓存与新专业缓存辅助度量
  - 构建专业大类映射缓存

##### 分层硬约束、TOP8 优先与 TOP3/TOP5 最小数量
- 分层硬约束（priority 过滤）：构造 `SchoolSelectionProblem` 时，根据背景校级与 GPA 设定可推荐院校的最高 `priority`，直接在候选集阶段过滤低等级院校。**特别地，对于高背景高 GPA 用户，会强制保留 TOP5 学校，即使其 priority 超过阈值**：
  - `school_level ∈ {"985","211","1-50","51-100"}`：
    - `GPA ≥ 3.2` → 仅保留 `priority ≤ 5`（TOP8 + 101-200）
    - `GPA ≥ 2.8` → 放宽至 `priority ≤ 6`（201-300）
    - 其他 → `priority ≤ 7`（500+）
  - `school_level == "普通本科"`：
    - `GPA ≥ 3.0` → `priority ≤ 5`
    - 其他 → `priority ≤ 7`
  - 兜底：若过滤后为空或不足以满足 `min_schools`，会回退到原候选集以避免无解。
- TOP8 优先（在满足上述分层约束的前提下）：当 `school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2` 时，进一步优先选取 `TOP8_SCHOOLS`；若不足 `min_schools`，仅为补足数量按概率从非 TOP8 候选补齐。
- TOP3/TOP5 最小数量约束：当 `school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2` 时，强制要求：
  - 至少包含 `MIN_TOP3_COUNT_FOR_HIGH_BG`（默认 2）所 TOP3 学校（港大/港中文/港科大/澳大/NTU/NUS）
  - 至少包含 `MIN_TOP5_COUNT_FOR_HIGH_BG`（默认 3）所 TOP5 学校（TOP3 + 港理工/港城大）
  - 目的：确保高背景用户的推荐结果包含足够数量的顶级院校，避免被高概率的一般院校"淹没"

###### TOPN 学校 reach/safety 约束细则（对应 `g6`）
- reach 约束适用：
  - 若 `school_level ∈ {"985","211","1-50","51-100"}` → reach 学校需来自 `TOP3_SCHOOLS`；
  - 若 `school_level == "普通本科"`：
    - `GPA ≥ 3.0` → reach 学校需来自 `TOP3_SCHOOLS`；
    - 其他 → reach 学校需来自 `TOP5_SCHOOLS`，且额外要求 reach 中至少包含 1 所 TOP3（若候选中存在）。
- safety 约束适用：
  - 若 `school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 2.5` → safety 学校需来自 `TOP8_SCHOOLS`；
  - 若 `school_level == "普通本科"` 且 `GPA > 3.0` → safety 学校需来自 `TOP8_SCHOOLS`。
- 仅在“候选中存在可满足条件的选项”时计入违反：
  - reach 的“候选存在性”以 `target_lower` 阈值（自适应）判断 reach 候选是否存在；
  - safety 的“候选存在性”以 `safety` 阈值（自适应）判断 safety 候选是否存在；
  - 若存在但未选入，则违反数量等于违规项数量；若需“至少 1 所 TOP3 reach”而未满足，则在 reach 候选存在 TOP3 时额外+1 违反。

#### 4) 指标计算（metrics_calculator.py）
- `calculate_metrics(schools, background_major, adaptive_thresholds, bg_target_similarity_cache, new_major_cache, background_faculty, major_category_cache) -> dict`
  - 计算全拒概率、多样性、平衡度、专业相似度、新专业占比等指标；如提供相关性矩阵，将在优化器中叠加仿真指标。

#### 5) 学校选择器（school_selector.py）
- `reduce_schools_balanced(schools, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：按平衡比例裁剪学校列表
- `generate_balanced_selection(schools, min_count, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：生成平衡的学校组合

#### 6) 候选过滤器（filters.py）
- `filter_candidates_by_background(all_schools_data, school_level, gpa, min_schools, background_faculty=None, adaptive_thresholds=None) -> list`
  - 根据背景校级和 GPA 进行优先级阈值过滤，并对高背景高 GPA 应用 TOP8 优先与最低保底校数量保障；结合 `adaptive_thresholds` 兜底保底校数量。

#### 7) 预处理过滤器 (filters.py)
- `deduplicate_majors(schools: list[dict[str, Any]]) -> list[dict[str, Any]]`
  - 对同一大学下的相似专业（如 `计算机科学` vs `计算机科学(授课型)`)进行去重，仅保留录取概率最高的一个。

- `deduplicate_universities_by_similarity(schools, background_major, similarity_cache, target_universities) -> list[dict]`
  - 澳门院校“一校一专业”规则：当 `target_universities` 为 `optimizer_config.MACAU_UNIVERSITIES` 时，对每一所澳门大学仅保留一个专业；选择准则为先比较“背景专业→目标专业”的相似度（更高者优先），若相同再比较录取概率（更高者优先）。当无法获取相似度缓存时将保守回退，不进行该裁剪。

#### 8) 跨学院规则过滤器 (filters.py)
- `filter_schools_by_faculty_rules(schools: list, background_faculty: str | None) -> list`
  - 基于 `faculty_rules.py` 中定义的 `CROSS_FACULTY_RULES`，根据用户的背景学院，过滤掉规则不允许申请的目标学院下的所有专业。
  - “专业大类”在此处作为“学院”使用。

#### 9) 问题初始化器（problem_initializer.py）
- `build_major_category_cache(details_df) -> Dict[str, str]`：构建"学校|专业"到"专业大类"（被视为学院）的映射缓存。
- `build_new_major_cache(all_schools_data) -> Dict[str, Any]`：构建新专业缓存

#### 10) 公共工具（utils.py）
- `clip_probability(value, default=0.0) -> float`：概率裁剪到 [0, 1]
- `normalize_school_name(name) -> str`：学校名称归一化
- `calculate_adaptive_thresholds(all_school_probabilities, reach_percentile_val=None, safety_percentile_val=None) -> dict`：自适应阈值计算
- `calibrate_cross_major_probabilities(schools, background_faculty) -> list[dict]`：跨学院轻度概率校准

#### 11) 蒙特卡洛仿真（monte_carlo.py）
- `run_monte_carlo_simulation(selected_schools, correlation_matrix, pair_weight_matrix=None, n_simulations=None, min_simulations=None, max_simulations=None, convergence_threshold=None) -> Tuple[float, float]`
  - 使用 Sobol 序列和相关矩阵，对整体全拒/至少一录取概率做相关性仿真估计；当矩阵缺失或不可逆时，自动回退。
  - 性能：
    - Cholesky 分解结果使用 `@lru_cache` 缓存
    - 按维度 `k` 缓存 Sobol 采样器实例，避免多次构造开销
    - 相同概率组合与相关矩阵切片使用 `_run_monte_carlo_simulation_cached` 结果缓存
  - 可选 `pair_weight_matrix`：当提供时，与子相关矩阵按元素相乘，用于对成对相关性施加业务权重。

#### 12) 自适应阈值（utils.calculate_adaptive_thresholds）
- 根据给定概率分布的分位数估计 `target_lower/safety`，并做边界修正；样本不足时回退到默认阈值。

#### 13) 概率工具（utils.calibrate_cross_major_probabilities）
- 对跨学院项进行轻度概率校准（先验、收缩、乘子），入参与返回均为字典列表，字段包含 `university/major/probability`。

#### 14) 配置常量（config.py）
- 阈值与比例：
  - `SCHOOL_CATEGORY_THRESHOLDS = {'safety': 0.75, 'target_lower': 0.55}`
  - `BALANCE_RATIOS = {'safety': 0.3, 'target': 0.4, 'reach': 0.3}`
  - `ADAPTIVE_THRESHOLD_PERCENTILES = {'reach_percentile_val': 10, 'safety_percentile_val': 70}`
  - `MONTE_CARLO_DEFAULTS = {'n_simulations': 5000, 'min_simulations': 1000, 'max_simulations': 10000, 'convergence_threshold': 0.01, 'batch_size': 500}`
- TOPN 学校名单：`TOP3_SCHOOLS`, `TOP5_SCHOOLS`, `TOP8_SCHOOLS`（成员以 `config.py` 为准）
- 澳门院校集合：`MACAU_UNIVERSITIES`（用于“一校一专业”预处理规则）
- TOP 学校最小数量：
  - `MIN_TOP3_COUNT_FOR_HIGH_BG = 2`：高背景高 GPA 用户的 TOP3 最小数量
  - `MIN_TOP5_COUNT_FOR_HIGH_BG = 3`：高背景高 GPA 用户的 TOP5 最小数量
- 多目标优化权重（`OBJECTIVE_WEIGHTS`）：用于平衡不同优化目标的重要性
  - `rejection_probability`: 0.9
  - `diversity`: 5.0
  - `balance_score`: 1.0
  - `major_similarity`: 2.0
  - `new_major_ratio`: 0.5
- 高背景用户专属配置（`school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2`）：
  - `ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG`
    - `reach_percentile_val`: 20
    - `safety_percentile_val`: 60
  - `BALANCE_RATIOS_HIGH_BG`：平衡比例（更激进的冲刺策略）
    - `safety`: 0.2（降低保底校比例）
    - `target`: 0.4（保持主申比例）
    - `reach`: 0.4（提高冲刺校比例）

#### 15) 跨学院规则 (faculty_rules.py)
- 定义 `CROSS_FACULTY_RULES` 字典，规定了从一个背景学院可以申请的目标学院集合。
  - 例如 ` "文学院": {"文学院", "社会科学学院", "教育学院", "商学院", "艺术学院"}`
- `get_allowed_target_faculties(background_faculty: str) -> set`：获取允许申请的目标学院集合。

#### 补充：跨专业相关配置（optimizer_config.py）
- `GLOBAL_MIN_SIMILARITY`：全局最低专业相似度，低于此值的候选在优化前被过滤。
- `CROSS_MAJOR_CALIBRATION`：跨专业概率轻度校准（先验、收缩参数、分位裁剪、同/跨/严格多组乘子）。
- 其他：`MAJOR_SIMILARITY_WEIGHT`、`CONSTRAINT_FLEXIBILITY`、`PRESTIGE_WEIGHT`。

### 典型调用流程
```python
from src.pages.prediction.school_combination_optimizer_algorithm import SchoolSelectionOptimizer

# 输入：来自预测流水线的候选列表（需包含 probability）
all_schools_data = [
    {"university": "香港大学", "major": "计算机科学", "probability": 0.62},
    {"university": "香港科技大学", "major": "数据科学", "probability": 0.55},
    # ... 更多候选
]

optimizer = SchoolSelectionOptimizer(population_size=50, n_generations=60, correlation_matrix=your_corr_matrix)
recommendations, adaptive_thresholds = optimizer.optimize(
    all_schools_data=all_schools_data,
    background_major="计算机科学",
    background_faculty="工程学院",
    school_level="普通本科",
    gpa=3.2,
)

for rec in recommendations:
    print(rec["type"], len(rec["schools"]))
    print(rec["metrics"])  # 包含 balance/diversity/simulated_* 等指标
```

### 注意事项
- `probability` 必须在 [0, 1]；当候选过少（< min_schools），可能退化到平衡启发式方案。
- 相关性矩阵的索引/列名需为 `"{university} - {major}"`，缺失时将回退为独立假设估计。
- 过滤器（背景、学院、相似度）在构造阶段生效，优化器不会看到被过滤的院校，这可显著提升推荐质量。
- 需确保 `school_base` 的 `school_level/priority` 标注准确；可参考 `logs/page3/prediction/school_level_missing.json` 进行清洗补全。
- 若业务需要放宽/收紧，可在 `optimizer_config.py` 中调整阈值常量（如 `TOP_BG_LEVELS_SET`、`PRIORITY_THRESHOLD_*`）。
- 过滤器与概率校准可能改变候选规模；当前实现会在该步骤后“重建”问题对象并自适应算法参数，避免索引错位导致的越界。
- 澳门院校“一校一专业”：同一所澳门大学若在候选集中包含多个专业，将在预处理阶段自动折叠为一个专业（相似度优先、概率次之）。如需强制保留某特定专业，请在输入候选侧先行过滤。

---
维护人：lijiapeng8@xdf.cn
版本：v2.9
