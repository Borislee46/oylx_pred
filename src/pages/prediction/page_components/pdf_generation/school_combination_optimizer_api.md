<!-- !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files. -->
## 学校组合优化模块 API 文档（src/pages/prediction/school_combination_optimizer_algorithm）

本模块用于基于多目标优化（NSGA-III）和约束规则，在给定候选院校专业列表和用户背景下，自动生成若干套“平衡-多样-风险可控”的申请组合方案。

### 模块结构
- **核心优化器**：`optimizer/optimizer.py`（主优化器类 `SchoolSelectionOptimizer`）
- **优化执行**：`optimizer/optimization_runner.py`（NSGA-III 算法执行）、`optimizer/solution_selector.py`（解选择）
- **问题定义**：`problem.py`（`SchoolSelectionProblem` 多目标优化问题）
- **上下文管理**：`optimizer/context.py`（`OptimizationContext` 上下文对象）
- **过滤器链**：`optimizer/filters_handler.py`（过滤器链编排）、`filters.py`（具体过滤器实现）
- **指标计算**：`optimizer/metrics_calculator_wrapper.py`（指标计算包装）、`metrics_calculator.py`（核心指标计算）
- **推荐构建**：`optimizer/recommendation_builder.py`（推荐结果构建）、`optimizer/school_adjuster.py`（学校调整）
- **阈值计算**：`optimizer/threshold_calculator.py`（自适应阈值计算）
- **缓存管理**：`optimizer/cache_manager.py`（缓存管理工具）
- **学校选择**：`school_selector.py`（平衡选择算法）
- **工具与配置**：`config.py`（配置常量与规则）、`utils.py`（公共工具/缓存/自适应阈值/跨专业轻校准）
- **蒙特卡洛**：`monte_carlo.py`（相关性仿真）
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

#### 1) SchoolSelectionOptimizer（optimizer/optimizer.py）
- 构造：
  - `SchoolSelectionOptimizer(population_size: int = 50, n_generations: int = 50, correlation_matrix: pd.DataFrame | None = None, plan_configs: Iterable[PlanConfig] | None = None, cache_capacity: int = 256)`
  - `population_size`：基础种群大小（默认 50，内部会按候选规模自适应调整，且不小于参考方向数）
  - `n_generations`：基础迭代代数（默认 50，会根据问题规模自适应调整）
  - `correlation_matrix`：可选的"(大学-专业) × (大学-专业)"相关性矩阵，键名格式：`"{university} - {major}"`，用于蒙特卡洛仿真
  - `plan_configs`：自定义申请策略列表（可选），不传则使用 `DEFAULT_PLAN_CONFIGS`（默认 3 种策略：6/9/10 所学校）
  - `cache_capacity`：内部 LRU 缓存容量（默认 256），包含三种缓存类型：`result`（结果缓存）、`metric`（指标缓存）、`simulation`（仿真缓存）
- 方法：
  - `optimize(all_schools_data: list[dict[str, Any]], background_major: str, background_faculty: str | None = None, school_level: str | None = None, gpa: float | None = None, major_category_cache: dict | None = None, bg_target_similarity_cache: dict | None = None) -> tuple[list[dict[str, Any]], dict[str, float]]`
    - `all_schools_data`：候选院校专业列表（必需）
    - `background_major`：用户背景专业（必需）
    - `background_faculty`：用户背景学院（可选），用于跨学院规则过滤与轻度概率校准
    - `school_level`：用户背景校级（可选），如 `"985"`, `"211"`, `"普通本科"` 等
    - `gpa`：用户 GPA（可选）
    - `major_category_cache`：专业大类缓存（可选），不提供时自动加载
    - `bg_target_similarity_cache`：背景专业到目标专业的相似度缓存（可选）
    - 返回：`(final_recommendations, adaptive_thresholds)`
    - `final_recommendations` 为若干方案（默认 3 种"申请策略X"，来源于 `config.py` 的 `DEFAULT_PLAN_CONFIGS`），每个方案包含：
      - `type: str`（策略名，如 "申请策略1"）
      - `schools: List[Dict[str, Any]]`（入选院校专业，源自输入并可能被裁剪/替换，已进行概率调整）
      - `metrics: Dict[str, Any]`（指标见下）
      - `objective_values: List[float]`（5 维目标的数值）
    - `adaptive_thresholds`：动态阈值 `{ 'safety': float, 'target_lower': float }`，用于分类 safety/target/reach
  - `clear_cache() -> None`：清理内部所有 LRU 缓存（result/metric/simulation）
  - `visualize_recommendations(recommendations: list[dict[str, Any]], adaptive_thresholds: dict[str, float]) -> None`：可视化推荐结果（调用 `visualizer.py`）

优化流程说明：
1. **输入验证**：检查 `all_schools_data` 是否为空，为空则返回空结果
2. **缓存检查**：根据输入参数构建 hash key，检查结果缓存，命中则直接返回
3. **上下文构建**：创建 `OptimizationContext` 对象，包含所有优化上下文信息
4. **自适应阈值计算**：根据用户背景和 GPA 计算动态阈值（高背景高 GPA 使用不同分位数）
5. **概率调整**：对选校池进行概率纠偏（`adjust_probability_by_university_difficulty`）
6. **多策略优化**：对每个 `plan_config` 执行优化：
   - 应用过滤器链（见下方过滤器链说明）
   - 创建 `SchoolSelectionProblem` 问题对象
   - 运行 NSGA-III 优化（参数自适应调整）
   - 选择最佳解并构建推荐结果
   - 若优化失败，回退到启发式平衡方案
7. **后处理**：对最终推荐结果进行概率调整和指标计算
8. **结果缓存**：将结果存入缓存

过滤器链执行顺序（`optimizer/filters_handler.py`）：
- `deduplicate_majors`：同一大学下的相似专业去重（保留概率最高者）
- `similarity_filter`：澳门院校"一校一专业"规则（相似度优先，概率次之）
- `similarity_filter_only`：全局最低相似度过滤（`GLOBAL_MIN_SIMILARITY`，默认 0）
- `probability_calibration`：跨学院概率轻度校准（收缩+乘子）

若某过滤器将候选集过滤为空且该过滤器不允许空结果，则停止后续过滤器。

#### 2) OptimizationContext（optimizer/context.py）
- 作用：封装优化过程中的所有上下文信息
- 字段：
  - `all_schools_data: list[dict[str, Any]]`：候选院校专业列表
  - `background_major: str`：背景专业
  - `background_faculty: Optional[str]`：背景学院
  - `school_level: Optional[str]`：背景校级
  - `gpa: Optional[float]`：GPA
  - `adaptive_thresholds: Optional[dict[str, float]]`：自适应阈值
  - `problem: Optional[SchoolSelectionProblem]`：当前优化问题对象
  - `major_category_cache: Optional[dict]`：专业大类缓存

#### 自定义申请策略（plan_config）

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

#### 3) SchoolSelectionProblem（problem.py）
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
  - 构建专业大类映射缓存（`major_category_cache`）
  - 在构造时自动应用 `filter_candidates_by_background` 进行背景分层过滤
  - 高背景高 GPA 判断：`school_level in {"985","211","1-50","51-100"} and gpa >= 3.2`

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

#### 4) 指标计算（metrics_calculator.py / optimizer/metrics_calculator_wrapper.py）
- `calculate_metrics(schools, background_major, adaptive_thresholds, bg_target_similarity_cache, new_major_cache, background_faculty, major_category_cache) -> dict`
  - 计算全拒概率、多样性、平衡度、专业相似度、新专业占比等指标
  - 如提供相关性矩阵，将在 `metrics_calculator_wrapper.py` 中叠加蒙特卡洛仿真指标
  - 返回指标字典包含：`rejection_probability`, `admission_probability`, `diversity`, `safety_count`, `target_count`, `reach_count`, `balance_score`, `major_similarity`, `new_major_ratio`, `new_major_count`, `simulated_rejection_probability`, `simulated_admission_probability`

#### 5) 学校选择器（school_selector.py）
- `reduce_schools_balanced(schools, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：按平衡比例裁剪学校列表
- `generate_balanced_selection(schools, min_count, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：生成平衡的学校组合（用于 fallback 方案）

#### 6) 概率调整（optimizer/school_adjuster.py）
- `adjust_probability_by_university_difficulty(schools, adaptive_thresholds) -> list[dict[str, Any]]`：根据学校难度对概率进行纠偏调整
- `enforce_school_limits(schools, min_schools, max_schools, adaptive_thresholds) -> list[dict[str, Any]]`：强制执行学校数量限制

#### 7) 候选过滤器（filters.py）
- `filter_candidates_by_background(all_schools_data, school_level, gpa, min_schools, background_faculty=None, adaptive_thresholds=None) -> list`
  - 根据背景校级和 GPA 进行优先级阈值过滤，并对高背景高 GPA 应用 TOP8 优先与最低保底校数量保障；结合 `adaptive_thresholds` 兜底保底校数量
  - 内部调用 `_determine_max_allowed_priority` 确定优先级阈值
  - 高背景高 GPA 用户强制保留 TOP5 学校（即使 priority 超过阈值）
  - 应用 TOP8 优先策略（若适用）
  - 确保最低保底校数量（`MIN_SAFETY_SCHOOL_COUNT_DEFAULT` 或 `MIN_SAFETY_SCHOOL_COUNT_HIGH_BG`）

#### 8) 预处理过滤器 (filters.py)
- `deduplicate_majors(schools: list[dict[str, Any]]) -> list[dict[str, Any]]`
  - 对同一大学下的相似专业（如 `计算机科学` vs `计算机科学(授课型)`)进行去重，使用 `major_norm` 字段进行匹配，仅保留录取概率最高的一个

- `deduplicate_universities_by_similarity(schools, background_major, similarity_cache, target_universities) -> list[dict]`
  - 澳门院校"一校一专业"规则：当 `target_universities` 为 `config.MACAU_UNIVERSITIES` 时，对每一所澳门大学仅保留一个专业
  - 选择准则：先比较条目自带的 `similarity` 字段（更高者优先），若相同再比较 `probability`（更高者优先）
  - 当无法获取相似度缓存时将保守回退，不进行该裁剪

#### 9) 跨学院规则过滤器 (filters.py)
- `filter_schools_by_faculty_rules(schools: list, background_faculty: str | None) -> list`
  - 基于 `config.py` 中定义的 `CROSS_FACULTY_RULES`，根据用户的背景学院，过滤掉规则不允许申请的目标学院下的所有专业
  - "专业大类"在此处作为"学院"使用
  - 若学校没有 `faculty` 字段或为空，则保留该学校

#### 10) 工具函数（utils.py）
- `build_major_category_cache(details_df) -> Dict[str, str]`：构建"学校|专业"到"专业大类"（被视为学院）的映射缓存
- `build_new_major_cache(all_schools_data) -> Dict[str, Any]`：构建新专业缓存，键为 `"{university}|{major}"`
- `normalize_major_name(major_name: str | None) -> str`：归一化专业名称（去除括号内容）
- `normalize_school_name(name: str | None) -> str`：归一化学校名称（去除空格、特殊字符）
- `get_cached_reference_directions(method: str, n_dim: int, n_points: int)`：获取缓存的参考方向（用于 NSGA-III）
- `clip_probability(value, default=0.0) -> float`：概率裁剪到 [0, 1]

#### 11) 自适应阈值计算（utils.py / optimizer/threshold_calculator.py）
- `calculate_adaptive_thresholds(all_school_probabilities, reach_percentile_val=None, safety_percentile_val=None) -> dict`：根据概率分布的分位数计算自适应阈值
  - `reach_percentile_val`：默认 10（高背景高 GPA 为 20）
  - `safety_percentile_val`：默认 70（高背景高 GPA 为 60）
  - 样本不足时回退到默认阈值 `SCHOOL_CATEGORY_THRESHOLDS`
- `calculate_adaptive_thresholds_for_context(context, safe_execute) -> dict`：基于上下文计算自适应阈值（考虑高背景高 GPA 情况）

#### 12) 概率校准（utils.py）
- `calibrate_cross_major_probabilities(schools, background_faculty) -> list[dict]`：跨学院轻度概率校准
  - 对跨学院项应用收缩+乘子：`(shrinkage_alpha * prob + (1 - shrinkage_alpha) * prior_p) * cross_faculty_multiplier`
  - `shrinkage_alpha = 0.5`, `prior_p = 0.1`, `cross_faculty_multiplier = 0.85`
  - 同学院项保持不变

#### 13) 蒙特卡洛仿真（monte_carlo.py）
- `run_monte_carlo_simulation(selected_schools, correlation_matrix, pair_weight_matrix=None, n_simulations=None, min_simulations=None, max_simulations=None, convergence_threshold=None) -> Tuple[float, float]`
  - 使用 Sobol 序列和相关矩阵，对整体全拒/至少一录取概率做相关性仿真估计；当矩阵缺失或不可逆时，自动回退。
  - 性能：
    - Cholesky 分解结果使用 `@lru_cache` 缓存
    - 按维度 `k` 缓存 Sobol 采样器实例，避免多次构造开销
    - 相同概率组合与相关矩阵切片使用 `_run_monte_carlo_simulation_cached` 结果缓存
  - 可选 `pair_weight_matrix`：当提供时，与子相关矩阵按元素相乘，用于对成对相关性施加业务权重。

#### 14) 优化执行（optimizer/optimization_runner.py）
- `create_problem(schools_data, plan_config, context) -> SchoolSelectionProblem`：创建优化问题对象
- `compute_algo_params(problem_size, population_size, n_generations) -> tuple[int, int, Any]`：根据问题规模自适应计算算法参数
  - `problem_size < 30`：`pop = max(base_pop, problem_size * 2)`
  - `problem_size < 50`：`pop = max(base_pop, int(base_pop * 1.2))`, `n_gen = int(base_gen * 1.2)`
  - 其他：使用基础参数
- `run_optimization(problem, population_size, n_generations, safe_execute) -> Optional[Result]`：执行 NSGA-III 优化
  - 使用 `NSGA3` 算法，`BinaryRandomSampling` 采样，`HUX` 交叉，`BitflipMutation` 变异
  - 参考方向数量：`DEFAULT_REFERENCE_DIRECTIONS_COUNT = 42`
  - 使用 `safe_execute` 包装，优化失败时返回 `None`

#### 15) 缓存管理（optimizer/cache_manager.py）
- `build_optimization_input_hash(...) -> str`：构建优化输入的 hash key（用于结果缓存）
- `get_cached_data(caches, cache_type, key, calculation_func) -> Any`：从缓存获取数据，未命中则计算并缓存
- `clear_all_caches(caches) -> None`：清理所有缓存

#### 16) 配置常量（config.py）
- 阈值与比例：
  - `SCHOOL_CATEGORY_THRESHOLDS = {'safety': 0.75, 'target_lower': 0.55}`：默认分类阈值
  - `BALANCE_RATIOS = {'safety': 0.3, 'target': 0.4, 'reach': 0.3}`：默认平衡比例
  - `BALANCE_RATIOS_HIGH_BG = {'safety': 0.2, 'target': 0.4, 'reach': 0.4}`：高背景高 GPA 平衡比例
  - `ADAPTIVE_THRESHOLD_PERCENTILES = {'reach_percentile_val': 10, 'safety_percentile_val': 70}`：默认自适应阈值分位数
  - `ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG = {'reach_percentile_val': 20, 'safety_percentile_val': 60}`：高背景高 GPA 自适应阈值分位数
  - `MONTE_CARLO_DEFAULTS = {'n_simulations': 5000, 'min_simulations': 1000, 'max_simulations': 10000, 'convergence_threshold': 0.01, 'batch_size': 500}`：蒙特卡洛默认参数
  - `SAFETY_SCHOOL_THRESHOLD = 0.7`：保底校阈值
  - `MIN_SAFETY_SCHOOL_COUNT_DEFAULT = 3`：默认最低保底校数量
  - `MIN_SAFETY_SCHOOL_COUNT_HIGH_BG = 1`：高背景高 GPA 最低保底校数量
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
- 优化算法默认参数：
  - `DEFAULT_POPULATION_SIZE = 48`：默认种群大小
  - `DEFAULT_N_GENERATIONS = 60`：默认迭代代数
  - `DEFAULT_REFERENCE_DIRECTIONS_COUNT = 42`：默认参考方向数量
- 高背景用户专属配置（`school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2`）：
  - `ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG`
    - `reach_percentile_val`: 20
    - `safety_percentile_val`: 60
  - `BALANCE_RATIOS_HIGH_BG`：平衡比例（更激进的冲刺策略）
    - `safety`: 0.2（降低保底校比例）
    - `target`: 0.4（保持主申比例）
    - `reach`: 0.4（提高冲刺校比例）

#### 17) 跨学院规则 (config.py)
- `CROSS_FACULTY_RULES: dict[str, set[str]]`：定义了从一个背景学院可以申请的目标学院集合
  - 例如 `"文学院": {"文学院", "社会科学院", "教育学院", "商学院", "艺术学院"}`
  - 例如 `"工程学院": {"工程学院", "理学院", "商学院", "计算机学院", "建筑学院", "设计学院", "科学学院"}`
- `get_allowed_target_faculties(background_faculty: str | None) -> set[str]`：获取允许申请的目标学院集合，未找到则返回空集

#### 18) 其他配置
- `GLOBAL_MIN_SIMILARITY = 0`：全局最低专业相似度，低于此值的候选在优化前被过滤（默认 0，即不过滤）
- `PRESTIGE_WEIGHT = 3.0`：声望权重（用于平衡度计算）
- `CONSTRAINT_FLEXIBILITY: dict[str, bool | float | int] = {}`：约束灵活性配置（如 `enable_strict_hk_constraint`）
- `DEFAULT_PLAN_CONFIGS`：默认申请策略配置
  - `PlanConfig(name="申请策略1", min_schools=6, max_schools=6)`
  - `PlanConfig(name="申请策略2", min_schools=9, max_schools=9)`
  - `PlanConfig(name="申请策略3", min_schools=10, max_schools=10)`

### 典型调用流程
```python
from src.pages.prediction.school_combination_optimizer_algorithm import SchoolSelectionOptimizer

# 输入：来自预测流水线的候选列表（需包含 probability）
all_schools_data = [
    {"university": "香港大学", "major": "计算机科学", "probability": 0.62},
    {"university": "香港科技大学", "major": "数据科学", "probability": 0.55},
    # ... 更多候选
]

# 创建优化器（使用默认参数）
optimizer = SchoolSelectionOptimizer(
    population_size=50,
    n_generations=60,
    correlation_matrix=your_corr_matrix,  # 可选
    cache_capacity=256  # 可选
)

# 执行优化
recommendations, adaptive_thresholds = optimizer.optimize(
    all_schools_data=all_schools_data,
    background_major="计算机科学",
    background_faculty="工程学院",  # 可选
    school_level="普通本科",  # 可选
    gpa=3.2,  # 可选
    major_category_cache=None,  # 可选，不提供时自动加载
    bg_target_similarity_cache=None,  # 可选
)

# 处理结果
for rec in recommendations:
    print(f"策略: {rec['type']}, 学校数量: {len(rec['schools'])}")
    print(f"指标: {rec['metrics']}")  # 包含 balance/diversity/simulated_* 等指标
    print(f"目标值: {rec['objective_values']}")  # 5 维目标值

# 可视化（可选）
optimizer.visualize_recommendations(recommendations, adaptive_thresholds)

# 清理缓存（可选）
optimizer.clear_cache()
```

### 注意事项
- **输入要求**：
  - `probability` 必须在 [0, 1]，否则会被 `clip_probability` 裁剪
  - `all_schools_data` 为空时直接返回空结果
  - 候选过少（< min_schools）时，可能退化到平衡启发式方案（fallback）
- **相关性矩阵**：
  - 索引/列名格式需为 `"{university} - {major}"`
  - 缺失时将回退为独立假设估计（不进行蒙特卡洛仿真）
  - 矩阵不可逆时会自动回退
- **过滤器**：
  - 过滤器在优化前生效，优化器不会看到被过滤的院校
  - 过滤器链可能改变候选规模，问题对象会在过滤后重建
  - 算法参数会根据过滤后的候选规模自适应调整
- **数据质量**：
  - 需确保 `school_base` 的 `school_level/priority` 标注准确
  - 可参考 `logs/page3/prediction/school_level_missing.json` 进行清洗补全
  - 学校名称会进行归一化处理（去除空格、特殊字符）
- **配置调整**：
  - 若业务需要放宽/收紧，可在 `config.py` 中调整阈值常量（如 `TOP_BG_LEVELS_SET`、`PRIORITY_THRESHOLD_*`）
  - 可通过 `CONSTRAINT_FLEXIBILITY` 控制约束的严格程度
- **澳门院校规则**：
  - 同一所澳门大学若在候选集中包含多个专业，将在预处理阶段自动折叠为一个专业
  - 选择准则：相似度优先、概率次之
  - 如需强制保留某特定专业，请在输入候选侧先行过滤
- **缓存机制**：
  - 结果缓存基于输入参数的 hash key，相同输入会直接返回缓存结果
  - 指标和仿真结果也会被缓存，提升重复计算性能
  - 可通过 `clear_cache()` 手动清理缓存
- **性能优化**：
  - 参考方向使用 LRU 缓存（`get_cached_reference_directions`）
  - 蒙特卡洛仿真的 Cholesky 分解结果使用 `@lru_cache` 缓存
  - 学校优先级查询使用 `@lru_cache` 缓存

### 内部实现细节

#### 优化器内部方法（optimizer/optimizer.py）
- `_safe_execute(operation, fallback_operation, error_message)`：安全执行操作，捕获异常并记录日志
- `_get_cached_data_wrapper(cache_type, key, calculation_func)`：缓存数据包装器
- `_calculate_adaptive_thresholds(context)`：计算自适应阈值
- `_apply_all_filters(schools_data, context)`：应用所有过滤器
- `_create_problem(schools_data, plan_config, context)`：创建优化问题
- `_run_optimization(problem)`：运行优化算法
- `_calculate_metrics(selected_schools, context, problem)`：计算指标
- `_find_best_solution_indices(res, problem, min_schools, limit)`：查找最佳解索引
- `_build_final_recommendation(x, f_values, problem, context, plan_config)`：构建最终推荐
- `_adjust_probability_by_university_difficulty(schools, adaptive_thresholds)`：概率调整
- `_get_fallback_recommendation(context, plan_config)`：获取 fallback 推荐
- `_optimize_single_plan(plan_config, context)`：优化单个策略

#### 推荐构建（optimizer/recommendation_builder.py）
- `build_final_recommendation(...)`：从优化解构建最终推荐
  - 应用后过滤器（`apply_post_filters`）
  - 强制执行学校数量限制（`enforce_school_limits`）
  - 调整概率（`adjust_probability_by_university_difficulty`）
  - 计算指标（包含蒙特卡洛仿真）
- `get_fallback_recommendation(...)`：生成 fallback 推荐（使用平衡选择算法）
- `get_fallback_recommendation_with_filtered_schools(...)`：基于过滤后的学校生成 fallback 推荐

#### 解选择（optimizer/solution_selector.py）
- `find_best_solution_indices(res, problem, context, min_schools, limit)`：从优化结果中选择最佳解索引
  - 优先选择满足约束的可行解
  - 按目标值排序选择最优解

---
维护人：lijiapeng8@xdf.cn
版本：v3.0
