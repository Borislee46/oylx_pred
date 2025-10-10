## 学校组合优化模块 API 文档（src/pages/prediction/school_combination_optimizer_algorithm）

本模块用于基于多目标优化（NSGA-III）和约束规则，在给定候选院校专业列表和用户背景下，自动生成若干套“平衡-多样-风险可控”的申请组合方案。

### 模块结构
- **核心优化**：`optimizer.py`（主优化器）、`problem.py`（问题定义）
- **指标与选择**：`metrics_calculator.py`（指标计算）、`school_selector.py`（平衡选择）
- **过滤器**：`pre_filter.py`（预处理过滤器）、`candidate_filter.py`（背景分层过滤）、`faculty_based_filter.py`（跨学院规则过滤）
- **初始化**：`problem_initializer.py`（缓存构建）
- **工具与配置**：`common_utils.py`（公共工具）、`optimizer_config.py`（配置常量）、`faculty_rules.py`（学院规则定义）
- **辅助模块**：`monte_carlo.py`、`probability_utils.py`、`adaptive_admission_probability_threshold.py`、`plan_config.py`、`visualizer.py`、`cache_utils.py`、`reference_direction_cache.py`
- **聚合导出**：`utils.py`（向后兼容的导入聚合器）、`__init__.py`

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
  - `optimize(all_schools_data: List[Dict[str, Any]], background_major: str, background_faculty: str | None, school_level: str | None = None, gpa: float | None = None) -> Tuple[List[Dict[str, Any]], Dict[str, float]]`
    - 新增 `background_faculty` 参数，用于基于学院规则进行二次过滤。
    - 返回：`(final_recommendations, adaptive_thresholds)`
    - `final_recommendations` 为若干方案（默认 3 种“申请策略X”，来源于 `plan_config.py` 的 `PlanConfig`），每个方案包含：
      - `type: str`（策略名）
      - `schools: List[Dict[str, Any]]`（入选院校专业，源自输入并可能被裁剪/替换）
      - `metrics: Dict[str, Any]`（指标见下）
      - `objective_values: List[float]`（6 维目标的数值）
    - `adaptive_thresholds`：动态阈值 `{ 'safety': float, 'target_lower': float }`

稳定性改进（2025-10-09）：
- 在进行跨专业过滤与概率轻校准之后，若候选集合规模发生变化，优化器会“重建”一次 `SchoolSelectionProblem`，并据新规模“自适应”参考方向数量、种群规模与迭代代数，确保 `n_var` 与候选长度严格一致，避免解向量与候选索引不匹配（典型报错：list index out of range）。
- 若过滤/校准或优化过程中出现异常，将记录日志并回退到启发式的平衡方案（`_get_fallback_recommendation`）。

可选：自定义申请策略（plan_config）

```python
from src.pages.prediction.school_combination_optimizer_algorithm.plan_config import PlanConfig
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
  - `safety_count`/`target_count`/`reach_count`：按阈值分类的数量
  - `balance_score`：与理想配比（安全:目标:冲刺 = 0.3:0.4:0.3）的接近度（越高越好）
  - `major_similarity`：与原专业的平均相似度
  - `new_major_ratio` / `new_major_count`：新专业占比 / 数量
  - `major_category_score`：专业大类合理性得分（越高越好，基于背景学院与目标学院关系计算）
  - `cross_major_ratio`：跨专业比例
  - `major_category_diversity`：专业大类多样性（不同大类计数）
  - `simulated_rejection_probability` / `simulated_admission_probability`：考虑相关性的蒙特卡洛估计（如果提供相关性矩阵）
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
  5. `f5` = -新专业占比
  6. `f6` = -专业大类合理性
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
  - 动态阈值 `adaptive_thresholds` 来自全局概率分位数（见下）
  - 会使用业务相似度缓存与新专业缓存辅助度量
  - 通过 pandas 构建详情表与专业大类映射缓存。

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
- TOP3/TOP5 最小数量约束（`g9`/`g10`）：当 `school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2` 时，强制要求：
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
- `calculate_metrics(schools, background_major, adaptive_thresholds, bg_target_similarity_cache, new_major_cache, background_major_category, major_category_cache) -> Dict[str, Any]`
  - 计算全拒概率、多样性、平衡度、专业相似度、新专业占比、专业大类得分等指标

#### 5) 学校选择器（school_selector.py）
- `reduce_schools_balanced(schools, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：按平衡比例裁剪学校列表
- `generate_balanced_selection(schools, min_count, max_count, adaptive_thresholds) -> List[Dict[str, Any]]`：生成平衡的学校组合

#### 6) 候选过滤器（candidate_filter.py）
- `filter_candidates_by_background(all_schools_data, school_level, gpa, min_schools) -> List[Dict[str, Any]]`
  - 根据背景校级和 GPA 进行分层硬约束过滤（priority 阈值）
  - 应用 TOP8 优先策略

#### 7) 预处理过滤器 (pre_filter.py)
- `deduplicate_majors(schools: list[dict[str, Any]]) -> list[dict[str, Any]]`
  - 对同一大学下的相似专业（如 `计算机科学` vs `计算机科学(授课型)`)进行去重，仅保留录取概率最高的一个。

#### 8) 跨学院规则过滤器 (faculty_based_filter.py)
- `filter_schools_by_faculty_rules(schools: list, background_faculty: str | None, major_category_cache: dict | None) -> list`
  - 基于 `faculty_rules.py` 中定义的 `CROSS_FACULTY_RULES`，根据用户的背景学院，过滤掉规则不允许申请的目标学院下的所有专业。
  - “专业大类”在此处作为“学院”使用。

#### 9) 问题初始化器（problem_initializer.py）
- `build_major_category_cache(details_df) -> Dict[str, str]`：构建"学校|专业"到"专业大类"（被视为学院）的映射缓存。
- `build_new_major_cache(all_schools_data) -> Dict[str, Any]`：构建新专业缓存

#### 补充（2025-10-09）：当无法从学校-专业详情表中识别出背景专业的大类时，会进行“关键词回退映射”（保守原则）以便后续跨专业规则生效。例如：
- 金融/finance → `金融学`
- 经济/econom/贸（贸易/国际贸易）→ `经济学`
- business/工商 → `工商管理`
- 管理/management → `管理学`
- 会计/account → `会计学`
- 市场/marketing → `市场营销`

#### 10) 公共工具（common_utils.py）
- `clip_probability(value, default=0.0) -> float`：概率裁剪到 [0, 1]
- `normalize_school_name(name) -> str`：学校名称归一化

#### 11) 蒙特卡洛仿真（monte_carlo.py）
- `run_monte_carlo_simulation(selected_schools, correlation_matrix, pair_weight_matrix=None, n_simulations=None, min_simulations=None, max_simulations=None, convergence_threshold=None) -> Tuple[float, float]`
  - 使用 Sobol 序列和相关矩阵，对整体全拒/至少一录取概率做相关性仿真估计；当矩阵缺失或不可逆时，自动回退。
  - 性能：
    - Cholesky 分解结果使用 `@lru_cache` 缓存
    - 按维度 `k` 缓存 Sobol 采样器实例，避免多次构造开销
    - 相同概率组合与相关矩阵切片使用 `_run_monte_carlo_simulation_cached` 结果缓存
  - 可选 `pair_weight_matrix`：当提供时，与子相关矩阵按元素相乘，用于对成对相关性施加业务权重。

#### 12) 自适应阈值（adaptive_admission_probability_threshold.py）
- `calculate_adaptive_thresholds(all_school_probabilities, reach_percentile_val=None, safety_percentile_val=None) -> Dict[str, float]`
  - 根据给定概率分布的分位数估计 `target_lower/safety`，并做边界修正；不足样本时回退到默认阈值。

#### 13) 概率工具（probability_utils.py）
- `calibrate_cross_major_probabilities(schools: List[Dict], background_faculty: str | None, major_category_cache: Dict[str, str] | None, school_cross_major_factor: Dict[str, float] | None = None, calibration_cfg: Dict | None = None) -> List[Dict]`：对跨专业项进行轻度概率校准（先验、收缩、分位裁剪、同/跨/严格多组乘子）；入参与返回均为字典列表，字段包含 `university/major/probability`。

#### 14) 配置常量（optimizer_config.py）
- 阈值与比例：
  - `SCHOOL_CATEGORY_THRESHOLDS = {'safety': 0.8, 'target_lower': 0.6}`
  - `BALANCE_RATIOS = {'safety': 0.3, 'target': 0.4, 'reach': 0.3}`
  - `ADAPTIVE_THRESHOLD_PERCENTILES = {'reach_percentile_val': 10, 'safety_percentile_val': 70}`
  - `MONTE_CARLO_DEFAULTS = {'n_simulations': 5000, 'min_simulations': 1000, 'max_simulations': 10000, 'convergence_threshold': 0.01, 'batch_size': 500}`
- TOPN 学校名单：`TOP3_SCHOOLS`, `TOP5_SCHOOLS`, `TOP8_SCHOOLS`（业务定义集合，用于高背景高 GPA 的保底与优先策略；成员以 `optimizer_config.py` 为准）
- TOP 学校最小数量：
  - `MIN_TOP3_COUNT_FOR_HIGH_BG = 2`：高背景高 GPA 用户的 TOP3 最小数量
  - `MIN_TOP5_COUNT_FOR_HIGH_BG = 3`：高背景高 GPA 用户的 TOP5 最小数量
- 多目标优化权重（`OBJECTIVE_WEIGHTS`）：用于平衡不同优化目标的重要性
  - `rejection_probability`: 1.0（拒录概率）
  - `diversity`: 2.5（多样性，提高以增加学校档次多样性）
  - `balance_score`: 2.0（平衡度，包含 prestige）
  - `major_similarity`: 1.0（专业相似度）
  - `new_major_ratio`: 0.5（新专业比例）
  - `major_category_score`: 1.0（专业大类得分）
- 高背景用户专属配置（`school_level ∈ {"985","211","1-50","51-100"}` 且 `GPA ≥ 3.2`）：
  - `ADAPTIVE_THRESHOLD_PERCENTILES_HIGH_BG`：自适应阈值百分位
    - `reach_percentile_val`: 15（提高 reach 阈值）
    - `safety_percentile_val`: 85（提高 safety 阈值，降低保底校强制要求）
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

#### 16) 包导出（__init__.py）
- `SchoolSelectionProblem`, `SchoolSelectionOptimizer`, `visualize_recommendations`, `run_monte_carlo_simulation`

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
    background_faculty="工程学院", # 新增
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

---
维护人：lijiapeng8@xdf.cn
版本：v2.8
更新日期：2025-10-10

更新记录：
- 2025-10-10 v2.8:
  - 新增基于学院（Faculty）的跨专业申请规则过滤。
  - 优化流程中增加候选专业去重与全局相似度预过滤。
  - 更新优化问题的约束条件，移除已废弃的约束。
  - 同步更新 API 文档的模块结构、函数签名与核心逻辑描述。
- 2025-10-09 v2.7：
  - 优化器在过滤/校准后重建 `Problem` 并自适应算法参数，修复规模变动引起的潜在越界。
  - 初始化器新增商科关键词回退映射，确保商科背景能正确触发法学等严格大类的排除规则。
