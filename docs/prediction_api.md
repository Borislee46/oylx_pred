## 预测模块 API 文档（src/pages/prediction）

### 模块结构
- 类型定义：`src/pages/prediction/prediction_types.py`
- 资源加载：`src/pages/prediction/page_data_loader.py`
- 模型封装：`src/pages/prediction/prediction_model.py`
- 组合与结果处理：`src/pages/prediction/prediction_processor.py`
- 并行推理器：`src/pages/prediction/run_prediction.py`
- 并行执行器：`src/pages/prediction/prediction_execution/prediction_executor.py`
- 输入准备：`src/pages/prediction/prediction_preparation/`
- 结果调整：`src/pages/prediction/prediction_result_adjuster.py`
- 结果处理：`src/pages/prediction/results_handler.py`
- 预测管线：`src/pages/prediction/prediction_pipeline.py`
- 统一编排入口：`src/pages/prediction/prediction_handler.py`
- 结果模型定义：`src/utils/session_manager.py`（`PredictionResultModel`）
- 结果展示：`src/pages/prediction/result_display/results_display.py`

### 数据类型（src/pages/prediction/prediction_types.py）
- `PredictionInput`（TypedDict，total=False，所有字段可选）
  - `background_university: str`
  - `background_major: str`
  - `target_universities: list[str]`
  - `target_majors: list[str]`
  - `gpa: float`
  - `language_score: float`
  - `internship_count: int`
  - `research_count: int`
  - `award_count: int`
  - `paper_count: int`
  - `school_level: int`  # 类型定义中为 int，由 `prepare_input_data` 通过 `school_level_service` 注入
  - `experience_details: dict[str, str]`
  - `faculty: str`  # 运行时由 `prepare_input_data` 注入（如果可用）
- 结果项字典（运行期结构）
  - 基本字段：`university: str`, `major: str`, `probability: float`, `similarity: float`
  - 处理阶段注入：`chinese_name: str`、`faculty: str`、`is_new_major: bool`
  - 内部元数据（合并去重时使用）：`_source: str`（"similarity" | "cross_major" | "user_specified"）、`_priority: int`
- `MetaInfo = dict[str, Any]`
  - 当前包含：`combination_count: int`

### 资源加载（src/pages/prediction/page_data_loader.py）
- `cached_get_prediction_model(model_name: str) -> PredictionModel`（`@st.cache_resource`）
  - 说明：加载并缓存模型与全局类别映射（来自 `utils.app_data_loader`）。
- `cached_load_cases_data() -> pd.DataFrame`（`@st.cache_data`）
  - 说明：加载 `cases_min.feather`（压缩后的数据集，只取app必要的表头）；由 `utils.app_data_loader.load_raw_cases_data` 完成。
- `cached_load_bg_target_similarity_cache() -> dict`（`@st.cache_data`）
  - 说明：加载并缓存"原专业-目标专业"相似度缓存（默认从根目录 `cache/background_target_similarity.feather` 读取）。
- `@dataclass machine_learning_model`
  - `prediction_model: Any`
  - `loaded_feature_names: list[str]`
  - `cases_df: pd.DataFrame`
  - `@classmethod resource_loader(cls) -> machine_learning_model`（`@st.cache_resource(show_spinner=False)`）
    - 说明：统一加载模型、特征列表和案例数据的资源类。

### 模型封装（src/pages/prediction/prediction_model.py）
- `class PredictionModel(model_type: str, global_categories_df: pd.DataFrame | None)`
  - 初始化：通过 `load_model(model_type)` 加载模型、特征名和 `level_fallback_mapping`；建立全局类别映射（`CATEGORICAL_COLUMNS`）；检查是否支持分类特征（XGBoost `enable_categorical`）。
  - 属性：
    - `model`: 训练好的模型实例
    - `feature_names: list[str] | None`: 模型特征列表
    - `level_fallback_mapping: dict`: 学校等级到后备学校的映射
    - `global_categories: dict[str, list]`: 分类列的全局类别列表
    - `global_category_index: dict[str, dict[str, int]]`: 类别值到索引的映射
    - `school_level_service`: 学校等级服务实例
  - `predict_batch(input_data: dict[str, Any], combinations: list[tuple[str, str]], expected_features: list[str]) -> list[dict[str, Any]]`
    - 批量推理：对每个 `(university, major)` 组合预测概率。
    - 预处理：
      - 计算并缓存基础特征（`_get_preprocessed_base_features`）。
      - 自动编码分类特征（使用全局类别映射或索引映射）。
      - 对计数列（`COUNT_COLUMNS_FOR_LOG_TRANSFORM`）进行 `log1p` 变换。
      - 未知学校使用 `level_fallback_mapping` 回退。
    - 输出：每个结果包含 `university`、`major`、`probability`。
    - 异常：输入验证失败抛出 `InvalidInputError`；模型预测失败记录日志并重新抛出。
  - `predict_probability(input_df: pd.DataFrame) -> float | None`
    - 单样本概率预测；返回 `None` 如果模型未初始化。

### 组合与结果处理（src/pages/prediction/prediction_processor.py）
- `ProcessingContext` (Dataclass): 封装处理所需的上下文数据（如 `cases_df`、`probability_adjuster`、`gpa` 等）。
- `generate_prediction_combinations(input_data: PredictionInput, all_universities_target: list[str], all_majors_target: list[str]) -> tuple[list[tuple[str, str]], dict[str, int]]`
  - 纯函数：生成有效 `(university, major)` 组合（过滤无详情项）。
  - 策略：
    - 如果估计总数 ≤ 100：遍历所有组合，使用 `has_school_major_details` 过滤。
    - 如果估计总数 > 100：使用 `get_valid_school_major_set` 预过滤，然后生成有效组合。
  - 返回：`(valid_combinations, meta)`，其中 `meta['combination_count']: int`。
- `process_prediction_results(...)`
  - 核心处理函数，接收批量预测结果及 `ProcessingContext` 中的参数。
  - 流程：
    1. 预处理 (`_preprocess_results`)：
       - 过滤 `part-time` 专业。
       - 批量注入 `similarity`（使用缓存）。
       - 批量注入 `faculty`（使用映射）。
    2. 生成推荐 (`_generate_recommendations`)：
       - 院系过滤（如果提供了 `background_faculty`）。
       - 生成 `top_similarity_results` (相似推荐)。
       - 生成 `top_cross_major_results` (跨专业推荐)。
    3. 结果平衡 (`_apply_agent_balance_adjustment`)：
       - 使用 `BoundaryCaseAgent` 动态调整相似专业与跨专业结果的平衡。
       - 基于 `balance_diff` 和相似度阈值进行微调。
    4. 用户指定结果处理 (`_get_user_specified_results`)：
       - 根据组合数量阈值和相似度阈值进行过滤和排序。
  - 返回：`(top_similarity_results, top_cross_major_results, final_user_specified_results)`。

### 并行推理器（src/pages/prediction/run_prediction.py）
- `run_single_prediction(...)`
  - 参数更新：新增 `probability_adjuster`, `gpa`, `language_score`, `background_university` 等可选参数用于结果处理。
  - 步骤：
    1. 构建 `PredictionInput` 并调用 `generate_prediction_combinations` → `(combinations, meta)`。
    2. 使用 `prepare_model_inputs` 准备模型输入特征。
    3. 创建 `PredictionExecutor` 并调用 `execute_parallel` 进行并发批量推理。
    4. 对结果做稳定排序（概率降序→学校→专业）。
    5. 使用 `get_user_specified_combinations` 获取用户指定组合。
    6. 调用 `process_prediction_results` 处理为三类结果（包含 Agent 平衡逻辑）。
  - 返回：`(top_similarity_results, top_cross_major_results, user_specified_results, meta)`。
  - 异常处理：捕获异常并记录日志，返回空结果列表。

### 并行执行器（src/pages/prediction/prediction_execution/prediction_executor.py）
- `class PredictionExecutor(total_tasks: int)`
  - `get_execution_strategy() -> tuple[type | None, int, int]`
    - 策略选择：
      - `< 128 任务：单线程执行，chunk_size = total_tasks`。
      - `128-512 任务：单线程执行，chunk_size = total_tasks`。
      - `≥ 512 任务：根据环境变量选择线程池或进程池，worker 数 = min(4, cpu_count)`，chunk_size 自适应。
    - 环境变量：
      - `PREDICTION_USE_PROCESS_POOL=1`：启用进程池（否则使用线程池）。
      - `PREDICTION_MAX_WORKERS`：限制最大并发数。
  - `execute_parallel(...)`
    - 将组合列表分块，根据策略选择执行方式。
    - 进程池：使用 `run_prediction_chunk_in_process`，需要 `init_worker_process` 初始化。
    - 线程池：使用 `run_prediction_chunk`，直接传递模型实例。
    - 子任务超时：120 秒；异常记录日志但不阻断整体。

### 输入准备（src/pages/prediction/prediction_preparation/）
- `prepare_model_inputs(...)`
  - 提取基础特征（排除 `target_university` 和 `target_major`），返回 `(model_input_features, missing_inputs)`。
- `get_user_specified_combinations(...)`
  - 从会话中获取用户指定的专业和类别，结合 `target_universities` 生成组合列表。

### 编排入口（src/pages/prediction/prediction_handler.py）
- `run_prediction_with_guard(...)`
  - 带异常保护的预测执行；准备输入数据（包含指纹计算），调用 `run_prediction_pipeline`，更新 Session 状态。
- `handle_form_submission(...)`
  - 处理表单提交：
    - 验证必需字段。
    - 执行跨院系检查 (`quick_cross_faculty_check`)，必要时弹窗确认。
    - 准备输入数据、执行预测。
    - 记录首次提交日志。

### 预测管线（src/pages/prediction/prediction_pipeline.py）
- `run_prediction_pipeline(...)`
  - 顶层管线（`@st.cache_data(ttl=600, show_spinner=True)`）：
    1. 加载模型（`cached_get_prediction_model`）和案例数据。
    2. 计算数据指纹并校验。
    3. 初始化 `ProbabilityAdjuster`（如果 GPA 和语言成绩可用）。
    4. 调用 `run_single_prediction` 获取三类结果（传入调节器和分数信息）。
    5. 行业专业规则修正：`adjust_for_professional_majors`。
    6. 跨专业惩罚：`penalize_cross_major_without_cases`。
    7. 文本加成准备：检查是否有有效经历文本，获取 `text_boost_provider`。
    8. 批量调整：`batch_adjust_results` 统一处理概率调整、文本加成和新专业标记。
    9. 合并去重：`combine_and_deduplicate_results` → `unified_results`。
    10. 返回 `PredictionResultModel`。

### 结果调整（src/pages/prediction/prediction_result_adjuster.py）
- `pipeline_adjust_results(...)`
  - 依次应用概率调整、文本加成和新专业标记；概率值限制在 [0, 1] 范围内。
- `batch_adjust_results(...)`
  - 批量处理多个结果列表，统一构建 `is_new_major_cache` 以提高效率。

### 结果处理（src/pages/prediction/results_handler.py）
- `combine_and_deduplicate_results(sim_results, cross_results, user_specified_results) -> list[dict]`
  - 合并三类结果并去重：
    - 优先级：`user_specified` (3) > `cross_major` (2) > `similarity` (1)。
    - 跨专业结果仅保留 `admitted == 1` 的项。
    - 相同 `(university, major)` 组合，优先保留高优先级或同优先级下概率更高的结果。
    - 移除内部元数据字段。
- `reset_prediction_results(...)`
  - 重置预测结果和相关的会话状态。

### 结果展示（src/pages/prediction/result_display/results_display.py）
- `class ResultsDisplay`
  - 初始化：接收三类结果列表，配置结果类型字典。
  - `display(...)`
    - 展示三类结果为可交互表格。
    - 功能：显示概率、中文名称、专业详情链接、"New!" 标记。
    - 排序：优先按固定院校顺序，其次同校内按概率降序。
    - 策略：根据组合池大小自适应展示逻辑。

### 文本加成（TF‑IDF Logit Uplift）
- 提供方：`result_modifier/text_boost_provider.py`。
- 核心：`providers/logit_uplift_provider.py`。
- 逻辑：基于经历文本与预设文案的 TF-IDF 相似度，对概率进行 Logit 空间的增益。
- 配置：支持动态封顶、平滑系数等配置。

### 结果模型（src/utils/session_manager.py）
- `@dataclass class PredictionResultModel`
  - `similarity_results: list[dict] | None`
  - `cross_major_results: list[dict] | None`
  - `user_specified_results: list[dict] | None`
  - `unified_results: list[dict] | None`
  - `meta: dict[str, Any] | None` (包含 `combination_count` 等)

### 缓存、性能与一致性
- 资源/数据加载：使用 `@st.cache_resource` 和 `@st.cache_data`。
- 管线缓存：`run_prediction_pipeline` 缓存 10 分钟。
- 并行推理：自适应线程/进程池策略。
- 结果指纹：基于输入数据和配置指纹保证缓存有效性。

### 错误与返回约定
- 关键资源缺失：页面层展示错误，API 返回空结构。
- 预测失败：记录日志，尽可能返回部分结果或空结果，不阻断 UI。
