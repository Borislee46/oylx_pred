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
    - 预处理：自动编码分类特征（使用全局类别映射或索引映射）；对计数列（`COUNT_COLUMNS_FOR_LOG_TRANSFORM`）进行 `log1p` 变换；未知学校使用 `level_fallback_mapping` 回退。
    - 输出：每个结果包含 `university`、`major`、`probability`。
    - 异常：输入验证失败抛出 `InvalidInputError`；模型预测失败记录日志并重新抛出。
  - `predict_probability(input_df: pd.DataFrame) -> float | None`
    - 单样本概率预测；返回 `None` 如果模型未初始化。

### 组合与结果处理（src/pages/prediction/prediction_processor.py）
- `generate_prediction_combinations(input_data: PredictionInput, all_universities_target: list[str], all_majors_target: list[str]) -> tuple[list[tuple[str, str]], dict[str, int]]`
  - 纯函数：生成有效 `(university, major)` 组合（过滤无详情项）。
  - 策略：
    - 如果估计总数 ≤ 100：遍历所有组合，使用 `has_school_major_details` 过滤。
    - 如果估计总数 > 100：使用 `get_valid_school_major_set` 预过滤，然后生成有效组合。
  - 返回：`(valid_combinations, meta)`，其中 `meta['combination_count']: int`。
- `process_prediction_results(results: list, background_major: str, bg_target_similarity_cache: dict, num_target_universities: int, cases_df: pd.DataFrame | None = None, user_specified_combinations: list[tuple[str, str]] | None = None, background_faculty: str | None = None) -> tuple[list, list, list]`
  - 流程：
    1. 过滤 `part-time` 专业（名称同时包含 "part" 与 "time"）。
    2. 批量计算并注入 `similarity`（使用 `get_cached_major_similarities_batch` 和 `adjust_similarity_score`）。
    3. 批量注入 `chinese_name` 和 `faculty`（通过 `_attach_chinese_names_batch`）。
    4. 当提供 `background_faculty` 时，使用 `filter_schools_by_faculty_rules` 过滤。
    5. 生成三类结果：
       - `top_similarity_results`: 通过 `get_similar_major_recommendations` 获取。
       - `top_cross_major_results`: 通过 `get_cross_major_recommendations` 获取。
       - `final_user_specified_results`: 通过 `_get_user_specified_results` 获取（根据组合数量阈值和相似度阈值过滤）。
  - 返回：`(top_similarity_results, top_cross_major_results, final_user_specified_results)`，每个结果项至少包含：`university`、`major`、`probability`、`similarity`、`chinese_name`、`faculty`。

### 并行推理器（src/pages/prediction/run_prediction.py）
- `run_single_prediction(current_input_data: dict[str, Any], prediction_model: PredictionModel, cases_df: pd.DataFrame, bg_target_similarity_cache: dict[str, float], expected_features: list[str], all_universities_target: list[str], all_majors_target: list[str], num_target_universities: int) -> tuple[list[dict], list[dict], list[dict] | None, None]`
  - 步骤：
    1. 构建 `PredictionInput` 并调用 `generate_prediction_combinations` → `(combinations, meta)`；将 `meta` 写入会话（`SessionManager.set(**meta)`）。
    2. 使用 `prepare_model_inputs` 准备模型输入特征。
    3. 创建 `PredictionExecutor` 并调用 `execute_parallel` 进行并发批量推理。
    4. 对结果做稳定排序（概率降序→学校→专业）。
    5. 使用 `get_user_specified_combinations` 获取用户指定组合。
    6. 调用 `process_prediction_results` 处理为三类结果。
  - 返回：`(top_similarity_results, top_cross_major_results, user_specified_results, None)`。
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
  - `execute_parallel(prediction_model: PredictionModel, combinations: list[tuple[str, str]], model_input_features: dict[str, float | int | str], expected_features: list[str]) -> list[dict[str, float | str]]`
    - 将组合列表分块，根据策略选择执行方式。
    - 进程池：使用 `run_prediction_chunk_in_process`，需要 `init_worker_process` 初始化。
    - 线程池：使用 `run_prediction_chunk`，直接传递模型实例。
    - 子任务超时：120 秒；异常记录日志但不阻断整体。

### 输入准备（src/pages/prediction/prediction_preparation/）
- `prepare_model_inputs(current_input_data: dict[str, Any], expected_features: list[str]) -> tuple[dict[str, float | int | str], list[str]]`
  - 提取基础特征（排除 `target_university` 和 `target_major`），返回 `(model_input_features, missing_inputs)`。
- `get_user_specified_combinations(current_input_data: dict[str, Any], all_universities_target: list[str], session_manager: SessionManager) -> list[tuple[str, str]] | None`
  - 从会话中获取用户指定的专业和类别，结合 `target_universities` 生成组合列表；如果无用户指定则返回 `None`。

### 编排入口（src/pages/prediction/prediction_handler.py）
- `validate_model_and_features(prediction_model: PredictionModel | None) -> list[str] | None`
  - 校验模型与特征名；如果模型为 `None` 或特征列表为空，页面层会展示中文错误提示并返回 `None`。
- `prepare_input_data(input_data_from_form: dict) -> dict`
  - 验证必需字段（`background_university`、`background_major`），缺失时抛出 `InvalidInputError`。
  - 注入 `school_level`（通过 `school_level_service.get_school_level`）。
  - 如果 `background_major` 可用，注入 `faculty`（通过 `get_background_faculty`）。
- `run_prediction_with_guard(session_manager, page_state, current_input_data: dict, all_universities_target: list[str], all_majors_target: list[str], session_key_has_predicted: str, session_key_predict_lock: str) -> bool`
  - 带异常保护的预测执行；成功返回 `True`，失败返回 `False` 并重置结果。
- `handle_form_submission(session_manager, page_state, input_data_from_form: dict, all_universities_target: list[str], all_majors_target: list[str], ...) -> None`
  - 处理表单提交：验证必需字段、跨院系确认、准备输入数据、执行预测。

### 预测管线（src/pages/prediction/prediction_pipeline.py）
- `run_prediction_pipeline(
    input_data: dict[str, Any],
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: list[str],
    all_universities_fingerprint: tuple[int, int],
    all_majors_fingerprint: tuple[int, int],
) -> PredictionResultModel`
  - 顶层管线（`@st.cache_data(ttl=600, show_spinner=True)`）：
    1. 加载模型（`cached_get_prediction_model`）和案例数据（`_get_cached_cases_df`，`ttl=3600`）。
    2. 调用 `run_single_prediction` 获取三类结果。
    3. 行业专业规则修正：`adjust_for_professional_majors`（基于 `internship_count` 和 `user_specified_majors`）。
    4. 跨专业惩罚：`penalize_cross_major_without_cases`（主要作用于用户指定组合）。
    5. 概率调整：当提供 `gpa` 与 `language_score` 时，使用 `ProbabilityAdjuster` 做概率校正。
    6. 文本加成：如果 `experience_details` 包含有意义文本，使用 `get_text_boost_provider` 应用文本加成。
    7. 批量调整：`batch_adjust_results` 统一处理概率调整、文本加成和新专业标记。
    8. 合并去重：`combine_and_deduplicate_results` → `unified_results`。
    9. 返回 `PredictionResultModel`。

### 结果调整（src/pages/prediction/prediction_result_adjuster.py）
- `pipeline_adjust_results(results: list[dict], probability_adjuster: ProbabilityAdjuster | None, text_boost_provider: TextBoostProvider | None, experience_details: dict[str, str], gpa: float | None, language_score: float | None, background_university: str | None, is_new_major_cache: dict[tuple[str, str], bool] | None = None) -> list[dict]`
  - 依次应用概率调整、文本加成和新专业标记；概率值限制在 [0, 1] 范围内。
- `batch_adjust_results(results_list: list[list[dict]], ...) -> list[list[dict]]`
  - 批量处理多个结果列表，统一构建 `is_new_major_cache` 以提高效率。

### 结果处理（src/pages/prediction/results_handler.py）
- `combine_and_deduplicate_results(sim_results, cross_results, user_specified_results) -> list[dict]`
  - 合并三类结果并去重：
    - 优先级：`user_specified` (3) > `cross_major` (2) > `similarity` (1)。
    - 跨专业结果仅保留 `admitted == 1` 的项。
    - 相同 `(university, major)` 组合，优先保留高优先级或同优先级下概率更高的结果。
    - 移除内部元数据字段（`_source`、`_priority`）。
- `reset_prediction_results(session_manager: SessionManager) -> None`
  - 重置预测结果和相关的会话状态。

### 结果展示（src/pages/prediction/result_display/results_display.py）
- `class ResultsDisplay(top_similarity_results=None, top_cross_major_results=None, user_specified_results=None)`
  - 初始化：接收三类结果列表，配置结果类型字典（包含标题和 UI 配置）。
  - `display(target_universities, target_majors, background_university=None, background_major=None) -> None`
    - 展示三类结果为可交互表格，默认列：`目标院校`、`目标专业`、`录取概率`（进度条）、`专业中文名称`；在连续预测上下文中可展示 `变化` 列（概率变化）。
    - 当结果项带有 `is_new_major=True` 时，在 `目标专业` 后追加 "(New!)" 并在渲染时高亮。
    - 排序规则：优先按固定院校顺序，其次同校内按概率降序。
    - 大组合池（>100）时优先展示 Top N 的相似/跨专业结果；用户指定组合在组合池较小时展示。
  - 内部组件：
    - `DataFrameBuilder`: 构建结果 DataFrame。
    - `DataFrameStyler`: 样式化 DataFrame。
    - `DeltaCalculator`: 计算概率变化。
    - `LayoutManager`: 管理布局展示。

### 文本加成（TF‑IDF Logit Uplift）
- 提供方：`result_modifier/text_boost_provider.py`（外层门控与缓存）与 `providers/logit_uplift_provider.py`（核心增益计算）。
- 生效条件（门控）：当四段文本相似度向量 `s` 满足 `sum(s) ≥ sim_gate_sum_min` 且 `max(s) ≥ sim_gate_max_min` 时生效。
- 增益方式：对 `logit(p)` 加上经平滑后的非负增量，仅对中段概率区间（0.2–0.8）生效，并受动态封顶约束。
- 封顶策略：`cap_boost = max_total_boost × cap_factor(质量) × (1 − 2×|p − 0.5|)`；其中质量分 `quality = 0.7×max(s)+0.3×mean(s)`，`cap_factor = clamp(cap_min_factor, 1.0, quality^cap_quality_gamma)`。
- 关键配置（`src/pages/prediction/result_modifier/config.py`）：
```json
{
  "enabled": true,
  "max_total_boost": 0.15,
  "sim_gate_sum_min": 0.10,
  "sim_gate_max_min": 0.08,
  "smoothing": 0.7,
  "cap_min_factor": 0.10,
  "cap_quality_gamma": 1.2,
  "model_paths": {
    "tfidf_vectorizer": "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib",
    "tfidf_centroids": "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz",
    "text_uplift_weights": "src/machine_learning_models/pre-trained_models/text_uplift_weights.json"
  }
}
```
- 训练与线上一致性（TF‑IDF 向量器）：`analyzer='char_wb'`, `ngram_range=(2,4)`, `min_df=1`, `max_features=20000`；详见 `docs/ml_training_api.md` 的“文本加成训练”。

### 结果模型（src/utils/session_manager.py）
- `@dataclass class PredictionResultModel`
  - `similarity_results: list[dict] | None`
  - `cross_major_results: list[dict] | None`
  - `user_specified_results: list[dict] | None`
  - `unified_results: list[dict] | None`
  - 说明：所有结果字段均为字典列表，而非 DataFrame。

### 缓存、性能与一致性
- 资源/数据加载：
  - `cached_get_prediction_model`: `@st.cache_resource`（资源缓存，不设 TTL）。
  - `cached_load_cases_data`: `@st.cache_data`（数据缓存）。
  - `cached_load_bg_target_similarity_cache`: `@st.cache_data`（数据缓存）。
  - `_get_cached_cases_df`: `@st.cache_data(ttl=3600)`（案例数据缓存，1小时）。
- 编排函数：`run_prediction_pipeline` 使用 `@st.cache_data(ttl=600, show_spinner=True)`（10分钟缓存）。
- 并行推理：
  - 线程/进程池大小随组合规模与 CPU 数自适应。
  - `< 128 任务：单线程执行`。
  - `≥ 512 任务：多线程/多进程，worker 数 = min(4, cpu_count)`。
  - `PREDICTION_USE_PROCESS_POOL=1` 时启用进程池；`PREDICTION_MAX_WORKERS` 限制并发数。
  - 子任务超时：120 秒；结果做稳定排序（概率降序→学校→专业），避免并发顺序抖动导致的 UI 差异。
- 输入语言分数：不再从案例中位数回填默认值，按用户输入为准；仍进行归一化处理。

### 错误与返回约定
- 关键资源缺失（模型/特征/数据）：页面层会展示中文错误，API 返回空结构或 `None`。
- 并发子任务失败：记录日志但不阻断整体；能返回的结果尽可能返回。

### 最小使用示例（脚本/服务）
```python
from src.pages.prediction.prediction_pipeline import run_prediction_pipeline
from src.pages.prediction.prediction_data_preparer import prepare_input_data
from src.pages.prediction.prediction_validator import validate_model_and_features
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_fingerprint import (
    compute_df_fingerprint,
    compute_list_fingerprint,
)
from src.utils.app_data_loader import load_raw_cases_data

model = cached_get_prediction_model("xgboost")
features = validate_model_and_features(model)
if features is None:
    raise ValueError("模型或特征列表加载失败")

input_data = {
    "background_university": "香港大学",
    "background_major": "计算机科学",
    "target_universities": ["香港大学", "香港科技大学"],
    "target_majors": ["计算机科学", "数据科学"],
    "gpa": 3.6,
    "language_score": 105,
    "internship_count": 2,
    "research_count": 1,
    "award_count": 0,
    "paper_count": 0,
    "experience_details": {"research_details": "NLP 实验室 RA，1 篇一作在投"}
}
input_prepared = prepare_input_data(input_data)

cases_df = load_raw_cases_data()
all_universities_target = ["香港大学", "香港科技大学", "香港中文大学"]
all_majors_target = ["计算机科学", "数据科学", "人工智能"]

input_prepared["_all_universities_target"] = all_universities_target
input_prepared["_all_majors_target"] = all_majors_target

cases_df_fingerprint = compute_df_fingerprint(cases_df)
all_universities_fingerprint = compute_list_fingerprint(all_universities_target)
all_majors_fingerprint = compute_list_fingerprint(all_majors_target)

result_model = run_prediction_pipeline(
    input_data=input_prepared,
    model_name="xgboost",
    cases_df_fingerprint=cases_df_fingerprint,
    loaded_feature_names=features,
    all_universities_fingerprint=all_universities_fingerprint,
    all_majors_fingerprint=all_majors_fingerprint,
)

print(f"相似专业结果数: {len(result_model.similarity_results) if result_model.similarity_results else 0}")
print(f"跨专业结果数: {len(result_model.cross_major_results) if result_model.cross_major_results else 0}")
print(f"用户指定结果数: {len(result_model.user_specified_results) if result_model.user_specified_results else 0}")
print(f"统一结果数: {len(result_model.unified_results) if result_model.unified_results else 0}")
```

### 最小 UI 展示片段（Streamlit）
```python
import streamlit as st
from src.pages.prediction.prediction_pipeline import run_prediction_pipeline
from src.pages.prediction.prediction_data_preparer import prepare_input_data
from src.pages.prediction.prediction_validator import validate_model_and_features
from src.pages.prediction.page_data_loader import cached_get_prediction_model
from src.pages.prediction.prediction_fingerprint import (
    compute_df_fingerprint,
    compute_list_fingerprint,
)
from src.pages.prediction.result_display import ResultsDisplay

model = cached_get_prediction_model("xgboost")
features = validate_model_and_features(model)
if features is None:
    st.error("模型加载失败")
    st.stop()

input_data = prepare_input_data(user_form_data)
input_data["_all_universities_target"] = all_universities_target
input_data["_all_majors_target"] = all_majors_target

cases_df_fingerprint = compute_df_fingerprint(cases_df)
all_universities_fingerprint = compute_list_fingerprint(all_universities_target)
all_majors_fingerprint = compute_list_fingerprint(all_majors_target)

res = run_prediction_pipeline(
    input_data=input_data,
    model_name="xgboost",
    cases_df_fingerprint=cases_df_fingerprint,
    loaded_feature_names=features,
    all_universities_fingerprint=all_universities_fingerprint,
    all_majors_fingerprint=all_majors_fingerprint,
)

st.metric("候选总数", len(res.unified_results) if res.unified_results else 0)

results_display = ResultsDisplay(
    top_similarity_results=res.similarity_results,
    top_cross_major_results=res.cross_major_results,
    user_specified_results=res.user_specified_results,
)
results_display.display(
    target_universities=input_data.get("target_universities", []),
    target_majors=input_data.get("target_majors", []),
    background_university=input_data.get("background_university"),
    background_major=input_data.get("background_major"),
)
```
---
维护人：lijiapeng8@xdf.cn
版本：v2.6
