## 预测模块 API 文档（src/pages/prediction）

### 模块结构
- 类型定义：`src/pages/prediction/prediction_types.py`
- 资源加载：`src/pages/prediction/page_data_loader.py`
- 模型封装：`src/pages/prediction/prediction_model.py`
- 组合与结果处理：`src/pages/prediction/prediction_processor.py`
- 并行推理器：`src/pages/prediction/run_prediction.py`
- 统一编排入口：`src/pages/prediction/prediction_handler.py`
- 结果模型定义：`src/utils/session_manager.py`（`PredictionResultModel`）

### 数据类型（src/pages/prediction/prediction_types.py）
- `PredictionInput`（TypedDict，可选字段）
  - `background_university: str`
  - `background_major: str`
  - `target_universities: List[str]`
  - `target_majors: List[str]`
  - `gpa: float`
  - `language_score: float`
  - `internship_count, research_count, award_count, paper_count: int`
  - `school_level: str`  # 例如："985" | "211" | "1-50" | "51-100" | "普通本科"
  - `experience_details: Dict[str, str]`
- `PredictionResultItem`（TypedDict，可选字段）
  - `university: str`, `major: str`, `probability: float`, `similarity: float`
  - 附加字段：`chinese_name: str`（专业中文名称，处理阶段注入）
- `MetaInfo = Dict[str, Any]`（例如：组合数量与提示消息）
  - 注：当前类型定义文件中 `school_level` 标注为 `int`，但运行时由 `prepare_input_data` 注入为 `str`；后续将统一为 `str`。

### 资源加载（src/pages/prediction/page_data_loader.py）
- `cached_get_prediction_model(model_name: str) -> PredictionModel`
  - 说明：加载并缓存模型与全局类别映射（来自 `utils.app_data_loader`）。
- `cached_load_cases_data() -> pd.DataFrame`
  - 说明：优先加载 `cases_min.feather`，失败回退到 `cases.feather`；自动补齐缺失必需列并做基本类型清洗（如 `admitted`/`gpa`），由 `utils.app_data_loader.load_raw_cases_data` 完成。
- `cached_load_bg_target_similarity_cache() -> dict`
  - 说明：加载并缓存“原专业-目标专业”相似度缓存（默认从根目录 `cache/background_target_similarity.feather` 读取）。

### 模型封装（src/pages/prediction/prediction_model.py）
- `class PredictionModel(model_type: str, global_categories_df: Optional[pd.DataFrame])`
  - 内部：`load_model(model_type)`、`feature_names`、`global_categories_`（分类特征全局类别）。
- `predict_batch(input_data: Dict[str, Any], combinations: List[Tuple[str, str]], expected_features: List[str]) -> List[PredictionResultItem]`
  - 批量推理；自动分批；输出每个 `(university, major)` 的概率。
- `preprocess_input(input_data: pd.DataFrame | dict, expected_features_list: list) -> pd.DataFrame`
  - 对数化计数列；分类列按全局类别编码；补齐缺失列并对齐列序。
- `predict_probability(input_df: pd.DataFrame) -> float | None`
  - 单样本概率预测。

### 组合与结果处理（src/pages/prediction/prediction_processor.py）
- `generate_prediction_combinations(input_data: PredictionInput, all_universities_target: List[str], all_majors_target: List[str]) -> Tuple[List[Tuple[str, str]], MetaInfo]`
  - 纯函数：生成有效 `(university, major)` 组合（过滤无详情项），并返回 `meta`：
    - `meta['combination_count']: int`
    - `meta['combination_message']: str`
- `process_prediction_results(results: list, background_major: str, bg_target_similarity_cache: dict, num_target_universities: int, cases_df: pd.DataFrame | None = None, user_specified_combinations: list[tuple[str, str]] | None = None) -> Tuple[list, list, list]`
  - 流程与输出：
    - 先拦截 `part-time` 专业（忽略名称同时包含 "part" 与 "time" 的项目）。
    - 批量计算并注入 `similarity`（使用缓存与规则修正器）。
    - 基于学校-专业详情维表批量映射中文名，新增字段 `chinese_name`。
    - 产出三类结果：相近专业、跨专业、用户指定组合。
    - 返回的每个结果项至少包含：`university`、`major`、`probability`、`similarity`、`chinese_name`。

### 并行推理器（src/pages/prediction/run_prediction.py）
- `run_single_prediction(current_input_data: PredictionInput, prediction_model: PredictionModel, cases_df: pd.DataFrame, bg_target_similarity_cache: dict, expected_features: List[str], all_universities_target: List[str], all_majors_target: List[str], num_target_universities: int) -> Tuple[list, list, list, None]`
  - 步骤：
    1) 调用 `generate_prediction_combinations` → `(combinations, meta)`；将 `meta` 写入会话（组合数量与提示）。
    2) 基于线程/进程池自适应并发批量 `predict_batch`（可通过环境变量 `PREDICTION_USE_PROCESS_POOL=1` 启用进程池），并对结果做稳定排序（概率降序→学校→专业）。
    3) 处理为三类结果并返回（最后一个返回值保留为 `None`）。
  - 鲁棒性：子任务设超时；异常不阻断整体，但记录日志。

### 编排入口（src/pages/prediction/prediction_handler.py）
- `validate_model_and_features(prediction_model: PredictionModel) -> List[str] | None`
  - 校验模型与特征名；页面层会给出中文错误提示。
- `prepare_input_data(input_data_from_form: dict) -> dict`
  - 注入 `school_level` 等派生字段（依赖 `utils.school_level_service`）。
- `run_prediction_pipeline(
    input_data: PredictionInput,
    model_name: str,
    cases_df_fingerprint: int,
    loaded_feature_names: List[str],
    all_universities_fingerprint: Tuple[int, int],
    all_majors_fingerprint: Tuple[int, int],
) -> PredictionResultModel`
  - 顶层管线（`@st.cache_data(ttl=600)`）：
    - 内部按 `model_name` 读取并缓存模型；`cases_df` 与候选全集通过指纹参数复用缓存。
    - 调用 `run_single_prediction` 获取三类结果；
    - 行业专业规则修正：`adjust_for_professional_majors`；
    - 跨专业惩罚：`penalize_cross_major_without_cases`；
    - 概率调整：当提供 `gpa` 与 `language_score` 时，使用 `ProbabilityAdjuster` 做概率校正；
    - 文本加成：`get_text_boost_provider(...).apply`；
    - 新专业标记：批量写入 `is_new_major` 标记以供前端展示；
    - 合并去重：`src/pages/prediction/results_handler.py` 中的 `combine_and_deduplicate_results` → `unified_results`；
    - 返回 `PredictionResultModel`。

### 结果展示（src/pages/prediction/prediction_result_display.py）
- `class ResultsDisplay`
  - 展示三类结果为可交互表格，默认列：`目标院校`、`目标专业`、`录取概率`（进度条）、`专业中文名称`。
  - 当结果项带有 `is_new_major=True` 时，在 `目标专业` 后追加 "(New!)" 并在渲染时高亮。
  - 排序规则：优先按固定院校顺序，其次同校内按概率降序。
  - 大组合池（>100）时优先展示 Top N 的相似/跨专业结果；用户指定组合在组合池较小时展示。

### 文本加成（TF‑IDF Logit Uplift）
- 提供方：`result_modifier/text_boost_provider.py`（外层门控与缓存）与 `providers/logit_uplift_provider.py`（核心增益计算）。
- 生效条件（门控）：当四段文本相似度向量 `s` 满足 `sum(s) ≥ sim_gate_sum_min` 且 `max(s) ≥ sim_gate_max_min` 时生效。
- 增益方式：对 `logit(p)` 加上经平滑后的非负增量，仅对中段概率区间（0.2–0.8）生效，并受动态封顶约束。
- 封顶策略：`cap_boost = max_total_boost × cap_factor(质量) × (1 − 2×|p − 0.5|)`；其中质量分 `quality = 0.7×max(s)+0.3×mean(s)`，`cap_factor = clamp(cap_min_factor, 1.0, quality^cap_quality_gamma)`。
- 关键配置（`src/pages/prediction/result_modifier/config.py`）：
```json
{
  "enabled": true,
  "max_total_boost": 0.10,
  "sim_gate_sum_min": 0.25,
  "sim_gate_max_min": 0.18,
  "smoothing": 0.5,
  "cap_min_factor": 0.05,
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
  - `similarity_results: Optional[pd.DataFrame]`
  - `cross_major_results: Optional[pd.DataFrame]`
  - `user_specified_results: Optional[pd.DataFrame]`
  - `unified_results: Optional[pd.DataFrame]`

### 缓存、性能与一致性
- 资源/数据加载：`cached_*` 系列使用 Streamlit 缓存。
- 编排函数：`run_prediction_pipeline` 使用 `@st.cache_data(ttl=600)`。
- 并行推理：线程/进程池大小随组合规模与 CPU 数自适应；`PREDICTION_USE_PROCESS_POOL=1` 时启用进程池；结果做稳定排序，避免并发顺序抖动导致的 UI 差异。
 - 输入语言分数：不再从案例中位数回填默认值，按用户输入为准；仍进行归一化处理。

### 错误与返回约定
- 关键资源缺失（模型/特征/数据）：页面层会展示中文错误，API 返回空结构或 `None`。
- 并发子任务失败：记录日志但不阻断整体；能返回的结果尽可能返回。

### 最小使用示例（脚本/服务）
```python
from src.pages.prediction.prediction_handler import (
    validate_model_and_features, prepare_input_data, run_prediction_pipeline,
)
from src.pages.prediction.page_data_loader import cached_get_prediction_model

model = cached_get_prediction_model("xgboost")
features = validate_model_and_features(model)

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

# 由调用方提供稳定指纹（例如 len(data), hash(cols) 等）以命中缓存
cases_df_fingerprint = 0
all_universities_fingerprint = (0, 0)
all_majors_fingerprint = (0, 0)

result_model = run_prediction_pipeline(
    input_data=input_prepared,
    model_name="xgboost",
    cases_df_fingerprint=cases_df_fingerprint,
    loaded_feature_names=features,
    all_universities_fingerprint=all_universities_fingerprint,
    all_majors_fingerprint=all_majors_fingerprint,
)

print(len(result_model.unified_results))
```

### 最小 UI 展示片段（Streamlit）
```python
import streamlit as st
from src.pages.prediction.prediction_handler import (
    validate_model_and_features, prepare_input_data, run_prediction_pipeline,
)
from src.pages.prediction.page_data_loader import cached_get_prediction_model

model = cached_get_prediction_model("xgboost")
features = validate_model_and_features(model)
input_data = prepare_input_data(user_form_data)
res = run_prediction_pipeline(
    input_data=input_data,
    model_name="xgboost",
    cases_df_fingerprint=len(all_unis) + len(all_majors),
    loaded_feature_names=features,
    all_universities_fingerprint=(len(all_unis), 0),
    all_majors_fingerprint=(len(all_majors), 0),
)

st.metric("候选总数", len(res.unified_results) if res.unified_results is not None else 0)
st.dataframe(res.unified_results.head(20) if res.unified_results is not None else None, use_container_width=True)
```
---
维护人：lijiapeng8@xdf.cn
版本：v2.4
