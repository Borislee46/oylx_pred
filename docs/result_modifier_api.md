## 结果修正模块文档（src/pages/prediction/result_modifier）

本模块在主模型（如 XGBoost）输出基础概率后，进行保守、可解释的后处理校正：学科相似度、专业类目规则、跨专业惩罚、关键词/TF‑IDF 文本加成等。

### 结构概览
- 配置与工具：`config.py`（集中配置）、`utils.py`（公共工具函数）、`keyword_config.py`（关键词权重配置）
- 文本加成入口：`text_boost_provider.py`
- 关键词加成：`keyword_booster.py`、`providers/keyword_provider.py`
- TF‑IDF 加成：`providers/tfidf_provider.py`（模型由脚本 `scripts/train_text_tfidf.py` 训练导出）
- 概率调整：`probability_adjuster.py`
- 相似度调整：`similarity_adjuster.py`
- 排序与推荐：`ranker.py`
- 行业专业调整：`professional_adjustment.py`
- 录取案例缓存：`admission_cache.py`

### 配置参数说明（config.py）
- 所有关键参数集中于 `src/pages/prediction/result_modifier/config.py`，修改后将影响对应模块的行为。

- 基础阈值与系数
  - `GPA_MINIMUM = 2.0`：用于 `ProbabilityAdjuster` 的 GPA 低阈，低于该值触发强惩罚。
  - `LANGUAGE_MINIMUM = 0.6`：用于 `ProbabilityAdjuster` 的语言低阈，低于该值触发强惩罚。
  - `CROSS_MAJOR_PENALTY_FACTOR = 0.5`：`penalize_cross_major_without_cases` 在“跨专业且历史无录取”时的乘法折减系数。

- 职业型专业规则
  - `PROFESSIONAL_MAJORS = ["Business Administration", "MBA"]`：职业型专业关键词。
  - `PROFESSIONAL_REDUCTION_FACTOR = 0.70`：无实习时对职业型专业的默认降权因子。
  - `PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR = 0.85`：若用户显式指定该职业型专业，降权较轻。

- 相似度阈值（用于筛选/推荐策略）
  - `MIN_SIMILARITY_THRESHOLD = 0.89`
  - `HIGHER_SIMILARITY_THRESHOLD = 0.92`
  - `UNIVERSITY_COUNT_THRESHOLD = 5`

- 推荐与缓存配置
  - `TOP_N_RECOMMENDATIONS = 50`：`ranker.py` 中返回的最大推荐数量。
  - `PROBABILITY_ADJUSTER_CACHE_SIZE = 50`：`ProbabilityAdjuster` 统计缓存容量。
  - `KEYWORD_BOOSTER_CACHE_SIZE = 1000`：`KeywordBooster` 关键词加成缓存容量。

- 文本加成默认配置（节选）
```python
DEFAULT_TEXT_BOOST_CONFIG = {
    "enabled": True,
    "max_total_boost": 0.05,
    "timeout_ms": 100,
    "similarity_thresholds": [[0.15, 0.05]],
    "model_paths": {
        "tfidf_vectorizer": "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib",
        "tfidf_centroids": "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz",
    },
}
```

- 使用位置速览
  - 概率调整：`probability_adjuster.py` 使用 `GPA_MINIMUM`、`LANGUAGE_MINIMUM`、`CROSS_MAJOR_PENALTY_FACTOR`。
  - 职业型专业：`professional_adjustment.py` 使用 `PROFESSIONAL_*` 参数。
  - 相似度阈值：`config.py` 集中定义，`ranker.py` 等处引用。
  - 关键词权重：`keyword_config.py` 定义 `KEYWORD_WEIGHTS` 和 `BOOST_MULTIPLIERS`，由 `keyword_booster.py` 使用。
  - 公共工具：`utils.py` 提供 `is_effectively_empty`、`clip_probability`、`generate_content_hash` 等函数，被多个模块复用。
  - 文本加成：`text_boost_provider.py` 使用 `DEFAULT_TEXT_BOOST_CONFIG`。

### 文本加成（Text Boost）
- 入口函数：`get_text_boost_provider(config) -> TextBoostProvider`
  - `config.enabled`：关闭/开启
  - `config.provider`：`keyword` | `tfidf` | `max_of_keyword_and_tfidf`（当前实现固定为 `max_of_keyword_and_tfidf`，该开关暂未启用）
  - `config.max_total_boost`：总上限（默认 0.05；建议不超过 0.05）
  - `config.timeout_ms`：TF‑IDF 计算超时（默认 100ms）
  - `config.similarity_thresholds`：TF‑IDF 相似度→加成可配置阈值（例如 `[[0.15,0.05]]` 或 `[[0.15,0.05],[0.10,0.03],[0.05,0.02]]`）
  - `config.model_paths.tfidf_vectorizer/tfidf_centroids`（仅 tfidf；vectorizer 默认后缀为 `.joblib`）
  - 灰度：`_should_rollout(config, experience_details)`（预留，当前未实现）

#### Provider 类型
- keyword：按经历文本命中不同级别关键词（`top/high/medium/general`）小幅累加；单段经历上限 10%（`MAX_SINGLE_EXPERIENCE_BOOST`），总体仍受 `max_total_boost` 约束。
- tfidf：对四段经历与各自“语料质心”算近似余弦相似度，按阈值映射为小额加成；仅对中段概率（0.2–0.8）生效；带强信号门控（需命中 `top_tier` 关键词才启用）。
- max_of_keyword_and_tfidf：并行计算两种方式，对每个概率位置取两者较大值，摘要选择提升更高的一侧；受同一 `max_total_boost` 限制。
  - 实现细节：组合器外层增加 `GatedTextBoostProvider` 与缓存包装，保证弱信号与超时场景下安全回退。
  - 说明：当前实现固定走 `max_of_keyword_and_tfidf` 路径，`config.provider` 暂未启用。

#### 关键词表（扩充）
- 已补充顶会/顶刊/机构与高价值竞赛（如 `NeurIPS/ICML/ICLR/CVPR/ACL/KDD/SIGMOD/AAAI/IJCAI/Nature Communications/PNAS/TPAMI`；`IMO/IOI/ICPC/丘成桐`；`Morgan Stanley/Goldman Sachs/J.P. Morgan` 等）。
- 原则：优先高辨识度名词，控制规模，避免模板词/停用片段。

#### TF‑IDF 训练（scripts/train_text_tfidf.py）
- 配置：`analyzer='char_wb'`、`ngram_range=(2,4)`、`min_df=1`、`max_features=20000`、`sublinear_tf=True`、`norm='l2'`，并统一 `scikit-learn==1.4.2`（训练与推理对齐）。
- 产物：`src/machine_learning_models/pre-trained_models/{tfidf_vectorizer.joblib, tfidf_centroids.npz}`。

### 概率调整（probability_adjuster.py）
- `ProbabilityAdjuster(cases_df)`：根据案例数据估计 `gpa/language` 的均值/方差与“通过线”，对低于均值的样本施加保守惩罚；极低档次做下限截断。
- `penalize_cross_major_without_cases(user_specified_results, background_major, cases_df)`：针对用户指明的跨专业组合，如果历史中无录取案例，则对概率乘以 `PENALTY_FACTOR`（默认 0.5）。
  - 截断细节：当 GPA 与语言均低于最低线（`GPA_MINIMUM=2.0`、`LANGUAGE_MINIMUM=0.6`）时直接返回 `0.001`；若惩罚后仍极低且显著低于均线/通过线，也会截断为 `0.001`。

### 相似度与推荐（similarity_adjuster.py, ranker.py）
- `adjust_similarity_score(background_major, target_major, similarity)`：根据配置 `config/similarity_adjustment_rules.json` 的关键字规则对相似度小幅修正（0–1 截断）。
- `_get_similar_major_recommendations`：先按相似度阈值过滤（申请院校数较少≤`UNIVERSITY_COUNT_THRESHOLD=5`时使用更高阈值 `0.92`，否则 `0.89`）；再取前 `TOP_N=50`（按相似度筛）；随后对概率做裁剪；最终按概率降序返回。
- `_get_cross_major_recommendations`：在历史中存在录取案例的跨专业中，筛选 `0.8 < similarity < MIN_SIMILARITY_THRESHOLD` 的候选，并按概率降序返回。

### 行业专业调整（professional_adjustment.py）
- 对职业类项目（如 MBA/BA）在缺少实习经历时降低概率；若用户显式指定该类专业，降幅较轻。

### 录取案例缓存（admission_cache.py）
- 以（目标院校、目标专业、背景专业）三元组过滤历史录取组合，并缓存结果用于跨专业相关的修正。实现使用 `@st.cache_data`；为支持缓存，将 DataFrame 投影为元组后传入缓存函数。

### 训练产物（文本）与部署
- 训练脚本：`scripts/train_text_tfidf.py`
  - 输入：`src/machine_learning_models/data/cases.feather`
  - 输出：`src/machine_learning_models/pre-trained_models/{tfidf_vectorizer.joblib, tfidf_centroids.npz}`
- 部署：在应用配置 `text_boost` 中标注模型路径与阈值；当前固定为关键词与 TF‑IDF 合并（max‑of‑two），`config.provider` 预留未启用；灰度 `rollout_ratio` 预留未实现。

### 设计边界与原则
- 后处理与主模型解耦，不改变样本排序的主导因素，仅做小幅、可解释的加减分。
- `max_total_boost` 控制总增益；TF‑IDF 仅对中段概率生效，避免极端情况被放大。
- 任何代价较高的计算（如 TF‑IDF）都带缓存与超时（默认 100ms）。

### 评估快照（5000 样本）
- 平均加权耗时 ≈ 0.16–0.22ms/样本；AUC 变化 ≈ -0.00033；Top@K 轻微波动（Top@50 -2、Top@100 +1、Top@200 +1）。

### 常见用法与注意事项
- 只做“小幅、可解释”的修正：避免替代主模型排序；`max_total_boost` 建议不超过 0.05。
- 文本加成只在中段概率生效（0.2–0.8），并带强信号门控，避免弱文本误放大。
- 跨专业惩罚 `penalize_cross_major_without_cases`：仅在用户显式“指定跨专业组合”且历史无成功案例时生效；自动推荐结果不受该惩罚影响。
- 关键词表维护：优先高辨识度名词，定期审查，避免误触。
- 平均加权耗时 ≈ 0.16–0.22ms/样本；AUC 变化 ≈ -0.00033；Top@K 轻微波动（Top@50 -2、Top@100 +1、Top@200 +1）。
 - `similarity_thresholds` 未配置时，TF‑IDF provider 回退为 `[(0.40,0.05),(0.30,0.03),(0.20,0.02)]`；当前实现固定走 `max_of_keyword_and_tfidf`（`config.provider` 暂未启用）。

---
维护人：lijiapeng8@xdf.cn
版本：v2.4


