## 结果修正模块文档（src/pages/prediction/result_modifier）

本模块在主模型（如 XGBoost）输出基础概率后，进行保守、可解释的后处理校正：学科相似度、专业类目规则、跨专业惩罚、基于 Logit 的文本加成等。

### 结构概览
- 配置与工具：`config.py`（集中配置）、`utils.py`（公共工具函数）
- 文本加成入口：`text_boost_provider.py`
- 文本加成：`providers/logit_uplift_provider.py`（模型由脚本 `scripts/train_text_tfidf.py` 训练导出）
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
  -（已移除）关键词缓存相关配置。

- 文本加成默认配置（节选，与实现保持一致）
```python
DEFAULT_TEXT_BOOST_CONFIG = {
    "enabled": True,
    # 最大总提升上限（在 p≈0.5 处达到上限；两端随 p 衰减）
    "max_total_boost": 0.15,
    # 相似度门槛（任一不达标则不生效）
    "sim_gate_sum_min": 0.10,
    "sim_gate_max_min": 0.08,
    # 平滑系数：将 logit 增量压缩，避免过激
    "smoothing": 0.7,
    # 封顶乘子最小值（低质量文本的上限占比），范围 (0,1]
    "cap_min_factor": 0.10,
    # 质量对封顶的非线性：>1 强化强文本；=1 线性；<1 平滑
    "cap_quality_gamma": 1.2,
    "model_paths": {
        "tfidf_vectorizer": "src/machine_learning_models/pre-trained_models/tfidf_vectorizer.joblib",
        "tfidf_centroids": "src/machine_learning_models/pre-trained_models/tfidf_centroids.npz",
        "text_uplift_weights": "src/machine_learning_models/pre-trained_models/text_uplift_weights.json",
    },
}
```

- 使用位置速览
  - 概率调整：`probability_adjuster.py` 使用 `GPA_MINIMUM`、`LANGUAGE_MINIMUM`、`CROSS_MAJOR_PENALTY_FACTOR`。
  - 职业型专业：`professional_adjustment.py` 使用 `PROFESSIONAL_*` 参数。
  - 相似度阈值：`config.py` 集中定义，`ranker.py` 等处引用。
  - 公共工具：`utils.py` 提供 `is_effectively_empty`、`clip_probability`、`generate_content_hash`、`has_valid_experience_details`、`has_meaningful_experience_text` 等函数，被多个模块复用。
  - 文本加成：`text_boost_provider.py` 使用 `DEFAULT_TEXT_BOOST_CONFIG`。

### 文本加成（Text Boost）
- 入口函数：`get_text_boost_provider(config) -> TextBoostProvider`
  - `config.enabled`：关闭/开启
  - `config.max_total_boost`：动态封顶最大值（默认 0.15，可调）；基础形状随概率远离中段而衰减
  - `config.sim_gate_sum_min` / `config.sim_gate_max_min`：文本相似度的生效门槛
  - `config.smoothing`：对 logit 增量的平滑系数（0~1），避免过激
  - `config.cap_min_factor` / `config.cap_quality_gamma`：基于文本“质量分数”的封顶调节
  - `config.model_paths`：`tfidf_vectorizer`、`tfidf_centroids`、`text_uplift_weights`

#### Provider（统一版，仅 TF‑IDF Logit Uplift）
- LogitUpliftProvider：
  - 特征：四段文本相似度 `s_r,s_a,s_i,s_p` 与交互项 `s_k*log1p(count_k)`（`count_k` 为 `research/award/internship/paper` 四个计数）。
  - 公式：`p' = sigmoid(logit(p) + max(0, b + Σ w_k*s_k + Σ u_k*(s_k*log1p(count_k))))`。
  - 生效保护：`sum(s_k) ≥ sim_gate_sum_min` 且 `max(s_k) ≥ sim_gate_max_min`；`delta_logit` 乘以 `smoothing` 做温和化。
  - 动态上限与中段限定：仅对中段概率（0.2–0.8）生效；封顶：
    - `scale = 1 − 2×|p − 0.5|`
    - `quality_raw = 0.7×max(s_k) + 0.3×mean(s_k)`
    - `cap_factor = clamp(cap_min_factor, 1.0, quality_raw^cap_quality_gamma)`
    - `cap_boost = max_total_boost × cap_factor × scale`
    - `cap = p × (1 + cap_boost)`，最终裁剪到 [0,1]
  - 缓存：基于文本与 count 的签名 LRU（≈512），默认 100ms 内完成；异常回退原概率。

#### TF‑IDF 训练与权重拟合（scripts/train_text_tfidf.py）
- 配置：`analyzer='char_wb'`、`ngram_range=(2,4)`、`min_df=1`、`max_features=20000`、`sublinear_tf=True`、`norm='l2'`（与推理对齐）。
- 产物：
  - `tfidf_vectorizer.joblib`、`tfidf_centroids.npz`（相似度）
  - `text_uplift_weights.json`（非负系数：`b,w_r,w_a,w_i,w_p,u_r,u_a,u_i,u_p`）

### 概率调整（probability_adjuster.py）
- `ProbabilityAdjuster(cases_df)`：根据案例数据估计 `gpa/language` 的均值/方差与“通过线”，对低于均值的样本施加保守惩罚；极低档次做下限截断。
- `penalize_cross_major_without_cases(user_specified_results, background_major, cases_df)`：针对用户指明的跨专业组合，如果历史中无录取案例，则对概率乘以 `PENALTY_FACTOR`（默认 0.5）。
  - 截断细节：当 GPA 与语言均低于最低线（`GPA_MINIMUM=2.0`、`LANGUAGE_MINIMUM=0.6`）时直接返回 `0.001`；若惩罚后仍极低且显著低于均线/通过线，也会截断为 `0.001`。

### 相似度与推荐（similarity_adjuster.py, ranker.py）
- `adjust_similarity_score(background_major, target_major, similarity)`：根据配置 `config/similarity_adjustment_rules.json` 的关键字规则对相似度小幅修正（0–1 截断）。
- `get_similar_major_recommendations`：先按相似度阈值过滤（申请院校数较少≤`UNIVERSITY_COUNT_THRESHOLD=5`时使用更高阈值 `0.92`，否则 `0.89`）；再取前 `TOP_N=50`（按相似度筛）；随后对概率做裁剪；最终按概率降序返回。
- `get_cross_major_recommendations`：在历史中存在录取案例的跨专业中，筛选 `0.8 < similarity < MIN_SIMILARITY_THRESHOLD` 的候选，并按概率降序返回。

### 行业专业调整（professional_adjustment.py）
- 对职业类项目（如 MBA/BA）在缺少实习经历时降低概率；若用户显式指定该类专业，降幅较轻。

### 录取案例缓存（admission_cache.py）
- 以（目标院校、目标专业、背景专业）三元组过滤历史录取组合，并缓存结果用于跨专业相关的修正。实现使用 `@st.cache_data`；为支持缓存，将 DataFrame 投影为元组后传入缓存函数。

### 训练产物（文本）与部署
- 训练脚本：`scripts/train_text_tfidf.py`
  - 输入：`src/machine_learning_models/data/cases.feather`
  - 输出：`src/machine_learning_models/pre-trained_models/{tfidf_vectorizer.joblib, tfidf_centroids.npz, text_uplift_weights.json}`
- 部署：在应用配置 `text_boost` 中配置上述模型路径与 `max_total_boost`；异常自动回退为原概率。

### 设计边界与原则
- 后处理与主模型解耦，不改变样本排序的主导因素，仅做小幅、可解释的加减分。
- `max_total_boost` 控制总增益；TF‑IDF 仅对中段概率生效，避免极端情况被放大。
- 任何代价较高的计算（如 TF‑IDF）都带缓存；无显式超时。

### 评估快照（5000 样本）
- 平均加权耗时 ≈ 0.16–0.22ms/样本；AUC 变化 ≈ -0.00033；Top@K 轻微波动（Top@50 -2、Top@100 +1、Top@200 +1）。

### 常见用法与注意事项
- 只做“小幅、可解释”的修正：避免替代主模型排序；`max_total_boost` 建议 0.02–0.15（默认 0.15，可按业务调低/调高）。
- 文本加成只在中段概率生效（0.1–0.9），并带强信号门控，避免弱文本误放大。
- 跨专业惩罚 `penalize_cross_major_without_cases`：仅在用户显式“指定跨专业组合”且历史无成功案例时生效；自动推荐结果不受该惩罚影响。
- 关键词表维护：优先高辨识度名词，定期审查，避免误触。
 - 平均加权耗时 ≈ 0.16–0.22ms/样本；AUC 变化极小；Top@K 轻微波动。

---
维护人：lijiapeng8@xdf.cn
版本：v2.7


