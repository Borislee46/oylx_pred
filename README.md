## EasyApply 留学择校系统

本数据产品提供一套从"模型训练 → 表单校验与归一 → 预测编排 → 结果调整"的端到端解决方案。
核心亮点为：
1）极致的I/O和开销，几乎0成本+秒推理。
2）和业务专家对齐效果，已非常逼近人工选校。
部分算法的边界case可用agents解或全部换成agent如有足够的资源

### 架构流程
1) **模型训练 (Offline)**: `src/machine_learning_models/`
   - **核心**: 使用 XGBoost 结合 SMOTE/SMOTENC 等采样方法处理不平衡数据，并进行概率 sigmoid 校准。
   - **产物**: 生成 `XGBoost` 主模型（`.ubj` 格式，包含特征名、校准参数、学校层级回退映射等属性）。
   - **文本加成训练**: 使用 TF‑IDF 训练文本相似度模型，生成向量器、质心和权重文件（`scripts/train_text_tfidf.py`）。

2) **专业相似度预计算 (Offline)**: `scripts/precompute_similarities.py`
   - **核心**: 使用 E5 Embedding 模型预计算背景专业-目标专业相似度缓存，加速线上查询。
   - **产物**: `cache/background_target_similarity.feather`（背景专业-目标专业相似度字典）。

3) **用户表单 (Online)**: `src/pages/prediction/input_form_components/`
   - **核心**: `FormStateManager` 管理复杂的UI状态，支持**自动保存**和**历史恢复**。`FormValidator` 和 `GPAConverter` 对用户输入进行严格的校验和智能转换。
   - **特色功能**: 跨学院申请提示拦截、目标选择自动扩展、海外院校语言成绩选填处理。

4) **预测编排 (Online)**: `src/pages/prediction/`
   - **核心**: `prediction_handler` 负责编排整个流水线；`run_prediction` 采用线程/进程池自适应并发（可通过环境变量启用进程池）并进行稳定排序与超时兜底。
   - **三类结果**: 相似专业推荐、跨专业推荐、用户指定组合结果。

5) **结果调整 (Online)**: `src/pages/prediction/result_modifier/`
   - **核心**: 对模型原始概率进行一系列业务专家规则微调，包括基于历史案例的 `ProbabilityAdjuster`、针对无实习申商科的 `professional_adjustment`，以及基于 TF‑IDF Logit Uplift 的**文本加成**（offline 训练，online 推理）。

---

## 一、训练（src/machine_learning_models）
- **数据与特征**：
  - 目标列：`admitted`（二分类）
  - 分类列：`background_university/background_major/target_university/target_major` → 转为 pandas `category`
  - 计数列：`research_count/award_count/internship_count/paper_count`（99%分位截尾 + log1p）
  - 语言列：统一为 `language_score`（toefl/ielts 归一后取 max）
  - 文本列：仅用于样本权重判定（所有 detail 文本列均为空时降权），不直接入模
- **采样**：SMOTE/SMOTENC（自动识别分类特征索引），`k_neighbors` 基于少数类样本量动态设置；异常时回退到原始数据
- **样本权重**：
  - 文本为空样本：权重设为 `TEXT_EMPTY_SAMPLE_WEIGHT`（默认 0.85）
  - 最近样本加权：最后 `RECENT_SAMPLE_BOOST_COUNT` 条样本乘以 `RECENT_SAMPLE_BOOST_WEIGHT`（默认 10000 条、1.1）
  - 采样后权重对齐：按采样索引对齐原权重，新增样本填充类别均值
- **训练与校准**：
  - XGBoost + 单调约束（GPA 等特征通过白名单控制）
  - `CalibratedClassifierCV(sigmoid, cv='prefit')`：训练集 80% 拟合基础模型，剩余 20% 作为校准集
  - 类别不平衡：自动计算 `scale_pos_weight = #neg/#pos`
- **入口命令**（参数，数据集和详细训练流程暂不公开，有需要请联系lijiapeng8@xdf.cn）：
```bash
python -m src.machine_learning_models.train --model xgboost --sampling_method smote
```
- **产物**：
  - 模型：`src/machine_learning_models/pre-trained_models/{model}_{timestamp}.ubj`（Booster 内写入 `feature_names/calibration_params/level_fallback_mapping` 等属性；属性缺失时回退读取同名 JSON）
  - 评估：`src/machine_learning_models/evaluation_results/{model}_evaluation_{timestamp}.json`
- **文本加成训练**（`scripts/train_text_tfidf.py`）：
  - 输入：`src/machine_learning_models/data/cases.feather`
  - 输出：`tfidf_vectorizer.joblib`、`tfidf_centroids.npz`、`text_uplift_weights.json`
  - 向量器参数：`analyzer='char_wb'`, `ngram_range=(2,4)`, `min_df=1`, `max_features=20000`（与线上一致）
  - 运行：`python scripts/train_text_tfidf.py`
- **文档**：`docs/ml_training_api.md`

---

## 二、表单校验与归一（src/pages/prediction/input_form_components）
- **校验器**：`FormValidator.validate_form_data(form_data, gpa_converter)` → 返回 `ValidationError` 对象列表（中文错误消息）
  - 背景院校/专业必填且映射有效
  - GPA 不能为空/不为 0，分制有效
  - 语言分数校验（雅思 0.5 步长、非海外院校不为 0）
  - 经历数量字段非空，经历详情与数量一致性检查
- **GPA 转换**：
  - 优先按院校/国家规则（`config/gpa_conversion_rules.json`）转换
  - 否则按分制上限线性缩放至 4.0 制 → `normalize_gpa`
- **语言分数**：
  - toefl/ielts 互转（基于区间映射表）
  - [0,1] 归一化 → `normalize_language_score`
  - 海外院校语言成绩为选填，非海外院校为必填
- **状态管理**：`FormStateManager` 统一会话键、自动保存与联动处理
  - 自动保存：文本输入 4s 节流，选择操作 1.5s 节流，使用快照哈希去重
  - 历史恢复：从用户级存储（`UserFormStorage`）恢复表单数据
  - 目标选择自动扩展：未选择院校但选择了国家/专业时，自动扩展为所有符合条件的院校
- **跨学院提示**：`cross_faculty_guard.py` 在用户选择跨学院专业时弹出确认对话框
- **文档**：`docs/input_form_components_api.md`

---

## 三、预测（src/pages/prediction）
- **类型定义**：`PredictionInput/PredictionResultItem`（见 `prediction_types.py`）
- **流程**：
  1. **组合生成**（纯函数）：`generate_prediction_combinations` → `(combinations, meta)`
     - 估计总数 ≤ 100：遍历所有组合，使用 `has_school_major_details` 过滤
     - 估计总数 > 100：使用 `get_valid_school_major_set` 预过滤，然后生成有效组合
  2. **并行推理**：`run_single_prediction`（确定性排序与超时兜底）
     - 执行策略：< 128 任务单线程；128-512 任务单线程；≥ 512 任务多线程/多进程（worker 数 = min(4, cpu_count)）
     - 子任务超时：120 秒；结果稳定排序（概率降序→学校→专业）
  3. **结果处理**：`process_prediction_results` 生成三类结果
     - `top_similarity_results`：相似专业推荐（相似度阈值过滤，Top N）
     - `top_cross_major_results`：跨专业推荐（历史存在录取案例，0.8 < similarity < 阈值）
     - `final_user_specified_results`：用户指定组合结果（根据组合数量阈值和相似度阈值过滤）
  4. **编排入口**：`run_prediction_pipeline`（缓存 600s）
     - 加载模型、案例数据、相似度缓存
     - 执行预测、结果调整、合并去重
  5. **背景分析**：`user_background_analyzer.py`（院校层级与替代院校建议）
  6. **结果合并**：`results_handler.py` → `combine_and_deduplicate_results`
     - 优先级：`user_specified` (3) > `cross_major` (2) > `similarity` (1)
     - 跨专业结果仅保留 `admitted == 1` 的项
- **返回模型**：`PredictionResultModel`（三类结果 + 合并去重的 `unified_results`）
- **文档**：`docs/prediction_api.md`

---

## 四、结果调整（src/pages/prediction/result_modifier）
- **概率调整**：
  - `ProbabilityAdjuster`：基于历史案例统计，对 GPA/语言分低于均值的样本施加保守惩罚；极低档次做下限截断（`GPA_MINIMUM=2.0`、`LANGUAGE_MINIMUM=0.6`）
  - `penalize_cross_major_without_cases`：针对用户显式指定的跨专业组合，如果历史中无录取案例，则对概率乘以 `CROSS_MAJOR_PENALTY_FACTOR`（默认 0.5）
- **行业规则**：`professional_adjustment`（针对无实习经历申请商科的情况进行调整）
  - 职业型专业关键词：`PROFESSIONAL_MAJORS = ["Business Administration", "MBA"]`
  - 无实习时降权：默认 `PROFESSIONAL_REDUCTION_FACTOR=0.70`；用户显式指定时 `PROFESSIONAL_USER_SPECIFIED_REDUCTION_FACTOR=0.85`
- **相似度调整**：`adjust_similarity_score` 根据配置 `config/similarity_adjustment_rules.json` 的关键字规则对相似度小幅修正
- **文本加成（TF‑IDF Logit Uplift）**：
  - 外层 `GatedTextBoostProvider` 做门控与缓存；核心 `LogitUpliftProvider` 基于四段文本相似度与计数交互项计算 logit 增量
  - 向量器参数：`analyzer='char_wb'`, `ngram_range=(2,4)`, `min_df=1`, `max_features=20000`（与训练一致）
  - 门控：`sum(s) ≥ sim_gate_sum_min`（默认 0.10）且 `max(s) ≥ sim_gate_max_min`（默认 0.08）才生效
  - 平滑：`smoothing`（默认 0.7）压缩 logit 增量，避免过激
  - 封顶：仅对中段概率（0.2–0.8）生效；`cap_boost = max_total_boost × cap_factor(质量) × scale(p)`
    - `scale(p) = 1 − 2×|p − 0.5|`
    - 质量分数 `quality = 0.7×max(s) + 0.3×mean(s)`，`cap_factor = clamp(cap_min_factor, 1.0, quality^cap_quality_gamma)`
  - 关键配置：`max_total_boost/sim_gate_sum_min/sim_gate_max_min/smoothing/cap_min_factor/cap_quality_gamma`
  - 缓存：基于文本与 count 的签名 LRU（≈512），默认 100ms 内完成；异常回退原概率
- **文档**：`docs/result_modifier_api.md`

---

## 五、专业相似度预计算（scripts/precompute_similarities.py）
- **模型**：Multilingual E5 Instruct（`intfloat/multilingual-e5-large-instruct`）
  - 优先本地加载：`src/services/multilingual-e5-large-instruct`
  - 支持 GPU/CPU，CPU 可选动态量化
- **预计算内容**：
  - 背景专业-目标专业相似度（BG-Target）：用于预测流水线
  - 背景专业-背景专业相似度（BG-BG）：用于相似案例使用
- **缓存格式**：
  - 输出：`cache/background_target_similarity.feather`（两列：`key`、`similarity`）
  - 键规则：按字母序排序后拼接为 `major1|major2`
  - 线上加载：`utils.app_data_loader.load_bg_target_similarity_cache`
- **运行**：
```bash
python scripts/precompute_similarities.py
```
- **文档**：`docs/major_similarity_precompute.md`

---

## 配置清单
- `config/app_config.json`：应用级配置（理论上应当放到 os.env 里）
- `config/dev_config.json`：开发环境配置（debug 模式等）
- `config/gpa_conversion_rules.json`：GPA 分制规则（院校/国家 → 区间映射/兜底公式）
- `config/similarity_adjustment_rules.json`：相似度关键字微调规则
- `config/school_specific_content.json`：学校特定内容配置
- `src/pages/prediction/result_modifier/config.py`：结果调整模块配置（概率阈值、文本加成参数等）

---

## 数据规范
- **训练主表**（`cases.feather` 最小字段）：
  - `admitted`、`background_university`、`background_major`、`target_university`、`target_major`
  - `toefl` 或 `ielts`（至少其一）
  - `research_count/award_count/internship_count/paper_count`
  - 文本列可选（供 TF‑IDF 训练）：`research_detail(s)/award_detail(s)/internship_detail(s)/paper_detail(s)`
- **学校基础表**（`school_base.feather`）：
  - `学校名称`、`国家`、`school_level`、`priority` 等
- **专业详情表**（`school_major_details.feather`）：
  - `学校`、`专业英文名称`、`专业英文名称_聚合`（可选）、`专业大类`、`专业中文名称`（可选）
- **相似度缓存**（`cache/background_target_similarity.feather`）：
  - `key`（格式：`major1|major2`）、`similarity`（float，范围 [0,1]）

---

## 运行与调试
- 安装依赖：`pip install -r requirements.txt`
- 启动应用：`streamlit run main.py`
- 可选环境变量：
  - `PREDICTION_USE_PROCESS_POOL=1`：预测阶段启用进程池（大规模组合更优）。
  - `PREDICTION_MAX_WORKERS`：限制并发工作线程/进程数（默认自适应 CPU 核数与组合规模）。
  - 训练脚本已设置 `LOKY_MAX_CPU_COUNT=4` 限制并行线程数。

---

## 缓存与持久化说明
- **预测缓存**：
  - `run_prediction_pipeline` 使用 `@st.cache_data(ttl=600)`，相同入参 10 分钟内复用结果
  - `_get_cached_cases_df` 使用 `@st.cache_data(ttl=3600)`，案例数据缓存 1 小时
- **资源缓存**：
  - `cached_get_prediction_model` 使用 `@st.cache_resource`（资源缓存，不设 TTL）
  - `cached_load_cases_data` 使用 `@st.cache_data`（数据缓存）
  - `cached_load_bg_target_similarity_cache` 使用 `@st.cache_data`（相似度缓存）
- **文本加成**：
  - TF‑IDF provider 内部有内存级 LRU 缓存（≈512 条目）与超时保护（默认 100ms）
  - 基于文本与 count 的签名进行缓存，异常回退原概率
- **相似度缓存**：
  - 预计算脚本会生成根目录 `cache/*.feather`
  - 线上通过 `utils.app_data_loader.load_bg_target_similarity_cache`/`load_bg_bg_similarity_cache` 读入
  - 缓存键由 `get_cached_major_similarity_key` 统一生成，避免中英文与空格差异导致 miss
- **用户表单**：
  - `FormStateManager` 会根据用户登录状态进行本地持久化（详见 `src/utils/user_form_storage.py`）
  - 自动保存：文本输入 4s 节流，选择操作 1.5s 节流，使用快照哈希去重

---

## 常见问题排查
- **模型无法加载**：
  - 检查 `pre-trained_models/*.ubj` 是否存在
  - Booster 属性包含 `feature_names/calibration_params/level_fallback_mapping` 等
  - 若属性缺失则回退读取同名 JSON（若存在）
  - 文本 TF‑IDF 模型为 `.joblib`（`tfidf_vectorizer.joblib`）
- **特征不对齐**：
  - 线上与训练的 `feature_names` 必须一致
  - 新增特征需同步训练与发布
  - 分类特征新类别导致编码不一致：训练/推理统一将分类列转为 pandas `category` 并对齐特征列
- **文本加成无效**：
  - 确认配置 `enabled/model_paths/sim_gate_sum_min/sim_gate_max_min/smoothing/cap_*` 是否合理
  - 确认三件产物是否存在：`tfidf_vectorizer.joblib`、`tfidf_centroids.npz`、`text_uplift_weights.json`
  - 检查文本相似度是否达到门控阈值（`sum(s) ≥ sim_gate_sum_min` 且 `max(s) ≥ sim_gate_max_min`）
- **相似度均为 0**：
  - 检查缓存文件是否生成（`cache/background_target_similarity.feather`）
  - 检查键名格式是否一致（`major1|major2`，按字母序排序）
  - 确认使用 E5 模型并采用 `query:` 提示词风格
- **预测结果异常**：
  - 检查输入数据是否完整（背景院校/专业、GPA、语言成绩等）
  - 检查目标院校/专业是否在详情表中存在
  - 检查相似度缓存是否加载成功

---

## 部署说明
- 单机：`streamlit run main.py --server.address localhost` 即可启动。
- 框架：streamlit前端和python后端耦合（后续可能会解耦django前端+Fastapi后端）。
- 环境参数：参考config/dev_config.json确认debug模式是否开启，关闭后再check src/utils/env_config_loader.py里的环境配置。

---

## 文档索引
- `docs/ml_training_api.md`：机器学习训练管线文档
- `docs/input_form_components_api.md`：表单组件与校验 API 文档
- `docs/prediction_api.md`：预测模块 API 文档
- `docs/result_modifier_api.md`：结果修正模块文档
- `docs/major_similarity_precompute.md`：专业相似度预计算文档

---

维护人：lijiapeng8@xdf.cn
版本：v2.9
