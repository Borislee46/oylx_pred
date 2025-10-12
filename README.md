## EasyApply 选校预测系统

本数据产品提供一套从“模型训练 → 表单校验与归一 → 预测编排 → 结果调整 → 智能组合优化（可选）”的端到端解决方案。

### 架构流程
1) **模型训练 (Offline)**: `src/machine_learning_models/`
   - **核心**: 使用 XGBoost 结合 ADASYN/SMOTE 等采样方法处理不平衡数据，并进行概率校准。
   - **产物**: 生成 `XGBoost` 主模型和 `TF-IDF` 文本模型。

2) **用户表单 (Online)**: `src/pages/prediction/input_form_components/`
   - **核心**: `FormStateManager` 管理复杂的UI状态，支持**自动保存**和**历史恢复**。`FormValidator` 和 `GPAConverter` 对用户输入进行严格的校验和智能转换。

3) **预测编排 (Online)**: `src/pages/prediction/`
   - **核心**: `prediction_handler` 负责编排整个流水线；`run_prediction` 采用线程/进程池自适应并发（可通过环境变量启用进程池）并进行稳定排序与超时兜底。

4) **结果调整 (Online)**: `src/pages/prediction/result_modifier/`
   - **核心**: 对模型原始概率进行一系列专家规则微调，包括基于历史案例的 `ProbabilityAdjuster`、针对无实习申商科的 `professional_adjustment`，以及基于软文案的**文本加成** (`keyword` / `tfidf`)。

5) **智能优化 (Optional Online)**: `src/pages/prediction/school_combination_optimizer_algorithm/`
   - **核心**: 使用 **NSGA-III 多目标优化算法**结合蒙特卡洛仿真，为用户生成风险和收益平衡的选校组合策略。

---

## 一、训练（src/machine_learning_models）
- 数据与特征：
  - 目标列：`admitted`
  - 分类列：`background_university/background_major/target_university/target_major`
  - 计数列：`research_count/award_count/internship_count/paper_count`（99%分位截尾 + log1p）
  - 语言列：统一为 `language_score`（toefl/ielts 归一后取 max）
- 采样：ADASYN、SMOTE/SMOTENC、随机过/欠采样（极端不平衡时回退策略）
- 训练与校准：XGBoost + 单调约束 + CalibratedClassifierCV(sigmoid, cv='prefit'，训练集内部按 80/20 拆分校准)
- 入口命令（参数，数据集和详细训练流程暂不公开，有需要请联系lijiapeng8@xdf.cn）：
```bash
python -m src.machine_learning_models.train --model xgboost --sampling_method smote
```
- 产物：
  - 模型：`src/machine_learning_models/pre-trained_models/{model}_{timestamp}.model`
  - 特征/校准：`{model}_{timestamp}_features.json` / `{model}_{timestamp}_calibration.json`
  - 评估：`src/machine_learning_models/evaluation_results/{model}_evaluation_{timestamp}.json`
- 文档：`docs/ml_training_api.md`

---

## 二、表单校验与归一（src/pages/prediction/input_form_components）
- 校验器：`FormValidator.validate_form_data(form_data, gpa_converter)` → 返回中文错误数组
- GPA 转换：按院校/国家规则或线性缩放至 4.0 制 → `normalize_gpa`
- 语言分数：toefl/ielts 互转与[0,1]归一 → `normalize_language_score`
- 状态管理：`FormStateManager` 统一会话键、自动保存与联动处理
- 文档：`docs/input_form_components_api.md`

---

## 三、预测（src/pages/prediction）
- 类型：`PredictionInput/PredictionResultItem`（见 `prediction_types.py`）
- 流程：
  - 组合生成（纯函数）：`generate_prediction_combinations` → `(combinations, meta)`
  - 并行推理：`run_single_prediction`（确定性排序与超时兜底）
  - 编排入口：`run_prediction_pipeline`（缓存 600s）
  - 背景分析：`src/pages/prediction/user_background_analyzer.py`（院校层级与替代院校建议）
  - 结果合并：`src/pages/prediction/results_handler.py`
  - 返回模型：`PredictionResultModel`（三类结果 + 合并去重）
- 文档：`docs/prediction_api.md`

---

## 四、结果调整（src/pages/prediction/result_modifier）
- **概率调整**: `ProbabilityAdjuster`（基于历史案例统计，对 GPA/语言分进行保守惩罚）与 `penalize_cross_major_without_cases`（对无历史成功案例的跨专业申请进行惩罚）。
- **行业规则**: `professional_adjustment`（针对无实习经历申请商科的情况进行调整）。
- **文本加成策略**: Keyword 与 TF‑IDF 双通道（MaxOfTwo），外层由 `GatedTextBoostProvider` 做门控与缓存，受统一 `max_total_boost` 约束。
  - 关键词表已补充顶会/顶刊/机构与高价值竞赛（如 `NeurIPS/ICML/ICLR/CVPR/ACL/KDD/SIGMOD/AAAI/IJCAI/Nature Communications/PNAS/TPAMI`；`IMO/IOI/ICPC/丘成桐` 等）。
  - TF‑IDF 训练：`analyzer='char_wb'`, `ngram_range=(2,4)`, `min_df=2`, `max_features=20000`；训练/推理 `scikit-learn==1.4.2`
  - TF‑IDF 仅对中段概率（0.2–0.8）加成；带“强信号门控”（需命中 top-tier 关键词）
  - 组合策略为 MaxOfTwo：同一位置取两者提升更大者。
- 文档：`docs/result_modifier_api.md`

---

## 五、智能组合优化（可选）
- 模块：`src/pages/prediction/school_combination_optimizer_algorithm/`
- 方法：NSGA-III + 背景/学院规则过滤 + TOPN学校强约束 + 蒙特卡洛相关性仿真（支持 batch_size）+ 平衡启发式回退
- 输入：候选 `[{university, major, probability, is_new_major}]`、用户背景（专业、学院等）与可选相关性矩阵（索引格式 `"{university} - {major}"`）
- 输出：多套方案（类型/学校清单/指标/目标值）与自适应阈值
- 文档：`docs/school_combination_optimizer_api.md`
 - 配置补充：跨学院申请规则、概率校准与 prestige 权重等详见文档“配置常量”章节。
 - 其他：支持 `clear_cache()` 清理内部缓存；`visualize_recommendations` 快速可视化；提供 `probability_utils.calibrate_cross_major_probabilities` 跨专业轻度校准。

---

## 配置清单
- `config/app_config.json`：应用级配置
- `config/gpa_conversion_rules.json`：GPA 分制规则（院校/国家 → 区间映射/兜底公式）
- `config/similarity_adjustment_rules.json`：相似度关键字微调规则

---

## 数据规范（训练主表 cases.csv 最小字段）
- `admitted`、`background_university`、`background_major`、`target_university`、`target_major`
- `toefl` 或 `ielts`（至少其一）
- `research_count/award_count/internship_count/paper_count`
- 文本列可选（供 TF‑IDF 训练）：`research_detail(s)/award_detail(s)/internship_detail(s)/paper_detail(s)`

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
- 预测缓存：`run_prediction_pipeline` 使用 `@st.cache_data(ttl=600)`，相同入参 10 分钟内复用结果。
- 文本加成：TF‑IDF provider 内部有内存级缓存与超时保护（默认 100ms）。
- 相似度缓存：
  - 预计算脚本会生成根目录 `cache/*.feather`；线上通过 `utils.app_data_loader.load_bg_target_similarity_cache`/`load_bg_bg_similarity_cache` 读入。
  - 缓存键由 `get_cached_major_similarity_key` 统一生成，避免中英文与空格差异导致 miss。
- 用户表单：`FormStateManager` 会根据用户登录状态进行本地持久化（详见 `src/utils/user_form_storage.py`）。

---

## 常见问题排查
- 模型无法加载：检查 `pre-trained_models/*.model` 及同时间戳的 `*_features.json`/`*_calibration.json` 是否齐全；版本不兼容时重新训练导出。文本 TF‑IDF 模型为 `.joblib`（`tfidf_vectorizer.joblib`）。
- 特征不对齐：线上与训练的 `feature_names` 必须一致；新增特征需同步训练与发布。
- 文本加成无效：确认配置 `enabled/model_paths/similarity_thresholds`。
- 相似度均为 0：检查缓存文件是否生成、键名格式是否一致。

---

## 部署说明
- 单机：`streamlit run main.py --server.address localhost` 即可启动。
- 框架：streamlit前端和python后端耦合（后续可能会解耦django前端+Fastapi后端）。
- 环境参数：参考config/dev_config.json确认debug模式是否开启，关闭后再check src/utils/env_config_loader.py里的环境配置。

---

维护人：lijiapeng8@xdf.cn
版本：v2.7
