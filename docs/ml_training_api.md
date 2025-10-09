## 机器学习训练管线文档（src/machine_learning_models）

本模块提供从数据加载、特征工程、采样、训练与校准、评估与落盘的一整套训练流水线（以 XGBoost 为主）。

### 目录结构与职责
- `data_loader.py`：加载 cases.feather（默认 `src/machine_learning_models/data/cases.feather`），特征工程，划分训练/测试，按需采样。
- `feature_engineer.py`：缺失值处理、分类特征编码、计数字段截尾与对数化、语言成绩统一为 `language_score`。
- `sampling_methods.py`：不平衡采样（ADASYN、SMOTE/SMOTENC、随机过采样/欠采样）与回退策略。
- `model_trainer.py`：模型训练（XGBoost）、单调约束、`CalibratedClassifierCV` 概率校准（cv='prefit' 单一校准器）、评估与特征重要度聚合。
- `hyperparameter_tuning.py`：Optuna 超参搜索（CV 内部使用 F1）。
- `data_config.py`：列名与常量（目标列、分类列、计数列、校准方式等）。
- `utils.py`：模型与评估结果落盘（.model + JSON）。
- `train.py`：命令行入口（--model/--sampling_method/--auto_tune）。

### 数据与特征
- 目标列：`admitted`（二分类）。
- 分类列：`background_university/background_major/target_university/target_major` → 转为 pandas `category`（`categorical_features_processor.prepare_categorical_columns`）。
- 文本列：仅参与清洗与填补，不直接入模（训练侧）。
  - 同时用于样本权重判定：当所有 detail 文本列均为空时，该样本在训练中会被轻度降权（见“样本权重与最近样本加权”）。
- 计数字段：`research_count/award_count/internship_count/paper_count`
  - 处理：99 分位截尾 → `log1p` 对数变换。
- 语言成绩统一：
  - 若同时有 `toefl/ielts`：各自归一化后取 max，落地为 `language_score`；原列删除。
  - 仅存在其一时按比例归一化。
 - 训练/测试一致性：特征工程在训练集拟合，测试集仅做变换；使用训练集统计（中位数、0.99 分位 cap、类别编码对齐）。

### 采样（sampling_methods.py）
- 自动识别分类特征索引用于 SMOTENC。
- ADASYN/SMOTE：正类样本过少时回退到 `RandomOverSampler`；`k_neighbors` 动态设置。
- 随机过/欠采样：提供基础策略，异常回退到原始数据。

### 样本权重与最近样本加权（data_loader.py / sampling_methods.py / data_config.py）
- 样本权重来源：
  - `TEXT_EMPTY_SAMPLE_WEIGHT`：当一条样本的所有 detail 文本列（`TEXT_COLUMNS`）均为空时，训练样本权重设为该折减系数（默认 0.85），其余为 1.0。
  - 最近样本加权：从数据集最后一行往上数 `RECENT_SAMPLE_BOOST_COUNT` 条样本，乘以 `RECENT_SAMPLE_BOOST_WEIGHT`（默认 10000 条、1.1）。
- 与采样的协同：
  - 采样函数支持同时传入并返回 `sample_weight`。
  - 当采样器提供 `sample_indices_` 时，按采样后索引对齐原权重；
  - 对新合成的少数类样本，权重采用该类样本权重的均值进行填充；
  - 若采样器不暴露 `sample_indices_`（如部分实现），则退化为按原序列权重对齐并同样为新增样本填充类别均值。
- 类型与稳定性：
  - 训练阶段将 `sample_weight` 转为 float32，避免校准阶段的 dtype mismatch 问题。
  - 所有策略在异常时均回退到原始数据与原始权重，保证训练可用性。

### 训练与校准（model_trainer.py）
- 单调约束（XGBoost）：
  - 分类特征约束 0，其他特征约束 +1（单调递增），用于符合业务单调性假设。
- 两种模式：
  - 固定参数（默认，业务验证过的一组最佳参数）。
  - 自动调参：Optuna 搜索（`hyperparameter_tuning.py`），随后带入校准。
- 概率校准（sigmoid，单一校准器）：
  - 使用 `cv='prefit'` 策略：训练集 80% 拟合基础 XGB，剩余 20% 作为校准集拟合 `CalibratedClassifierCV(method='sigmoid', cv='prefit')`。
  - 训练结束立即提取校准参数 a/b 以便落盘并在部署时还原。
 - 类别不平衡：自动计算 `scale_pos_weight = #neg/#pos` 并在训练或调参后的最终参数中应用。
- 评估指标：`accuracy/precision/recall/f1`（二分类）。
- 特征重要度：优先从基础模型读取；如为多折结构则做可用性聚合。

### 评估与落盘（utils.py）
- 模型保存（仅 .model + JSON）：
  - `pre-trained_models/{model}_{timestamp}.model`（XGBoost 原生权重）
  - `pre-trained_models/{model}_{timestamp}_features.json`（特征名列表）
  - `pre-trained_models/{model}_{timestamp}_calibration.json`（若使用校准则包含 sigmoid 的 a、b 参数）
- 评估结果：`evaluation_results/{model}_evaluation_{timestamp}.json`，包含：
  - `metrics`、`feature_importance`（可选）、`model_params`、`sampling_method`、`calibration_method`、`auto_tune_method`。

### 训练入口（train.py）
```bash
python -m src.machine_learning_models.train --model xgboost --sampling_method smote
```
流程：数据加载 → 特征工程 → 划分/采样 → 训练+校准 → 评估 → 保存模型与评估报告。
 - 默认数据路径：`src/machine_learning_models/data/cases.feather`。
 - 并行限制：训练脚本内部设置 `LOKY_MAX_CPU_COUNT=4` 以限制并行线程数。

### 与线上预测的契合点
- 线上 `PredictionModel` 会加载同一套 `feature_names` 并在推理时对输入进行相同的列对齐与类别编码。
- 线上优先加载 `.model` 并自动读取同时间戳的 `_calibration.json` 应用概率校准；文本 TF‑IDF 模型统一采用 `.joblib` 存储（`tfidf_vectorizer.joblib`）。
- 线上页面已有额外后处理（结果修正/文本加成/跨专业惩罚），与训练管线解耦。

### 常见问题
- “目标列缺失/NaN”：确认 `data_config.TARGET_COLUMN` 与特征工程步骤对齐。
- “分类特征新类别导致编码不一致”：线上通过 `global_categories_` 约束；训练时尽量覆盖主流类别。
- “正负样本极度不均衡”：优先尝试 SMOTENC（含分类索引），否则回落到随机过采样。
- “需要同时使用采样与样本权重”：已支持。若使用 SMOTE/ADASYN 等，会自动对新增少数类样本以类别权重均值填充。
- “训练环境”：同requiremetents.txt，python==3.12（预备使用xgboost3.1.0最新版本针对类别变量re-encoder编码）。

---
维护人：lijiapeng8@xdf.cn
版本：v2.6


