# 机器学习训练管线文档

**路径**: `src/machine_learning_models`

本模块提供从数据加载、特征工程、采样、训练与校准、评估与落盘的一整套训练流水线（以 XGBoost 为主）。

> （脚本暂不公开训练流水线，有需要请联系 lijiapeng8@xdf.cn）

## 1. 目录结构与职责

- `data_loader.py`：加载 cases.feather，特征工程，划分训练/测试，按需采样。
- `feature_engineer.py`：缺失值处理、分类特征编码、计数字段截尾与对数化、语言成绩统一为 `language_score`。
- `sampling_methods.py`：不平衡采样（SMOTE/SMOTENC，动态 k_neighbors），并对采样后的 `sample_weight` 做对齐与新增样本权重填充。
- `model_trainer.py`：模型训练（XGBoost）、单调约束、`CalibratedClassifierCV` 概率校准（cv='prefit' 单一校准器）、评估与特征重要度聚合。
- `hyperparameter_tuning.py`：Optuna 超参搜索（CV 内部使用 F1）。
- `data_config.py`：列名与常量（目标列、分类列、计数列、校准方式、单调约束白名单等）。
- `utils.py`：模型与评估结果落盘（.ubj；将特征名/校准参数/学校层级回退映射写入 Booster 属性；评估 JSON）。
- `train.py`：命令行入口（--model/--sampling_method/--auto_tune）。

## 2. 数据与特征

- **目标列**：`admitted`（二分类）。
- **分类列**：`background_university`, `background_major`, `target_university`, `target_major` → 转为 pandas `category`。
- **文本列**：仅参与清洗与填补，不直接入模（训练侧）。
  - 同时用于样本权重判定：当所有 detail 文本列均为空时，该样本在训练中会被轻度降权。
- **计数字段**：`research_count`, `award_count`, `internship_count`, `paper_count`
  - 处理：99 分位截尾 → `log1p` 对数变换。
- **语言成绩统一**：
  - 若同时有 `toefl/ielts`：各自归一化后取 max，落地为 `language_score`；原列删除。
  - 仅存在其一时按比例归一化。
  - 训练/测试一致性：特征工程在训练集拟合，测试集仅做变换；使用训练集统计。

## 3. 采样 (`sampling_methods.py`)

- 自动识别分类特征索引用于 SMOTENC。
- 使用 SMOTE/SMOTENC，`k_neighbors` 基于少数类样本量动态设置。
- **权重对齐**：
  - 若采样器提供 `sample_indices_` 则按索引对齐；
  - 对新合成的少数类样本，以该类样本权重均值填充。
- 异常时回退到原始数据与原始权重，保证训练可用性。

## 4. 样本权重与最近样本加权

- **样本权重来源**：
  - `TEXT_EMPTY_SAMPLE_WEIGHT`：当一条样本的所有 detail 文本列均为空时，训练样本权重设为折减系数（默认 0.85），其余为 1.0。
  - **最近样本加权**：从数据集最后一行往上数 `RECENT_SAMPLE_BOOST_COUNT` 条样本，乘以 `RECENT_SAMPLE_BOOST_WEIGHT`（默认 10000 条、1.1）。
- **类型与稳定性**：
  - 训练阶段将 `sample_weight` 转为 float32，避免校准阶段的 dtype mismatch 问题。
  - 所有策略在异常时均回退到原始数据与原始权重。

## 5. 训练与校准 (`model_trainer.py`)

- **单调约束 (XGBoost)**：
  - 分类特征约束 0；其余特征通过白名单控制：`MONOTONE_INCREASING_WHITELIST` (+1), `MONOTONE_DECREASING_WHITELIST` (-1)，未列入者为 0。
- **两种模式**：
  - 固定参数（默认，业务验证过的一组最佳参数）。
  - 自动调参：Optuna 搜索，随后带入校准。
- **概率校准 (sigmoid, 单一校准器)**：
  - 使用 `cv='prefit'` 策略：训练集按 8:2 划分，80% 拟合基础 XGB，剩余 20% 作为校准集拟合 `CalibratedClassifierCV`。
  - 训练结束立即提取校准参数 a/b 以便落盘并在部署时还原。
- **特征重要度聚合**：
  - 对于已校准模型，系统会提取各校准子估计器（calibrated classifiers）的基础模型重要度，并取其均值进行聚合展示。
- **类别不平衡**：自动计算 `scale_pos_weight` 并在训练中应用。

## 6. 评估与落盘 (`utils.py`)

- **模型保存 (.ubj)**：
  - `pre-trained_models/{model}_{timestamp}.ubj`（XGBoost 序列化权重）。
  - 将以下信息写入 Booster 属性：`feature_names`, `calibration_params`, `level_fallback_mapping`。
- **评估结果**：
  - `evaluation_results/{model}_evaluation_{timestamp}.json`，包含：metrics, feature_importance, model_params 等。

## 7. 训练入口 (`train.py`)

```bash
python -m src.machine_learning_models.train
```

流程：数据加载 → 特征工程 → 划分/采样 → 训练+校准 → 评估 → 保存模型与评估报告。
- 默认数据路径：`src/machine_learning_models/data/cases.feather`。
- 并行限制：训练脚本内部设置 `LOKY_MAX_CPU_COUNT=4` 以限制并行线程数。

## 8. 文本加成 (TF‑IDF + Logit Uplift) 训练

脚本：`scripts/train_text_tfidf.py`

目标：为线上 `LogitUpliftProvider` 生成三类产物，用于“文本提升”后处理。

### 产物
- `tfidf_vectorizer.joblib`：字符级 TF‑IDF 向量器（ngram 为 2-4，特征上限 20000）。
- `tfidf_centroids.npz`：四段文本（科研/获奖/实习/论文）的归一化质心（Centroids）。
- `text_uplift_weights.json`：非负的增益权重（基础项 + 与计数交互项）。

### 流程概览
1. **输入与预处理**：数据源为 `cases.feather`，对文本列进行归一化清洗。包含 `activity` 在内的 5 类文本用于样本降权判定，但仅前 4 类参与 Uplift 训练。
2. **向量器训练**：使用 `char_wb` 分析器，学习文本的字符级特征。
3. **质心计算**：提取录取案例中的背景特征中心点，并进行 L2 归一化。
4. **有效信息密度 (Entropy)**：计算文本的香农熵以评估丰富度，饱和阈值设为 5.0。
5. **增益权重拟合**：
   - 计算真实标签与 XGBoost 基础概率之间的 Logit 残差。
   - 构造包含“质量得分”与“质量×数量”交互项的特征矩阵。
   - 使用 **NNLS (非负最小二乘法)** 拟合权重，若结果异常则回退至 **Ridge (positive=True)** 回归。
6. **落盘**：默认保存至 `src/machine_learning_models/pre-trained_models/`。

### 运行

```bash
python scripts/train_text_tfidf.py
```

### 与线上预测的契合点
- 线上会加载同一套 `feature_names` 并在推理时对输入进行相同的列对齐与类别编码。
- 线上优先加载 `.ubj`，并从 Booster 属性读取 `feature_names/calibration_params`。
- 线上页面已有额外后处理（结果修正/文本加成/跨专业惩罚），与训练管线解耦。

## 9. 常见问题

- **目标列缺失/NaN**：确认 `data_config.TARGET_COLUMN` 与特征工程步骤对齐。
- **分类特征新类别导致编码不一致**：训练/推理统一将分类列转为 pandas `category` 并对齐特征列。
- **正负样本极度不均衡**：优先使用 SMOTENC（含分类索引）；异常时回退到原始数据。
- **需要同时使用采样与样本权重**：已支持。使用 SMOTE/SMOTENC 时，会对新增少数类样本以该类样本权重均值填充。

---

> **维护人**: lijiapeng8@xdf.cn
> **版本**: v2.7
