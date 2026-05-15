# Machine Learning Models 模块技术文档

## 1. 模块概述

`src/machine_learning_models` 是离线 XGBoost 训练流水线，负责从历史案例数据训练录取概率预测模型。完整流程覆盖数据加载 → 特征工程（含 monotonic constraints 特征）→ SMOTE/SMOTENC 不平衡采样 → XGBoost 训练 → CalibratedClassifierCV(sigmoid) 概率校准 → 可选 Optuna 超参调优 → 模型评估 → 保存为 `.ubj` 格式。

### DS 视角

训练流水线的几个关键 DS 决策值得追问：

- **Monotonic constraints 是双刃剑**：强制 GPA/语言/经历单调递增保证了业务可解释性（DEC-006），但也限制了模型学习非线性模式。如果真实录取中存在"GPA 太高反而减分"的现象（overqualified），约束会掩盖这个信号。有没有验证过无约束模型是否真的违反单调性？
- **SMOTE 在类别特征上可能制造无意义样本**：DEC-005 明确不在一键推理时用 SMOTE，但训练时用了 SMOTENC。在 1051 个专业的稀疏空间中插值，可能生成"不存在的专业组合"——这对比 XGBoost 的 tree split 影响有多大？
- **Platt scaling (sigmoid) 校准后输出范围被压缩到 [0.13, 0.72]**：这是 DEC-001 记录的已知行为（"nothing is certain in education"）。但从 DS 角度，训练集本身有 33.7% 的录取率，最高 72%、最低 13% 的校准输出意味着模型在最乐观时也只给 72%、最悲观时也给 13%——这对边界 case 的信息量很差。
- **没有 held-out test set**：校准和评估都在训练数据上完成。外部 ApplySquare 数据上 -67pp 是唯一真正的 out-of-sample 检验。

## 2. 目录结构

```
machine_learning_models/
├── __init__.py                # 包标记（无公共 API 导出）
├── train.py                   # CLI 入口：python -m src.machine_learning_models.train
├── data_config.py             # 数据配置：列定义、阈值、特征分组
├── data_loader.py             # 数据加载：cases.feather → train/test split
├── feature_engineer.py        # 特征工程：编码、归一化、monotonic 特征构建
├── model_trainer.py           # XGBoost 训练 + calibration
├── hyperparameter_tuning.py   # Optuna 超参搜索（auto_tune 模式）
├── school_level_mapper.py     # 院校等级映射（训练用，985/211/双非 → tier）
├── utils.py                   # 模型保存、评估结果导出
├── data/                      # 训练数据（cases.feather + school_major_details.feather）
└── pre-trained_models/        # 训练好的 .ubj 模型文件
```

## 3. 端到端流程

```
python -m src.machine_learning_models.train [--auto_tune]
        │
        ├── data_loader.load_data(data_path)
        │       ├── 读取 cases.feather
        │       ├── train/test split（时间序列分割）
        │       └── 返回 X_train, X_test, y_train, y_test, feature_names
        │
        ▼
model_trainer.train_model(X_train, y_train, model_type, auto_tune)
        │
        ├── feature_engineer 特征转换
        │       ├── 数值特征归一化
        │       ├── 类别特征编码
        │       ├── monotonic constraints 特征（GPA↑、语言↑、经历数↑）
        │       └── SMOTE/SMOTENC 采样
        │
        ├── [auto_tune] hyperparameter_tuning (Optuna)
        │       ├── 搜索空间：max_depth, learning_rate, n_estimators, ...
        │       ├── 目标：AUC
        │       └── monotonic constraints 强制保留
        │
        ├── XGBoost 训练（monotone_constraints）
        │
        ├── CalibratedClassifierCV(sigmoid, prefit)
        │       └── 概率校准（使 predict_proba 更准确）
        │
        └── 返回 model, model_params, calibration_method, calibration_params
        │
        ▼
evaluate_model(model, X_test, y_test, feature_names)
        ├── Accuracy, AUC, F1, Precision, Recall
        └── Feature Importance
        │
        ▼
utils.save_model() → pre-trained_models/*.ubj
        │
        ▼
run_post_train_smoke_check()
        ├── 重新加载模型
        ├── 校验 feature_names 一致性
        ├── 校验 prediction_threshold 一致性
        └── 执行 predict_proba 冒烟测试
```

## 4. 核心组件

### 4.1 train.py

CLI 入口，编排全流程：
- 参数：`--model xgboost`（当前唯一选项）、`--auto_tune`（启用 Optuna 调参）
- 执行：数据加载 → 训练 → 评估 → 保存 → 冒烟自检
- `run_post_train_smoke_check()`：训练后自动重载模型并校验指纹

### 4.2 data_config.py

训练数据配置：
- 特征列分组（数值/类别/文本）
- monotonic constraints 列定义
- `DEFAULT_PREDICTION_THRESHOLD`：二分类阈值

### 4.3 feature_engineer.py

特征工程流水线：
- 数值特征：StandardScaler 归一化
- 类别特征：OrdinalEncoder
- 文本特征：TF-IDF（可选）
- Monotonic 特征构建：GPA、语言成绩、四段经历计数（强制单调递增）

### 4.4 model_trainer.py

模型训练核心：
- `train_model()`：XGBoost + CalibratedClassifierCV(sigmoid, prefit)
- SMOTE/SMOTENC 处理类别不平衡
- `evaluate_model()`：多维度评估 + 特征重要性

### 4.5 hyperparameter_tuning.py

Optuna 超参搜索（`--auto_tune` 启用）：
- 搜索空间：`max_depth`、`learning_rate`、`n_estimators`、`subsample`、`colsample_bytree` 等
- 目标函数：验证集 AUC
- Monotonic constraints 保持

### 4.6 pre-trained_models/

训练产物：
- `*.ubj`：XGBoost 模型文件（UBJSON 格式）
- `*_features.json`：特征列表
- `*_calibration.json`：校准参数
- `*_metadata.json`：训练元数据

## 5. 关键命令

```bash
# 基础训练
python -m src.machine_learning_models.train --data-path data/cases.feather

# 带超参调优
python -m src.machine_learning_models.train --data-path data/cases.feather --auto_tune
```

## 6. 依赖

- `xgboost`：模型训练（monotone_constraints）
- `scikit-learn`：CalibratedClassifierCV、SMOTE、评估指标
- `pandas`、`numpy`：数据处理
- `optuna`（可选）：超参调优
- `imbalanced-learn`：SMOTE/SMOTENC
- [Utils](src/utils/README.md) — `model_loader`（模型加载工具）
