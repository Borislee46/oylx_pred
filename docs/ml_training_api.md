# 机器学习训练管线

| 项 | 说明 |
|----|------|
| 源码路径 | `src/machine_learning_models/` |
| 训练入口 | `python -m src.machine_learning_models.train`（见 `train.py`） |
| 目标变量 | 二分类列 `admitted`（`data_config.TARGET_COLUMN`） |
| 主模型 | XGBoost + 概率校准（`CalibratedClassifierCV`，sigmoid，`cv='prefit'`） |

本文档描述离线训练流水线的模块职责、数据契约与产物格式，与仓库内实现一致；行为细节以源码为准。

## 1. 模块映射

| 文件 | 职责 |
|------|------|
| `data_loader.py` | 读取 `cases.feather`，特征工程入口，训练/测试划分与采样 |
| `feature_engineer.py` | 缺失值、分类编码、计数列截尾与 `log1p`、语言成绩合并为 `language_score` |
| `sampling_methods.py` | SMOTE/SMOTENC，`k_neighbors` 随少数类规模调整；合成样本的 `sample_weight` 对齐 |
| `model_trainer.py` | XGBoost 训练、单调约束、校准、评估与特征重要度聚合 |
| `hyperparameter_tuning.py` | Optuna 超参搜索（CV 指标为 F1） |
| `data_config.py` | 列集合、校准方式、单调白名单、默认阈值等常量 |
| `utils.py` | 模型 `.ubj` 落盘；向 Booster 写入 `feature_names`、`calibration_params`、`level_fallback_mapping`；评估 JSON |
| `train.py` | CLI：`--model` / `--sampling_method` / `--auto_tune` 等 |

文本加成（TF-IDF + Logit uplift）离线训练见 `scripts/train_text_tfidf.py`，产物默认写入 `src/machine_learning_models/pre-trained_models/`。

## 2. 特征与数据契约

- **分类特征**：`background_university`、`background_major`、`target_university`、`target_major` → `pandas.category`。
- **文本列**：训练阶段用于清洗与缺失处理，不直接作为树模型输入；全空 detail 文本会触发样本降权（见 `data_config` 与 `sampling_methods` 中的权重逻辑）。
- **计数特征**：`research_count`、`award_count`、`internship_count`、`paper_count` — 分位数截尾后 `log1p`。
- **语言成绩**：托福/雅思在特征工程中归一并为单列 `language_score`（训练集拟合统计量，测试集仅变换）。

## 3. 采样与样本权重

- SMOTENC 使用分类列索引；失败时回退未采样数据与原权重。
- 文本全空样本：`TEXT_EMPTY_SAMPLE_WEIGHT` 折减。
- 最近样本：`RECENT_SAMPLE_BOOST_COUNT` / `RECENT_SAMPLE_BOOST_WEIGHT` 对尾部行加权。
- `sample_weight` 在训练链路中统一为 `float32`，避免校准阶段 dtype 不一致。

## 4. 训练与校准

- **单调约束**：分类列为 0；数值列由 `MONOTONE_INCREASING_WHITELIST` / `MONOTONE_DECREASING_WHITELIST` 指定 ±1，其余为 0。
- **校准**：训练集 80% 拟合基分类器，20% 拟合校准器；校准参数抽取后写入 Booster 属性供线上读取。
- **不平衡**：`scale_pos_weight` 按类频自动计算。
- **已校准模型的重要度**：从各子估计器基模型提取重要度再聚合（见 `model_trainer.py` 实现）。

## 5. 产物

- **模型**：`src/machine_learning_models/pre-trained_models/` 下 `{model}_{timestamp}.ubj`（见 `utils.py`）。
- **评估**：`evaluation_results/{model}_evaluation_{timestamp}.json`（metrics、feature_importance、model_params 等）。

## 6. 运行参数

- 默认数据：`src/machine_learning_models/data/cases.feather`（以 `train.py` 为准）。
- 并行：`train.py` 设置 `LOKY_MAX_CPU_COUNT=4` 限制 joblib/loky 线程数。

## 7. TF-IDF 与文本 Uplift 训练（`scripts/train_text_tfidf.py`）

流程概要：

1. 从 `cases.feather` 读取并清洗文本列；五类 detail 中用于“空文本降权”判定，Uplift 训练使用前四类（与脚本内注释一致）。
2. 训练字符级 TF-IDF（ngram、特征上限等见脚本）。
3. 计算录取子集上的文本质心并 L2 归一化。
4. 基于香农熵等信息量指标构造特征；用 XGBoost 基模型 logit 与标签的残差拟合权重（NNLS，失败则 Ridge `positive=True`）。
5. 输出：`tfidf_vectorizer.joblib`、`tfidf_centroids.npz`、`text_uplift_weights.json`。

```bash
python scripts/train_text_tfidf.py
```

线上推理需与训练使用相同的特征列约定；录取预测主流程的后处理见 `src/pages/prediction/result_modifier/`。

## 8. 故障排查

| 现象 | 检查项 |
|------|--------|
| 目标列缺失 | `data_config.TARGET_COLUMN` 与 Feather 列名一致 |
| 推理编码不一致 | 分类列 `category` 与 `feature_names` 顺序与训练一致 |
| 极不均衡 | 优先 SMOTENC；查看采样回退日志 |
| 采样与权重并用 | 合成少数类样本权重按类内均值填充（见 `sampling_methods.py`） |

---

维护：与 `src/machine_learning_models/` 同步更新。
