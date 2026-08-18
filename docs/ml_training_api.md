# 机器学习训练管线文档

**路径**: `src/ml`（2026-06 从 `src/machine_learning_models` 迁移）

本模块提供从数据加载、特征工程、训练与校准、评估与落盘的一整套 XGBoost 训练流水线。

## 1. 目录结构与职责

| 文件 | 职责 |
|------|------|
| `data_loader.py` | 加载 cases.feather → 特征工程 → 划分训练/测试（支持 random / time 两种切分） |
| `feature_engineer.py` | 特征工程 facade：缺失值处理、categorical 对齐、log1p 变换、语言归一化、内化特征注入（可选） |
| `model_trainer.py` | 训练 facade：XGBoost + isotonic 校准 + 评估 + 特征重要度 |
| `hyperparameter_tuning.py` | Optuna 超参搜索（支持 log_loss / brier 双目标） |
| `data_config.py` | 列名与常量（目标列、分类列、计数列、校准方式、单调约束白名单等） |
| `te_encoder.py` | TargetEncoder：5-fold CV + sigmoid shrinkage，`use_target_encoding` toggle 控制 |
| `_shared_features.py` | 特征计算共享模块（`major_to_faculty`、跨学部特征、学校档次），主管线 + 实验管线共用 |
| `held_out.py` | 时间冻结留存集：从 cases.feather 切分 year≥2024 的测试集，正例率差 <1pp |
| `school_level_mapper.py` | 学校层级映射，用于推理侧的学校等级回退逻辑 |
| `utils.py` | 模型与评估结果落盘（.ubj.gz；特征名/校准参数/学校层级回退写入 Booster 属性；评估 JSON） |
| `train.py` | 命令行入口（`--model` / `--auto_tune` / `--obj` / `--split`） |

另有 `experimental/` 子目录，独立于主管线，用于算法假设验证。详见 `src/ml/experimental/README.md`。

## 2. 数据与特征

- **目标列**：`admitted`（二分类）。
- **分类列**：`background_university`, `background_major`, `target_university`, `target_major` → 转为 pandas `category`。
  - `faculty` 被排除（不可直接作为模型特征），但可通过目标编码 (`use_target_encoding=True`) 或内化特征 (`use_internalized_features=True`) 间接利用。
- **文本列**：仅参与清洗与填补，不直接入模。
  - 同时用于样本权重判定：当所有 5 个 detail 文本列均为空时，该样本权重降为 `TEXT_EMPTY_SAMPLE_WEIGHT`（默认 0.85）。
- **计数字段**：`research_count`, `award_count`, `internship_count`, `paper_count`
  - 处理：99 分位截尾 → `log1p` 对数变换。
- **语言成绩统一**：
  - 若同时有 `toefl/ielts`：各自归一化后取 max，落地为 `language_score`；原列删除。
  - 仅存在其一时按比例归一化。
  - 训练/测试一致性：特征工程在训练集拟合（中位数/cap），测试集仅做变换。

### 2.1 内化特征（`use_internalized_features=True`）

(2026-06 从实验管线迁入) 将后处理惩罚链的底层信号注入训练特征，让 XGBoost 自行学习交互系数：

| 特征 | 来源 | 替代的惩罚层 |
|------|------|-------------|
| `major_similarity` | TF-IDF 专业相似度缓存 | 跨专业惩罚 (×0.5~1.0) |
| `cross_faculty_score` | `cross_faculty_rules.json` 查表 | 跨学部惩罚 (×0.3~0.7) |
| `is_severe_cross_faculty` | score ≤0.5 标记 | — |
| `bg_school_score` | `school_tiers.json` 学校档次（C9=1.0, 985=0.85, 211=0.65） | — |

默认关闭。开启需在 `data_loader.py` 中传 `FeatureEngineer(use_internalized_features=True)`。

### 2.2 目标编码（`use_target_encoding=True`）

(2026-06 从实验管线迁入) 对 5 个 categorical 特征做 Leave-One-Label-Out 5-fold CV 目标编码 + sigmoid shrinkage：

```
shrinkage = 1 / (1 + exp(-(n - k) / s))   # k=10, s=5
encoded  = shrinkage × category_mean + (1 - shrinkage) × global_mean
```

少于 10 样本的类别向全局均值强收缩，防止过拟合。默认关闭。

## 3. 样本权重与最近样本加权

- **文本为空降权**：所有 detail 文本列为空 → 权重 ×0.85（`TEXT_EMPTY_SAMPLE_WEIGHT`）。
- **最近样本加权**：数据集末尾 `RECENT_SAMPLE_BOOST_COUNT`（默认 10000）条样本 ×1.1（`RECENT_SAMPLE_BOOST_WEIGHT`），缓解 concept drift。
  - 仅在 `split=random` 时生效；`split=time` 时行序为地理扩张批次而非时间序列，禁用此策略。
- **稳定性**：训练阶段 `sample_weight` 转 float32，异常时回退到原始权重。

## 4. 训练与校准 (`model_trainer.py`)

### 4.1 单调约束

- 白名单内特征（GPA、语言、四段经历计数）约束 +1。
- 分类特征、内化特征不加约束（无先验单调性支撑）。

### 4.2 训练模式

- **固定参数**（默认）：`n_estimators=590, max_depth=14, lr=0.0246, ...`——Optuna 200-trial 产出的最优参数。
- **自动调参**（`--auto_tune`）：100-trial Optuna，可切换优化目标（见 §6）。

### 4.3 概率校准（isotonic, 2026-06-12）

```
校准方法：isotonic (PAV, non-parametric)
```

**为什么不是 sigmoid (Platt scaling)？**
- Platt 隐含 logit 对称假设，偏态正例率（~34%）下不一定成立。
- 校准集 ~12.3K（61.7K × 20%），远超 isotonic 安全门槛（n>1000）。
- 2026-06-12 实验：sigmoid → isotonic，全链路 ECE 0.0857→0.0792 (-7.6%)。

**流程**：`StratifiedShuffleSplit(80/20)` → 80% fit base XGBoost → 20% fit `CalibratedClassifierCV(FrozenEstimator(base), method='isotonic')`。

### 4.4 类别不平衡

自动 `scale_pos_weight = n_negative / n_positive`。

## 5. 评估与落盘 (`utils.py`)

- **模型保存**：`pre-trained_models/{model}_{timestamp}.ubj.gz`。
  - Booster 属性：`feature_names`, `calibration_params`, `level_fallback_mapping`, `feature_engineer_state`。
  - 线上加载时优先选最新 mtime 的 `.ubj.gz` 文件。
- **评估结果**：`evaluation_results/{model}_evaluation_{timestamp}.json`。
  - 含 metrics（Brier/ECE/ROC-AUC/F1）、feature_importance、model_params、data_hash、model_hash 等。
  - 含 post_train_smoke_check（模型回加载 + predict_proba 一致性校验）。

## 6. 训练入口 (`train.py`)

```bash
# 默认（固定参数 + log_loss 优化 + 随机切分）
python -m src.ml.train

# Brier Score 优化 + 时间冻结留存集
python -m src.ml.train --obj brier --split time

# 自动调参
python -m src.ml.train --auto_tune --obj brier
```

| Flag | 默认值 | 说明 |
|------|--------|------|
| `--model` | `xgboost` | 模型类型 |
| `--auto_tune` | false | 启用 Optuna 100-trial 超参搜索 |
| `--obj` | `log_loss` | Optuna 优化目标：`log_loss` 或 `brier` |
| `--split` | `random` | 切分方式：`random`（分层随机）或 `time`（year≥2024 冻结集） |
| `--data-path` | `src/ml/data/cases.feather` | 数据路径 |

- 并行限制：`LOKY_MAX_CPU_COUNT=4`。

## 7. 推理侧调整链与训练的关系

训练管线产出 base model 的概率，推理侧通过 `pipeline_config.json` 的 `adjustment_flags` 逐层叠加后处理：

```json
“adjustment_flags”: {
  “enable_gpa_penalty”: true,
  “enable_language_penalty”: true,
  “enable_cross_major_penalty”: false,    // ← 2026-06-08 关闭（E4 消融证实有害）
  “enable_cross_faculty_penalty”: true,
  “enable_professional_penalty”: true
}
```

- `enable_cross_major_penalty` 关闭后，全链路 ECE 0.1009→0.0849 (-15.9%)。
- 其余 flag 待评估。当前生产配置为 ECE 历史最优。
- 内化特征 (`use_internalized_features`) + 所有惩罚层全关可将 base ECE 降至 0.03 量级（实验已验证），但需产品确认概率变化可接受后重训模型。

## 8. 文本加成 (TF‑IDF + Logit Uplift) 训练

脚本：`scripts/train_text_tfidf.py`

### 产物
- `tfidf_vectorizer.joblib`：字符级 TF‑IDF 向量器（ngram 2-4，特征上限 20000）。
- `tfidf_centroids.npz`：四段文本（科研/获奖/实习/论文）的 L2 归一化质心。
- `text_uplift_weights.json`：非负增益权重（NNLS 拟合，回退 Ridge positive=True）。

### 运行
```bash
python scripts/train_text_tfidf.py
```

产物默认落盘至 `src/ml/pre-trained_models/`。

## 9. 常见问题

- **目标列缺失/NaN**：确认 `data_config.TARGET_COLUMN` 与数据对齐。
- **新增类别导致推理崩溃**：分类列统一转 pandas `category` 并对齐训练特征列。
- **正负样本不均衡**：`scale_pos_weight` 自动处理。
- **split=time 提示留存集索引不匹配**：`held_out_test.feather` 与 `cases.feather` 不同步，重新运行 `python -m src.ml.held_out --freeze`。

---

> **维护人**: support@demo.local
> **版本**: v3.0
> **最后更新**: 2026-06-22 — 路径迁移至 src/ml；新增 te_encoder / _shared_features / held_out；校准 sigmoid→isotonic；新增内化特征与目标编码 toggle；新增 --obj brier / --split time flags；新增 adjustment_flags 说明；移除已删除的实验模型引用
