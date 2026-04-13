# 文本背景 Logit Uplift（Text Boost）

| 项 | 说明 |
|----|------|
| 实现路径 | `src/pages/prediction/result_modifier/providers/logit_uplift/` |
| 训练脚本 | `scripts/train_text_tfidf.py` |
| 产物 | `tfidf_vectorizer.joblib`、`tfidf_centroids.npz`、`text_uplift_weights.json`（默认目录见 `machine_learning_models/pre-trained_models/`） |

本模块在**主模型基线概率**之上，根据科研/获奖/实习/论文等文本计算 **Logit 空间增量** \(\Delta\text{logit}\)，再映射回概率。与全量重训不同，增量权重在离线阶段用非负约束拟合，线上仅做向量与代数运算，**不调用大模型推理**。

## 1. 设计约定

**Uplift 语义**：估计在固定结构化特征（GPA、院校等）下，文本带来的额外信息对 log-odds 的贡献。权重非负（NNLS / Ridge positive），避免与“背景仅作加分项”的产品假设冲突。

**Logit 而非概率直接相加**：\(\text{logit}(p') = \text{logit}(p_{\text{base}}) + \Delta\)。在概率接近 0 或 1 时变化更平滑。

## 2. 运行时组件（概要）

| 组件 | 职责 |
|------|------|
| `SimilarityComputer` | 文本向量与录取子集质心的余弦相似度；可选高信号词与新颖度加权（见 `config` 与 `text_high_signal_terms.json`） |
| `_fast_entropy` | 基于字节频次的香农熵，用于检测重复或低多样性文本，调节有效经历强度 |
| `DeltaCalculator` | 由相似度、熵修正后的丰富度、经历数量及交互项线性组合得到 \(\Delta\text{logit}\)；低于门控阈值时不触发 |
| `ProbabilityApplier` | 平滑与动态封顶（与 `result_modifier/config.py` 中 `max_total_boost` 等一致），防止概率异常 |

典型公式形态（系数以训练产出为准）：

\[
\Delta\text{logit} = \beta + \sum_k w_k S'_k + \sum_k u_k \cdot S'_k \cdot \ln(1 + n_k \cdot r_k)
\]

其中 \(S'_k\) 为第 \(k\) 类经历的相似度经丰富度修正，\(r_k\) 由熵映射到 \([0,1]\)，\(n_k\) 为计数。

## 3. 离线训练（`scripts/train_text_tfidf.py`）

1. 从 `cases.feather` 提取四类文本并清洗。
2. 拟合 TF-IDF 向量器（字符级 wb，ngram 与特征上限见脚本）。
3. 在正例子集上计算质心并 L2 归一化。
4. 用基模型 logit 与标签构造残差，NNLS 拟合权重；病态时回退 Ridge（`positive=True`）。
5. 落盘向量器、质心与权重 JSON。

与线上一致性要求：列名、`feature_names` 及文本预处理规则与预测主流程对齐。

## 4. 可观测性

运行时可向日志写入加成摘要（如各段落的信号标签、熵门控是否生效）。具体字段以 `TextBoostProvider` 与 `SimilarityComputer` 实现为准。

## 5. 降级行为

向量器或权重加载失败、或输入全空时，应退化为 \(\Delta=0\)，不阻断主预测流程（实现中见具体 `try` 边界与日志级别）。

## 6. 性能与缓存

实现使用 NumPy 向量化与 `lru_cache`（同会话重复文本）减少重复计算；**不对延迟作固定 SLA 承诺**，需在目标环境自行压测。

---

相关：[ml_training_api.md](ml_training_api.md) 第 7 节、[result_modifier_api.md](result_modifier_api.md)。

维护：与 `providers/logit_uplift/` 及 `train_text_tfidf.py` 同步更新。
