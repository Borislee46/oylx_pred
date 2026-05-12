# 决策边界解释器 — 设计文档

## 一句话定位

当前 trace 解释了 **调整链**（post-processing chain），没有解释 **模型本身**（XGBoost 为什么给出这个 base prob）。这个模块补上模型内部解释，让 trace 从"调整链可视化"升级为"模型决策全链路透明"。

---

## 1. 为什么 SHAP 不是答案，树结构才是

SHAP 的代价：
- 加性假设强加于树模型（树决策不是加性的）
- 需要背景数据采样 → 采样质量影响解释质量
- 计算开销大（TreeSHAP 比 SHAP 快，但仍需 O(n_trees × n_features × n_samples)）
- 输出的 feature contribution 对非技术用户无意义

XGBoost 的树结构天然提供三层解释，**不引入任何外部依赖**：

| 你拿到的 | 你回答的问题 | XGBoost API |
|---------|------------|-------------|
| Per-tree leaf weights | 模型有多确定？（ensemble variance） | `pred_leaf=True` |
| Split points across trees | 差多少能翻盘？（decision boundary distance） | `trees_to_dataframe()` |
| Leaf → training cohort | 模型眼里谁跟你像？（leaf cohort stats） | 训练时预计算 |

三个加起来不到 200 行 Python，零额外依赖。

---

## 2. 三个组件

### 2.1 Ensemble Uncertainty — 树间方差

**原理**

XGBoost 是 B 棵树的加性模型：

```
raw_margin = sum(leaf_weight_b) + base_score   for b in 1..B
prob = sigmoid(raw_margin)
```

每棵树产出一个叶子权重。树与树之间的方差直接反映"模型对自己的判断有多不一致"。不需要 bootstrap、不需要 Bayesian——这是 ensemble 模型自带的 epistemic uncertainty 近似。

**实现**

```
1. booster.predict(dmatrix, pred_leaf=True) → (1, B) leaf index matrix
2. 用 get_dump(json) 解析每棵树，构建 (tree_id, leaf_id) → weight 字典
3. per_tree_contributions = [weight[(b, leaf_b)] for b in 1..B]
4. mean = sum(per_tree_contributions) + base_score
5. std = np.std(per_tree_contributions)
6. 在 margin 空间做 ±1σ / ±2σ，过 sigmoid 映射回概率空间
```

**展示形式**

```
Ensemble 一致性
├── Raw margin: -0.42 ± 0.15
├── 概率范围:  34% [22% — 48%] (±1σ)
│   ████████████░░░░░░░░░░░░  (22%-48%)
└── 300 棵树中，127 棵倾向录取 (weight > 0)，173 棵倾向拒录
```

**DS 面试的叙事点**

> "我不用 SHAP 做不确定性，因为 XGBoost 本身就是 300 个弱学习器的投票——树间权重方差是天然的 uncertainty proxy。这比 Monte Carlo dropout 或 bootstrap ensemble 更直接，因为它用的是模型训练过程中已经存在的信息，不需要额外采样。"

---

### 2.2 Split Proximity — 关键分裂点距离

**原理**

每棵树的每个内部节点是一个 if-else 分裂。对于 case 的每个 numeric feature，所有树中都有大量以此 feature 为分裂条件的节点。有些分裂点在当前值上方（case 没跨过去），有些在下方（已经跨过去了）。距离最近的上方分裂点就是"最小可行提升"。

**实现**

```
1. booster.trees_to_dataframe() → 所有节点的 (Feature, Split) 表
2. 筛选 numeric features（gpa, language_score, research_count, award_count,
   internship_count, paper_count）
3. 对每个 feature：
   - above_splits = [split for split in all_splits if split > current_value]
   - below_splits = [split for split in all_splits if split < current_value]
   - nearest_above = min(above_splits) if above_splits else None
   - nearest_below = max(below_splits) if below_splits else None
   - trees_affected = count of trees where a split on this feature exists within
     0.5σ of current_value
4. 排序：按"跨过上方分裂点能影响的树的数量"降序
```

**展示形式**

```
关键分裂点距离 (300 trees)
┌──────────┬────────┬──────────────┬──────────┬────────────┐
│ Feature  │ 当前值  │ 最近上方分裂  │ 距离     │ 影响树数    │
├──────────┼────────┼──────────────┼──────────┼────────────┤
│ GPA      │ 3.20   │ 3.35         │ +0.15    │ 47 trees   │
│ 语言     │ 95     │ 102          │ +7       │ 23 trees   │
│ 实习     │ 1.1    │ —            │ 已过所有  │ —          │
│ 科研     │ 0.7    │ 1.4          │ +0.7     │ 12 trees   │
└──────────┴────────┴──────────────┴──────────┴────────────┘

★ 最小可行提升: GPA 3.20 → 3.35 (+0.15)
  跨过 47 棵树的正向分裂 → 预估 margin 提升 ~0.18 → 概率 34% → ~48%
```

此处"预估 margin 提升"可以精确计算：把对应当前值在 47 棵树中替换后的 leaf weight 加总即可——不是近似，是 exact counterfactual。

**DS 面试的叙事点**

> "树模型的可解释性不在于 feature importance，在于分裂点。每个分裂点都是一个可操作的阈值——你不需要笼统地说'提高 GPA'，你可以精确地说'你的 GPA 需要从 3.2 提到 3.35，因为 47 棵树在这个位置设了分裂边界'。这是 XGBoost 独有的优势，深度模型做不到。"

---

### 2.3 Leaf Cohort Statistics — 叶群统计

**原理**

两笔申请落在同一片叶子 → 模型用完全相同的规则链在判断它们 → 它们在模型眼里是"相似的"。这比欧氏距离相似、cosine 相似更本质——它是模型自己定义的相似。

**实现（离线预处理）**

训练完成后跑一次：

```python
# 对全量训练样本
leaf_indices = booster.predict(train_dmatrix, pred_leaf=True)  # (N, B)

# 构建 leaf signature → sample_ids 索引
from collections import defaultdict
leaf_cohort = defaultdict(list)
for sample_id in range(N):
    signature = tuple(leaf_indices[sample_id, :])  # or hash
    leaf_cohort[signature].append(sample_id)

# 计算每个 cohort 的统计
cohort_stats = {}
for sig, ids in leaf_cohort.items():
    cohort_stats[sig] = {
        "n": len(ids),
        "admit_rate": train_labels[ids].mean(),
        "gpa_median": train_gpa[ids].median(),
        "lang_median": train_lang[ids].median(),
        # ...
    }
```

**预测时**

```python
case_leaf_indices = booster.predict(case_dmatrix, pred_leaf=True)  # (1, B)
case_signature = tuple(case_leaf_indices[0, :])
# 精确匹配可能为空 → fuzzy match: top-k most similar leaf signatures
cohort = cohort_stats.get(case_signature) or fuzzy_match(case_signature, k=5)
```

**展示形式**

```
模型眼中的相似案例 (leaf cohort, n=847)
├── 历史录取率: 19%（你的预测: 34%）
│   — 你的 case 高于叶群均值，因为在 text boost 等特征上优于叶群中位数
├── 录取者 GPA 中位数: 3.50（你 3.20, P35）
├── 录取者 语言 中位数: 100（你 95, P42）
└── 录取者 实习 中位数: 1 段（你 1 段, P50）
```

**DS 面试的叙事点**

> "leaf cohort 是树模型自带的'相似案例检索'——两笔数据被同一套 if-else 规则判断，它们就在模型意义下相似。这和 k-NN 的欧氏距离不同，更像是'同班同学的录取结果'。对于留学顾问来说，这个类比直觉：你的 profile 在模型分类路径上，跟往年这 847 个申请人走的是同一条路。"

---

## 3. 技术实现总览

### 3.1 依赖

- `xgboost.Booster.predict(pred_leaf=True)` — 叶子索引
- `xgboost.Booster.trees_to_dataframe()` — 树结构（分裂点）
- `xgboost.Booster.get_dump(dump_format='json')` — 叶子权重 JSON
- `json` 标准库解析
- `numpy` 做统计

**外部依赖数量: 0**

### 3.2 数据流

```
用户提交表单
    │
    ▼
PredictionModel.predict_batch()
    │
    ├── booster.predict(dmatrix) → 校准后概率 (现有)
    │
    ├── [新] booster.predict(dmatrix, pred_leaf=True) → leaf indices (N, B)
    │     ├── → per-tree weight lookup → uncertainty
    │     └── → leaf signature → cohort lookup → 历史统计
    │
    └── [新] booster.trees_to_dataframe() → 分裂点表
          └── → split proximity per feature → counterfactual
    │
    ▼
结果写入 result dict
    │
    ▼
boundary_explainer.py 渲染 HTML → st.html()
```

### 3.3 需要新增的文件

| 文件 | 职责 |
|------|------|
| `boundary_explainer.py` | 计算 + 渲染入口 |
| `boundary_assets.py` | CSS + 文案常量 |
| `leaf_cohort_stats.parquet` | 训练时预计算，预测时加载（~2MB） |

### 3.4 性能

| 操作 | 耗时 | 频率 |
|------|------|------|
| `pred_leaf=True` | ~5ms（复用已有 DMatrix） | 每次预测 |
| `trees_to_dataframe()` | ~50ms | 缓存，只算一次 |
| JSON dump 解析 | ~100ms | 缓存，只算一次 |
| cohort lookup | ~1ms（dict key lookup） | 每次预测 |
| split proximity | ~5ms（遍历已缓存的 DataFrame） | 每次预测 |

总计**单次预测额外 ~10ms**，不影响用户体验。

### 3.5 与现有 trace_display 的关系

不是替代，是补充。现有 trace 继续做调整链可视化，`boundary_explainer` 负责 XGBoost 内部解释。两者在 trace container 中分上下两个 section：

```
┌─ Trace Container ──────────────────────────────┐
│                                                 │
│  [boundary_explainer]  ← 新：模型内部解释        │
│  ├── uncertainty                                │
│  ├── split proximity                            │
│  └── leaf cohort                                │
│                                                 │
│  [trace_display]  ← 现有：调整链可视化           │
│  ├── waterfall                                  │
│  ├── counterfactual                             │
│  └── calibration                                │
│                                                 │
└─────────────────────────────────────────────────┘
```

---

## 4. 面试叙事指南

### 30 秒版本（面试官问"你这个 trace 是怎么回事"）

> "trace 分两层。上层是 XGBoost 内部的决策边界分析——我从叶子权重算不确定性、从分裂点算最小可行提升、从叶群算历史参照。下层是 domain-specific 的调整链。两层加在一起，相当于把模型从输入到输出的每一步都透明化了。"

### 深入版本（面试官追问"为什么不用 SHAP"）

> "SHAP 有价值，但三件事它做不好：第一，SHAP 的加性假设对树模型来说是外挂的——树决策本质是 if-else 路径，不是特征贡献的加权和。第二，SHAP 不能告诉你'你的 GPA 差 0.15 就能跨过 47 棵树的分裂边界'——因为 SHAP 不暴露分裂点。第三，SHAP 需要背景采样，采样分布一变，解释就变，不可复现。我用的是树结构自带的信息——叶子权重、分裂点、leaf signature——这些在模型训练完那一刻就固定了，100% 可复现。"

### 如果面试官问"leaf cohort 会不会太稀疏"

> "会的，精确 signature 匹配的覆盖率不高。所以不做精确匹配——做 fuzzy match。把 B 棵树的 leaf signature 做降维，比如用叶子的 admit rate 替代叶子 ID，变成一个 B 维的 rate vector，然后 cosine similarity 找 top-k 邻居。这样几乎每个 case 都能找到参照。这也更合理——'模型用了相似但非完全相同的规则路径'比'完全相同的规则路径'更有统计意义。"

---

## 5. 实现优先级

| 优先级 | 组件 | 理由 |
|--------|------|------|
| P0 | Uncertainty | 代码量最小（~40行），收益最直观，面试最容易讲 |
| P1 | Split proximity | 需要解析树结构（~80行），但直接产生 actionable insight |
| P2 | Leaf cohort | 需要离线预处理，但面试叙事最强——"模型眼中的你" |

建议第一轮做 P0 + P1（合并实现，因为共用树结构解析），第二轮补 P2。
