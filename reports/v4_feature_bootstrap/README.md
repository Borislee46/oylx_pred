# V4: 特征重要性 Bootstrap 分析

## 方法

对训练集做 1000 轮 bootstrap 采样（每轮 80%），训练简化版 XGBoost（150 trees, depth=6），记录每轮特征重要性排名。用 95% CI 评估排名稳定性。

## 结果

| 特征 | Importance | Rank (95% CI) | 稳定性 |
|------|-----------|---------------|--------|
| target_major | 0.332 ±0.006 | **[1.0, 1.0]** | 完美稳定 |
| background_university | 0.303 ±0.005 | **[2.0, 2.0]** | 完美稳定 |
| background_major | 0.108 ±0.004 | **[3.0, 3.0]** | 完美稳定 |
| target_university | 0.066 ±0.004 | [4.0, 5.0] | 与 gpa 重叠 |
| gpa | 0.057 ±0.004 | [4.0, 5.0] | 与 target_univ 重叠 |
| award_count | 0.037 ±0.004 | [6.0, 7.0] | 与 language 重叠 |
| language_score | 0.033 ±0.004 | [6.0, 8.0] | 最不稳定 |
| research_count | 0.027 ±0.003 | [7.0, 8.0] | 稳定 |
| paper_count | 0.021 ±0.002 | [9.0, 10.0] | 与 internship 重叠 |
| internship_count | 0.017 ±0.001 | [9.0, 10.0] | 与 paper 重叠 |

## 发现

### 1. Top-3 特征 1000 轮中排名从未改变

target_major、background_university、background_major 的排名完全锁定，0 次变动。这三个特征是系统的核心骨架，砍任何一个都会严重损伤排序质量。

### 2. 三对特征排名不可区分

- **target_university ↔ gpa** (100% 重叠) — 不能说目标院校比 GPA 更重要
- **paper_count ↔ internship_count** (98% 重叠) — 论文和实习几乎等价，**砍特征时可以二选一**
- **award_count ↔ language_score** (50% 重叠) — 部分重叠，勉强可区分

### 3. language_score 是最不稳定的特征

std=0.63，95% CI 跨度 3 个排名位次。这可能因为 language_score 是 TOEFL/IELTS 归一化后的合成特征，携带的信息量不如原始两个特征独立存在时多。

### 4. Bootstrap 与 production 的差异

简化模型（150 trees, depth=6）比生产模型（375 trees, depth=10）更集中地将重要性分配给 top 分类特征（target_major +0.09, background_university +0.16）。深层模型会分散重要性到更多特征——但排名趋势一致，验证了特征工程方向的正确性。

## 面试叙事

> "我做了 bootstrap 特征重要性分析——1000 轮采样训练，target_major 的 top-1 地位非常稳定（95% CI rank 1-3），但 paper_count 和 internship_count 的排名 CI 几乎完全重叠，不能严格说谁更重要。这对 feature selection 有直接指导——如果要砍特征，论文和实习可以二选一。"

## DS 批判反思 (2026-05-13)

Bootstrap 给了特征排名的置信区间——这是标准做法，正确。但有两个前提未被质疑：

1. **Gain-based importance ≠ causal importance**。XGBoost 的 gain-based importance 反映的是"特征对 loss reduction 的贡献"，不是"特征对录取的因果重要性"。target_major 排名第一可能是因为它在树结构中获得了最多的分裂机会（类别多、信息增益大），而不是因为它对录取的因果关系最强。如果要回答"应该优先提升 GPA 还是攒实习"（TODO-4），importance 不是答案——需要 causal inference。

2. **Bootstrap stability ≠ robustness to model specification**。Bootstrap 只重采样了训练数据，没变超参数。如果换 max_depth=6/8/10 或 learning_rate=0.03/0.05/0.1，特征排名还稳吗？当前报告不能回答这个问题。Bootstrap 测的是"给定超参下数据采样的稳定性"，不是"给定数据下超参选择的敏感性"。后者对于判断"这个排名在生产中是否可靠"同样重要。

另外 language_score 排名最不稳定（std=0.63）呼应了 DEC-013 的观察：TOEFL 和 IELTS 合并为单一特征可能丢了信号。如果未来做特征工程重构，language_score 值得优先重考虑。

## 产物

- `feature_rank_box.png` — 排名箱线图 + 95% CI
- `feature_importance_box.png` — 量值 + error bar
- `rank_correlation.png` — 特征间排名相关性
- `bootstrap_importance.json` — 完整数据
- `run_bootstrap_importance.py` — 可复现脚本
