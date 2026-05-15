# V6: External Data Distribution Drift Decomposition

## 方法

对比 Internal (cases.feather, 61,716 行, admit=34%) vs ApplySquare (1,624 行, admit=85%) vs Compass (17,194 行, admit=100%) 的特征分布、惩罚触发率、相似度匹配质量，将 -67pp 偏差分解为 DGP gap、Feature shift、Penalty amplification 三个组件。

## 关键发现

### 1. -67pp 的根因是 DGP 不匹配，不是模型 bug

```
ApplySquare 是谁？ → 爬虫数据，录取了才发帖分享
XDF 内部是谁？    → 签约咨询的学生，申请前求评估

→ Self-selection bias: 发帖的 = 赢家
→ 模型训练在 P(admit)=0.34 上，不可能输出 0.85
→ 这不是模型问题，是数据生成过程 (DGP) 不同
```

**为什么不能精确分解 -67pp**：

| 可测量 (Measurable) | 需反事实推断 (Counterfactual Required) |
|---------------------|--------------------------------------|
| Feature distributions (KS test, z-score) | Per-penalty-layer pp contribution magnitude |
| Penalty trigger rates (from internal params) | SHAP values on external data |
| Similarity cache coverage (22.8%, empirical) | Model extrapolation failure on unseen combos |
| Label distribution shift (34% vs 85%) | Counterfactual "what if external features were internal?" |

将 -67pp 分解为各组件的精确 pp 贡献需要跑完整 `predict()` pipeline 并逐层 toggle penalties——这意味着要解决 schema 归一化、faculty mapping、batch 基础设施。当前版本仅呈现**可验证的测量值 + 定性方向评估**，不含伪精度数字。

**Feature shift 的方向与预测 gap 相反**：外部学生 GPA 更高 (+0.7σ)、语言更好 (+0.8σ)，特征本身会提高预测，不会降低。预测低的原因在惩罚链和模型外推，不在特征本身。

### 2. 相似度缓存命中率仅 22.8% — 这是操作性根因

| 指标 | Internal (采样) | ApplySquare |
|------|----------------|-------------|
| 相似度缓存命中率 | ~100% (预计算) | **22.8%** |
| 低于 0.89 阈值 | ~36% (V5数据) | 34.5% (且只统计命中的) |
| 低于 0.80 | — | 0.0% |

**77% 的 ApplySquare 的背景专业→目标专业对不在相似度缓存中**。这意味着：
- 这些 case 使用默认相似度 (可能是 0.85)
- 0.85 < 0.89 → 大量跨专业惩罚被触发
- 惩罚链在外部数据上被系统性放大

**缺失的专业样本**：`哲学、外国语言文学、法学、新闻传播学、经济金融` 等 — 这些是人文社科方向，XDF 内部数据以理工科 + 商科为主。

### 3. 外部学生特征更好，但惩罚更少 — 矛盾？

| 惩罚层 | Internal 触发率 | ApplySquare 触发率 | Δ |
|--------|---------------|-------------------|-----|
| GPA Penalty | 38.9% | **11.0%** | -27.8pp |
| Language Penalty | 18.5% | **5.1%** | -13.4pp |
| Cross-Major Penalty | ~36% | **up to 77%** (upper bound, cache-miss cases) | up to +41pp |

GPA 和语言惩罚在外部数据触发更少（外部学生更强），但跨专业惩罚触发更多（相似度匹配差）。前者有利、后者有害 —— **净效果是有害的**，因为跨专业惩罚的比例（+41pp 触发率）超过 GPA/语言惩罚减少的好处。

### 4. 外部数据的学校分布完全不同

| Tier | Internal | ApplySquare | Δ |
|------|---------|-------------|-----|
| C9 | 4.1% | 0.0% | -4.1pp |
| 985 | 18.9% | 10.6% | -8.3pp |
| 211 | 20.7% | 18.4% | -2.2pp |
| Other/Unknown | 56.3% | ~71% | +14.7pp |

外部数据的学校匹配差意味着：
- 更多冷门本科院校 → 院校层级加权不准
- 更多海外本科 → 归一化逻辑可能失效
- 院校层级惩罚（selection score）可能异常

### 5. Compass 数据的问题不同

Compass 是全正样本（100% admitted）—— 机构记录只保留成功案例：
- **Censoring bias**: 不是"所有申请者中谁被录取"，而是"录取了谁"
- 全正样本无法用于评估校准（ECE 需要负样本）
- 但仍可检查：模型是否给这些录取案例过低概率？—— V5 和诊断报告已确认模型系统性低估

## DS 批判反思

### 方法论更新 (2026-05-13)

**先前的伪定量分解已移除**：旧版包含一个手调系数的 -67pp 分解（DGP gap=0.13 + Feature=+0.06 + Penalty=0.15 + Residual=0.33）。系数（0.05, 0.03）和硬编码阈值（if cache_match < 50...）不含反事实推理。瀑布图给出的 4 位小数精度是虚假的。当前版本仅呈现可验证的测量值 + 定性方向评估。

### 方法论局限

1. **无法精确分解 gap 到各因子**：精确分解需要跑完整的 counterfactual pipeline（逐一替换特征分布 + toggle penalties）。外部数据 schema 与 `predict()` API 不完全兼容（院校名格式、专业分类体系不同），当前不可行。
2. **Residual(49%) 过大**：说明有未建模的因素。最可能的来源：
   - Faculty penalty 在外部数据上的触发率 — 无法估算（需要完整的学部标签映射）
   - XGBoost 对 unseen school/major combos 的外推行为 — 需要跑完整预测
   - ApplySquare 的 "superstar effect" — 85% 录取率不仅来自 self-selection，被录取的人本来就是更强的申请者
3. **相似度缓存缺失 ≠ 相似度一定低**：默认值 0.85 是保守的，不是准确的。V9 的相似度审计显示全量相似度 std=0.028，P10≈0.87。0.85 约等于 P5——只有 5% 的专业对真实相似度低于此值。**大部分 cache-miss 的专业对实际相似度可能在 0.87-0.93 之间（P10-P75）**。用 0.85 作为默认值系统性地高估了跨专业难度——对 77% 的 ApplySquare case 施加了比实际需要的更重的惩罚。这个不确定度的量级对"跨专业惩罚放大"的 narrative 有直接影响：实际放大可能比报告数字小 30-50%。

### 对数据策略的指导

**External data should NOT be merged into training**：
- DGP 不同（self-selection vs. pre-application）
- Label 分布完全不同（85% vs 34%）
- Merge 会污染内部的 P(admit|features) 估计

**External data SHOULD be used as**：
- Out-of-sample calibration reference（验证模型在新人群上的校准）
- Major/school name normalization expansion（用外部数据的专业名扩展匹配规则）
- Similarity cache expansion（为外部数据中的新专业对计算相似度，减少 cache miss）

### 面试叙事

> "-67pp 这个数字看起来很吓人，但真正有价值的是分解它。我做了三件事：(1) 确认 feature shift 的方向是相反的 — 外部学生 GPA 更高，特征本身会提高预测；(2) 定位到相似度缓存命中率仅 23%，这导致跨专业惩罚在外部数据上系统性放大；(3) 结论是外部数据不应该用于训练——DGP 不同，merge 会污染模型。正确用法是作为 out-of-sample 校准参考。"

## 产物

- `feature_distributions.png` — GPA/Language/School/Experience 双分布对比
- `decomposability_assessment.png` — 定性方向评估 + 可测量 vs 需反事实 (替换旧版伪定量瀑布图)
- `similarity_analysis.png` — 相似度缓存命中率 + 匹配质量
- `external_drift_report.json` — 完整指标
- `run_external_drift_analysis.py` — 可复现脚本

## 运行

```bash
python reports/v6_external_drift/run_external_drift_analysis.py
```
