# 预测质量诊断全景报告

**日期**: 2026-05-13（V5/V6/V7 均于同日完成，深度诊断三部曲）
**数据来源**: cases.feather (61,716 行), ApplySquare (1,624 行), Compass (17K 行)
**方法**: XGBoost 校准报告 + Corner Case 边界测试 + Rigor 单调性测试 + AI 专家盲评 + 外部数据验证 + 稀疏度压测 + **V5 联合惩罚效应** + **V6 外部漂移分解** + **V7 预测失败模式**

---

## 一、诊断全景

| # | 诊断项 | 规模 | 结论 |
|---|--------|------|------|
| 1 | ECE 校准报告 | 500 样本 | ECE=0.1155 严重失校准，系统性低估 -9pp |
| 2 | Corner Case 边界测试 | 22 cases | 全通过，无崩溃/NaN，概率范围合理 |
| 3 | Rigor 单调性/稳定性测试 | 32 cases | 单调性/稳定性/阈值/TOEFL等价性全通过 |
| 4 | AI 专家盲评对比 | 40 样本 | MAE=0.18, Pearson r=0.44, 35%分歧>0.2 |
| 5 | 外部数据验证 | ApplySquare 1,624 | 偏差 -67pp — DGP mismatch + penalty amplification |
| 6 | 稀疏度 + Corner Case 压测 | 100 样本 + LLM×10 | 模型输出~0.70天花板, LLM给~0.35, Δ=+0.35 |
| **7** | **联合惩罚效应 (V5)** | **12,344 样本 ECE ablation** | **Faculty penalty 是头号校准杀手，关掉它 ECE 从 0.15→0.11** |
| **8** | **外部漂移分解 (V6)** | **ApplySquare 1,624 全量** | **相似度缓存命中仅 23%，跨专业惩罚触发率 ~77%** |
| **9** | **预测失败模式 (V7)** | **61,716 全量** | **缺失 GPA 与更高录取率相关（+13pp，correlational），DEC-010 需重审** |

---

## 二、稀疏度全景

(bg_uni × bg_major × target_uni × target_major) 52,327 个唯一组合：

| 档位 | 组合数 | 占比 |
|------|--------|------|
| 1-4 例 | 52,001 | **99.4%** |
| 5-9 例 | 296 | 0.6% |
| 10-29 例 | 29 | 0.1% |
| 30+ 例 | 1 | 0.0% |

**核心事实**：整个模型在极端稀疏数据上运行。99.4%的组合只有 ≤4 个历史样本，只有1个组合有30+样本。模型输出本质上是对少量样本的平滑外推，而非统计推断。

---

## 三、校准报告 (ECE)

> ⚠ **注意**: 以下 ECE=0.1155 是全链路（XGBoost + 5层调整链 + 文本提升）的校准误差。Base model (Platt scaling only) 的 ECE=0.0263 近乎完美，详见 V1。**问题在调整链，不在 XGBoost。**

端到端校准（500样本，stratified sampling），覆盖 XGBoost raw output → GPA惩罚 → 语言惩罚 → 跨专业 → 跨学部 → 职业学位 → 文本提升全链路。

| 指标 | 值 | 判定 |
|------|-----|------|
| ECE | 0.1155 | >0.10 严重失校准 |
| Brier Score | 0.176 | — |
| 预测均值 | 0.2416 | vs 实际录取率 0.3312 |
| 系统性偏差 | -8.96pp | 模型系统性低估 |
| 最高输出概率 | 0.72 | 天花板效应 |

**分层校准偏差**：

| 子群 | 预测均值 | 实际录取率 | 偏差 |
|------|---------|-----------|------|
| C9 学生 | 0.43 | 0.61 | -18pp |
| 985 学生 | 0.29 | 0.36 | -7pp |
| 211/双非 | 0.18 | 0.24 | -6pp |
| GPA 3.8+ | 0.36 | 0.49 | -13pp |
| GPA 3.0-3.5 | 0.19 | 0.28 | -9pp |

C9 高GPA学生被低估最严重——强者看不到"稳了"的信号。

> ECE > 0.10 为"严重失校准"是 heuristic 阈值（参考文献: Guo et al. 2017, "On Calibration of Modern Neural Networks"）。教育预测场景下尚无领域特定标准。

**可靠性图**：bins [0.8-0.9) 和 [0.9-1.0] 样本数为 0——调整链后模型从不输出高于 0.72 的概率。

**根因**：GPA 二次惩罚(0.15×z²) + 语言阶梯惩罚 + 跨专业×0.5 + 跨学部×0.3 + 职业学位惩罚，即使有衰减仲裁(0.85/layer, 总上限70%)，多层乘法叠加仍过于激进。

> **ECE 数值说明**: 本报告中出现三个 ECE 值，对应不同 pipeline 和样本:
> - V1 ECE=0.0263: Base model (Platt scaling only, 不经过调整链)
> - 本报告 ECE=0.1155: 全链路 (500 样本分层采样, 真实 `predict()` API)
> - V5 ECE=0.1497: 仿真全链路 (12,344 样本, 简化版 CrossMajor + 无文本提升)
> 
> 三者方向一致（调整链恶化校准），但绝对值不可直接比较。V5 的相对变化（ΔECE per layer removed）是调参指导的核心依据。

---

## 三-B、联合惩罚效应分析 (V5, 12,344 样本)

> 详见 `reports/v5_joint_penalty_effect/`

逐层关掉测量 ECE/Brier 变化——从 V2 的"排序影响"升级到"校准影响"。

### 逐层 ECE Ablation

| Layer Removed | ECE | Δ vs Full Chain | 对校准的影响 |
|--------------|-----|-----------------|-------------|
| Full Chain (5层全开) | 0.1497 | — | baseline |
| Raw XGBoost (全关) | 0.1179 | +0.0318 | **调整链整体恶化校准** |
| **Remove Faculty** | **0.1112** | **+0.0385** | **头号杀手——比全关还好** |
| Remove Language | 0.1476 | +0.0021 | 影响很小 |
| Remove GPA | 0.1483 | +0.0014 | 影响很小 |
| Remove CrossMajor | 0.1513 | -0.0016 | 唯一有正面贡献的层 |
| Remove Professional | 0.1497 | +0.0001 | 几乎无影响(仅24 cases) |

> **注**: 各层 ΔECE 不满足可加性。Faculty 单独关掉改善 ECE +0.0385，但全关只改善 +0.0318——因为 decay 机制使层间存在非线性交互：GPA/Language 独立看有轻微校准破坏，与 Faculty 的改善相互抵消。单层 ΔECE 解读为"该方向上的独立效应"，不可直接求和。

### 核心发现

**1. Faculty penalty (×0.3) 是校准的头号杀手**：单层贡献的 ECE 恶化（+0.0385）超过其他四层之和。关掉它后 ECE=0.1112，甚至比 Raw XGBoost（0.1179）更好。Faculty penalty 是硬编码固定值，不看证据、不衰减——2413 case（19.5%）被触发，penalty ratio 固定 0.7。

**2. 37.2% 的 case 触达 70% 惩罚上限**：4590 个 case 的 total_penalty_ratio = 0.7。天花板原本是"安全阀"，但对 37% 用户是常态。（2 层即可触顶：decay=0.85 下，两个 0.7 penalty → 0.7 + 0.7×0.85 = 1.295，cap at 0.7 → 触顶。触顶 case 平均仅 2.2 个活跃惩罚层——不是只有 4-5 层才触顶。）

**3. ECE 最差的是 2-3 层惩罚的 case**：2层 ECE=0.21，3层 ECE=0.22。4层反而略好（0.20）——衰减机制在 4 层以上开始起效，但在 2-3 层时衰减不够。

**4. 惩罚对不同层级学生的方向不同**：

| Tier | Mean Penalty Ratio | Mean N Penalties | Bias |
|------|-------------------|------------------|------|
| C9 | **0.20** | 0.98 | -0.04 |
| 985 | 0.29 | 1.16 | -0.06 |
| 211 | 0.28 | 1.10 | -0.08 |
| Other | **0.40** | 1.47 | -0.10 |

C9 学生被惩罚得最少（ratio 0.20），双非/其他被惩罚得最多（ratio 0.40）。**惩罚对弱者的伤害大于强者**——与之前"C9 -18pp"的表观方向相反。解释：C9 的 base_prob 更高，即使 penalty ratio 低，gap 来自模型本身高估；Other 的 base_prob 就低，惩罚进一步压低，gap 来自惩罚过度。

*注：V5/V6 的 school tier 分类基于院校名称关键词匹配，非生产环境 school_level_service。C9 分类可靠（精确名称匹配），但 985/211 边界可能有误分类。*

**5. CrossMajor 是设计典范**：五层中唯一使用 empirical Bayes evidence adjustment 的层（生产环境），也是唯一对校准有正面贡献的层（关掉它 ECE 恶化）。*注：V5 仿真中的 CrossMajor 使用简化版（无 evidence adjustment），即便如此仍对校准有正面贡献（Δ=-0.0016）。生产环境完整版（含 empirical Bayes shrinkage）效果预期更优。其他层应向 CrossMajor 的数据驱动设计学习。*

---

## 四、Corner Case 边界测试 (22 cases)

全通过。覆盖维度：

- GPA 边界：4.0 PKU、2.0下限、1.5超低
- 语言边界：IELTS 9.0/5.0/5.5阶梯
- 极端组合：PKU 4.0+IELTS9.0→珠海学院、双非2.0+IELTS5.0→HKU
- 跨专业/跨学部：English→CS、Nursing→CS
- 冷门学校/专业：职业技术学院、音乐学→Data Science
- 经历边界：0经历 vs max经历+文本
- 多目标排序：HKU < CityU 一致

无崩溃、无 NaN、概率范围 [0, 1]。

---

## 五、Rigor 单调性/稳定性测试 (32 cases)

全通过。覆盖维度：

- GPA 2.0→4.0 全量程单调性扫描（PKU + 南开两条曲线）
- 语言 IELTS 5.0→9.0 全量程单调性扫描
- 经历数量 0→3 单调性扫描
- 确定性复现性（同输入×5 → 完全一致）
- 阈值边界：GPA=2.0 临界、语言 minimum 临界
- TOEFL 100 ≈ IELTS 7.0 等效性校验（diff < 0.30）
- 无悬崖：GPA+0.01 不可导致 >0.05 概率跳变
- 学校层级排序：各 GPA 水平下 C9 ≥ 985
- 多目标查询无交叉污染
- 长文本经历 / 空经历不崩溃
- 概率永远在 [0,1]、无 NaN、结果结构完整

---

## 六、AI 专家盲评对比 (40 样本)

DeepSeek V4-flash 独立评估，对比 XGBoost 调整后概率。

| 指标 | 值 | 判定 |
|------|-----|------|
| MAE | 0.18 | 可接受 |
| Bias (model - AI) | +0.003 | 均值基本一致 |
| Pearson r | 0.44 | 排序一致性偏弱 |
| Spearman ρ | — | — |
| 分歧率 (>0.2) | 35% | 特定场景实质分歧 |

**分层偏差**：

| 分段 | MAE | Bias |
|------|-----|------|
| GPA 3.8+ | 0.22 | -0.08 (模型偏低) |
| GPA <3.0 | 0.21 | +0.05 (模型偏高) |
| C9 | 0.19 | -0.06 |
| 211/双非 | 0.17 | +0.04 |

模型对强学生偏保守，对弱学生偏乐观——与校准报告结论一致。

---

## 七、稀疏度压测 (100 样本 + LLM×10)

### 7.1 预测压测 (Step 2)

| 指标 | 值 |
|------|-----|
| 压测样本 | 100 (60 very_sparse + 25 sparse + 15 low) |
| 预测成功 | 94 |
| 预测失败 | 6 (6%) |
| 高概率(>0.7)在稀疏数据 | 4 个 |
| 极低概率(<0.02) | 0 个 |

**6 个预测失败的 case 共性**：全部无 GPA 数据 + 冷门本科院校（海外本科/独立学院/专科）。

### 7.2 LLM 盲评对比 (Step 3)

从异常 case 中选取 10 个进行 LLM 盲评（使用修复后的语言分数转换）：

| 背景 | 目标 | n | GPA | 语言 | Model | LLM | Δ |
|------|------|---|-----|------|-------|-----|------|
| 北航 CS | 港科大 大数据 | 6 | 3.60 | IELTS 7.5 | 0.703 | 0.75 | **-0.05** ✓ |
| 西南财经 金融 | 港城大 金融 | 13 | 3.25 | IELTS 7.0 | 0.714 | 0.55 | +0.16 |
| 大连理工 CS | 港科大 大数据 | 10 | 3.56 | IELTS 6.5 | **0.701** | 0.35 | **+0.35** |
| 西南财经 金融 | 港城大 会计 | 1 | 3.23 | IELTS 6.5 | **0.701** | 0.35 | **+0.35** |
| 奥克兰大学 | 港城大 GBM | 1 | 3.00 | IELTS 6.5 | FAILED | 0.60 | — |
| 大连理工 CS → NTU AI | | 7 | 3.00 | IELTS 7.9 | FAILED | 0.15 | — |
| 同济 土木 → NUS 土木 | | 11 | 3.00 | IELTS 6.5 | FAILED | 0.20 | — |

### 7.3 稀疏度 vs 全量对比

| 指标 | 40样本(全量) | 10样本(稀疏) |
|------|-------------|-------------|
| MAE | 0.18 | ~0.30 |
| Bias (model - AI) | +0.003 | **+0.20** |
| 分歧率 (>0.2) | 35% | 40% |

模型在稀疏区域显著比 LLM 乐观。但 LLM 自评置信度全部为 medium/high（从不给 low），实际准确性也存疑。

### 7.4 核心模式

**模型天花板效应**：稀疏组合上模型统一输出 ~0.70，无论学生实际竞争力如何。北航 CS 3.6 IELTS7.5 和西南财经 3.23 IELTS6.5 申会计都给 0.70。

**LLM 区分度更好**：好学生给 0.75，一般学生给 0.35，能拉开差距。

---

## 八、外部数据验证 + 漂移分解 (V6)

> 详见 `reports/v6_external_drift/`

### ApplySquare (1,624 行，85.4% 录取)

| 指标 | 值 |
|------|-----|
| 预测均值 | 0.17 |
| 实际录取率 | 0.84 |
| 偏差 | **-0.6734 (-67pp)** |

### -67pp 根因评估 (V6, 定性方向)

**可测量的事实**:
- Feature shift 方向与预测 gap 相反：外部 GPA +0.7σ, Language +0.8σ → 特征只会提高预测
- DGP 不匹配：Internal 34% vs ApplySquare 85% admitted → 不同数据生成过程
- 相似度缓存命中率 22.8% → ~77% 的 case 触发跨专业惩罚（默认相似度 0.85 < 0.89 阈值）

**不可精确分解**: 将 -67pp 分解为各组件的 pp 贡献需要反事实推理（跑完整 `predict()` pipeline 并逐层 toggle penalties）。外部数据 schema 与 `predict()` API 不兼容，当前无法做到。旧版的伪定量分解（DGP=0.13 + Feature=+0.06 + Penalty=0.15 + Residual=0.33）已移除——系数来自手调，不含反事实推理。

**操作性结论**: 外部数据不合并入训练集。用作 out-of-sample 校准参考。扩展相似度缓存覆盖外部专业。

### 关键发现

**1. Feature shift 方向与预测 gap 相反**：外部学生 GPA 更高(+0.7σ)、语言更好(+0.8σ)——特征本身会提高预测，不会降低。-67pp 的原因在惩罚链和模型外推，不在特征。

**2. 相似度缓存命中率仅 22.8%**：371/1,624 对外部 case 在缓存中。77% 的 case 使用默认相似度 (0.85) → 0.85 < 0.89 阈值 → 大量跨专业惩罚被触发。缺失专业主要为人文社科（哲学、法学、新闻传播学等）—— XDF 内部数据以理工科+商科为主。

**3. GPA/语言惩罚在外部数据触发更少**（外部学生更强），但跨专业惩罚触发更多（相似度匹配差）。净效果有害——跨专业惩罚放大 (+41pp 触发率) 超过 GPA/语言惩罚减少的好处 (-28pp)。

### Compass (17K 全正样本)

数据完整性太低（大量"有效组合为空"），但成功的预测进一步证实低估模式。

---

## 九、LLM 盲评的已知局限

1. **语言分数格式**：LLM 对归一化分数理解不稳定（已修复 `blind_eval_agent.py` 中 prompt 转换：`0.72 → IELTS 6.5`）
2. **置信度偏高**：LLM 从不给 low confidence 自评，实际判断准确性待验证
3. **参考价值**：LLM 可能比业务专家更客观（业务可能没申过某些冷门学校）
4. **Bias=+0.20 在稀疏区域**：到底是模型高估还是 LLM 低估？需要更多交叉验证

---

## 十、测试覆盖全景 (67 tests)

| 文件 | 数量 | 覆盖维度 |
|------|------|---------|
| test_calibration_report.py | 1 | ECE + Brier + 可靠性图 + 分层校准 |
| test_corner_cases.py | 22 | GPA/语言边界、极端组合、跨专业/学院、冷门学校/专业、经历边界 |
| test_prediction_rigor.py | 32 | 单调性扫描、稳定性、阈值边界、TOEFL/IELTS等价、无悬崖、学校层级 |
| test_external_data_validation.py | 3 | Compass全正样本、ApplySquare校准、分布漂移 |
| test_ai_blind_eval.py | 1 | AI专家 vs 模型 MAE/Bias/相关性/分歧分析 |
| test_model_outputs.py | 2 | 标准高低分概率范围 (已有) |
| test_sparsity_stress.py | 1 | 稀疏度扫描 + Corner Case压测 + LLM盲评对比 |
| V5 Joint Penalty Effect | 1 | ECE per layer ablation + penalty count + tier stratification + co-occurrence |
| V6 External Drift Assessment | 1 | Feature shift + penalty trigger direction + cache coverage + DGP mismatch |
| V7 Prediction Failure Patterns | 1 | Failure taxonomy + missing-as-signal (correlational) + DEC-010 review |
| **总计** | **65** | |

---

## 十一、优化方向

本质认知：**这不是算法问题，是数据问题**。99.4%的组合只有≤4个样本，任何统计模型都无法在小样本上做可靠推断。优化思路不是修模型，而是**承认不知道，管理不确定性**。

V5 联合惩罚效应分析揭示了一个更具体的问题：**5层惩罚链中，Faculty penalty 是头号校准破坏者**，且惩罚对弱者的伤害大于强者（与之前印象相反）。

### 🔴 P0 — 跨学部惩罚重构（最大ROI）

Faculty penalty 当前是硬编码 ×0.3，无证据调整。关掉它 ECE 从 0.15 降到 0.11——单层改善超过其他四层之和。

- 改为 evidence-adjusted：像 CrossMajor 那样用 empirical Bayes shrinkage
- 或至少降低 penalty ratio：×0.3 → ×0.5~×0.6
- 对"学部相邻但不在白名单"的情况（如教育学院→社会科学院）给 partial penalty 而非全罚
- 变更后必须重跑 V5 ablation 脚本验证 ECE 不恶化

### 🔴 P0 — 天花板差异化

37.2% case 触达 70% 上限——天花板太低且无差异化：
- 对0-1层惩罚的 case：ceiling = 0.5（不需要强天花板）
- 对2-3层惩罚的 case：ceiling = 0.7（当前值，合理）
- 对4+层惩罚的 case：ceiling = 0.8（极少数，不应过度惩罚）
- 或改为分段函数：`total_penalty_ratio = min(raw_ratio, 0.5 + 0.1 * n_active)`

### 🟡 短期 — UI层不置信标记 (改动最小，最快见效)

**概率旁标记置信度**：
- 样本数 <5 → `🟡 数据稀缺，仅供参考`
- 样本数 5-9 → `🟢 样本较少，谨慎参考`
- 预测失败 → `⚪ 暂无足够数据评估`

**召回排序惩罚**：
- 对稀疏组合排序权重打折（不改展示概率，只改排序位置）
- n<5 → 排序按 `prob × 0.85`，n=5-9 → `prob × 0.92`

### 🟡 短期 — 预测失败兜底

6%的case完全预测失败（无GPA + 冷门本科），不跑模型硬跑更糟：
- **首选**：直接标记"数据不完整，请咨询顾问"——诚实比猜测好
- 或走简单规则引擎（GPA×院校层级×语言门槛 → 粗估概率）
- V7 发现缺失 GPA 与更高录取率相关（+13pp，correlational），但在控制混杂因子（院校层级、地区效应）前，不宜直接为无 GPA case 设独立 base probability（有 target leakage 风险）

### 🟢 中期 — 调整链参数调优

基于 V5 的逐层 ECE 贡献，调参优先级：
1. **Faculty penalty**（P0，见上）
2. GPA二次惩罚系数 0.15 → 0.10（ECE 影响虽小但原理性偏大）
3. 语言 pass_line `mean - 0.5σ` → `mean - 0.3σ`（降低触发率）
4. CrossMajor 保持现有机制（唯一对校准有正面贡献的层）
5. 衰减因子 0.85 → 0.75（让 2-3 层惩罚的 case 更快衰减，改善最差区间）
6. 调完后必须重跑 `pytest tests/data_quality/ -v` + V5 ablation

### 🟢 中期 — 相似度缓存扩展（V6 驱动）

ApplySquare 的相似度缓存命中率仅 22.8%，直接导致跨专业惩罚在外部数据上系统性放大：
- 为外部数据中的新专业对计算相似度并加入缓存
- 优先覆盖遗漏的人文社科专业（哲学、法学、新闻传播学等）
- 缓存命中率目标：>80%
- 重跑 V6 验证 -67pp 是否缩小

### 长期 — LLM盲评持续基准

- BlindEvalAgent 已注册为 `"blind_eval"`，可复用
- 每次调整链参数变更后重跑 AI 盲评对比（~40样本）+ **V5 ECE ablation**
- 成本：每次约 $0.08（40 cases × $0.002）
- 可选：生产环境中对低置信度case实时调用LLM作为第二意见

### 长期 — 数据积累

- 收集上线后的真实录取结果反馈（admitted/rejected ground truth）
- 冷门组合随着时间推移自然积累样本
- 定期重训练模型 + 重跑全量诊断套件 + V5 ablation

### 长期 — CrossMajor-style evidence adjustment 推广

CrossMajor 是五层中唯一有 data-driven evidence adjustment 的层，也是唯一对校准有正面贡献的层。其他层（特别是 Faculty、Language）应向它学习：
- Faculty: 统计该学部跨度的历史录取率，用 shrinkage 调整基础惩罚
- Language: 统计该语言分数段的历史录取率，用 shrinkage 替代固定阶梯
- 核心公式：`adjusted_penalty = base_penalty × evidence_mult`，其中 `evidence_mult = confidence × empirical_ratio + (1-confidence) × 1.0`

---
## 十二、报告体系

| 报告 | 路径 | 核心贡献 |
|------|------|---------|
| V1 校准深度分析 | `reports/v1_calibration/` | Base model 校准近乎完美 (ECE=0.026)，问题在后处理 |
| V2 五层 Ablation | `reports/v2_ablation/` | 调整链是排序核心 (τ→0 when off)，Language 影响最大 (注：单层 τ 值包含 decay 级联效应，非独立贡献) |
| V3 阈值敏感性 | `reports/v3_threshold_sensitivity/` | Threshold 应由业务方拍板，cost ratio 驱动 |
| V4 特征 Bootstrap | `reports/v4_feature_bootstrap/` | Top-3 特征排名完美稳定，language_score 最不稳定 |
| **V5 联合惩罚效应** | **`reports/v5_joint_penalty_effect/`** | **Faculty 是头号校准杀手，37% case 触顶，2-3 层最差** |
| **V6 外部漂移分解** | **`reports/v6_external_drift/`** | **-67pp = DGP mismatch + penalty amplification，相似度缓存命中仅 23%** |
| **V7 预测失败模式** | **`reports/v7_prediction_failures/`** | **缺失 GPA 与更高录取率相关（Δ=+13pp, correlational），DEC-010 需重审** |
| 全景诊断 | 本报告 | 所有诊断的综合视图 + 优化路线 |

---

## 运行命令

```bash
# 全量诊断套件 (63 tests)
pytest tests/data_quality/ -v

# 单独跑某项
pytest tests/data_quality/test_calibration_report.py -v -s
pytest tests/data_quality/test_sparsity_stress.py -v -s

# 调整链参数修改后必须重跑
pytest tests/data_quality/test_calibration_report.py -v -s
python reports/v5_joint_penalty_effect/run_joint_penalty_analysis.py
```
