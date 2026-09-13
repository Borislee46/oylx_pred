# Signals

留学硕士选校的录取概率估计。给定学生画像与「目标校 × 目标专业」组合，输出校准后的录取概率与选校组合。

本文按问题、方法、实现、实验、局限组织。所有公式与常数均标注代码位置。

---

## 1. 问题

案例库有 81,830 条真实录取记录。但把它展开到预测所需的特征空间，绝大部分格子是空的：

| 组合维度 | 理论基数 | 观测数 | 覆盖率 |
|---|---:|---:|---:|
| 目标校 × 目标专业 | 30,636 | 1,518 | 5.0% |
| 背景校 × 目标校 × 目标专业 | 50,855,760 | 54,324 | 0.107% |
| + 背景专业 | 11,747,680,560 | 64,443 | 0.00055% |

三维视角下每个格子中位数 1 条记录（均值 1.51，最大 39），75.7% 的三元组只有单条观测。二维视角中 28.2% 的格子样本量 ≤5，11.4% 的格子正例数为 0。此外 `toefl` 字段 93.5% 缺失，`paper_detail` 78.7% 缺失。

判别式模型在这种条件下直接拟合会退化为对少数热门组合的记忆。系统需要的是从稀疏观测向未观测组合外推，并如实报告不确定性。

---

## 2. 方法

### 2.1 收缩目标编码

`src/ml/te_encoder.py:21`

类别均值按样本量做 logistic 收缩：

```
w(n) = σ((n − 10) / 5)
TE(c) = w(n_c) · ȳ_c + (1 − w(n_c)) · ȳ_global
```

其中 `σ` 为 logistic 函数（`src/utils/numeric/scalars.py:48`）。收缩权重取值为：

```
w(1)  = 0.1419      w(10) = 0.5000
w(20) = 0.8808      w(50) = 0.9997
```

样本量低于 10 时大部分权重仍落在全局先验上。

> 参数命名注意：`sigmoid_shrinkage(n, k=10, s=5)` 中 `k` 是**中点**、`s` 是**尺度**，与 `sigmoid_k(x, k, x0)` 的位置参数顺序相反（`te_encoder.py:22` 传的是 `sigmoid_k(n, 1/s, k)`）。

编码列：`target_university, target_major, background_university, background_major, faculty`。

训练集编码由 5 折 `StratifiedKFold(shuffle=True, random_state=42)` 的 **out-of-fold** 统计量生成（`te_encoder.py:59`），避免标签泄漏。推理时未见类别直接取全局均值（`te_encoder.py:108`）。

### 2.2 层次 Beta-Binomial 兜底

`src/adjustment/fallback.py:102`

定义四级匹配粒度，由粗到细：

```
L3  (target_uni)
L2  (target_uni, target_major)
L1  (bg_uni, target_uni, target_major)
L0  (bg_uni, bg_major, target_uni, target_major)
```

对每一级，以**上一级的后验**作为本级先验，做 Beta-Binomial 共轭更新：

```
p_l = (k_l + m · p_{l+1}) / (n_l + m),      m = 5.0
```

`m = BETA_BINOMIAL_PRIOR_STRENGTH`（`src/adjustment/config.py`）。级联从 L3 起（先验为全局录取率），逐级细化为 L0 的后验。样本量 n=0 的级别直接继承上一级后验，`shrinkage_weight = 1.0`。

输出区间用 Wilson score interval（`src/utils/numeric/confidence.py:10`，`z = 1.96`）：

```
denom  = 1 + z²/n
center = (p̂ + z²/(2n)) / denom
margin = z·√((p̂(1−p̂) + z²/(4n)) / n) / denom
CI     = [center − margin, center + margin]
```

区间在小样本下自然变宽，这是如实反映不确定性。`FallbackResult` 透出 `sample_count / effective_n / fallback_level / shrinkage_weight`，调用方据此决定是否向用户展示该概率。

### 2.3 学生画像 KNN 外推

`src/adjustment/knn_retrieval.py`

**候选池构造**（`_select_pool`，`:160`）按目标侧分层，顺序并集直至 |C| ≥ k：

```
同目标院校  →  同 Tier 院校  →  同目标专业（不限院校）
```

Tier 划分由 `TIER_BOUNDARIES = (5, 12, 19)` 对院校难度序切分为 4 档（`tier_calibration.py:16`）。

**距离度量**（`_compute_distances`，`:203`）作用在学生画像四维上，权重 `DEFAULT_WEIGHTS = (0.30, 0.20, 0.15, 0.35)`（`:26`）：

```
d = 0.30·d_school + 0.20·d_gpa + 0.15·d_lang + 0.35·d_major
```

各分量：

```
d_school = min(|score_pool − score_student| / 0.75, 1)        # _SCHOOL_SCORE_SPAN
           若同院校层次但非同一院校，取 max(d_school, 0.15)     # _SAME_TIER_DIFF_SCHOOL_PENALTY
d_gpa    = min(|z_pool − z_student| / 3.0, 1)                  # _Z_SPAN
           池内 GPA 缺失 → 1.0；学生 GPA 缺失 → 0.5
d_lang   = 同 d_gpa
d_major  = 1 − fuzzy_similarity(背景专业, 池内背景专业)          # rapidfuzz
```

同层次异校的 0.15 下界避免把「A 校的案例」直接当作「B 校的案例」。`_exact_mask`（`:266`）另行标出四元组完全命中的样本，优先按 GPA 距离排序。

62 万组背景-目标相似度离线预计算（`cache/background_target_similarity.feather`），在线仅查表。

### 2.4 领域先验补偿

`src/adjustment/` 共 30 个模块，覆盖语言分档、GPA 场景、跨院系约束、边际案例、logit 尺度上的文本 uplift 等。数据不足以支撑的部分由显式规则承担，每条规则可通过 ablation 配置单独关闭，不需要重训模型即可度量其对最终概率的贡献。

### 2.5 Tier 单调性修复

`src/adjustment/tier_calibration.py:59`

稀疏观测无法为模型提供「更高档院校的录取概率不应更高」这一约束。修复放在链尾，对相邻 tier 对求解：

```
if max(q_lower) ≤ min(q_upper) + δ:  不修复
else:  mid = (max(q_lower) + min(q_upper)) / 2
       将 q_lower 中取到 max 的元素、q_upper 中取到 min 的元素同时置为 mid
δ = TIER_RANK_REPAIR_MIN_VIOLATION = 0.08
```

迭代至最大调整量 < 1e-9 或达 `TIER_RANK_REPAIR_MAX_ITER = 20`，最终概率裁剪到 `[0.005, 0.95]`。

> 与 PAVA/isotonic regression 的区别：这里只移动**越界的那一对极值**到中点，不做全局单调投影，因此不改变非极值元素的相对序。选择这一形式是因为调整链各模块独立加分，全局投影会掩盖单个模块的贡献，不利于归因。实测一轮 `max_violation = 0.0691 < 0.08`，未触发。

---

## 3. 智能体层

LLM 的作用域被限定为：将自由文本映射到 17 个工具调用构成的序列。它不参与概率计算，也不产生准入判断。

```
form   11  read_form / write_form / submit_prediction / expand_form / start_new_profile ...
probe   3  只读查询：项目门槛、院校信息、相似已录案例
recall  3  录取统计、按背景分层的估计
```

单一 ReAct 执行体（`src/agent/lead_in/tool_agent.py`），不做 router-first 分流。早期按意图分类再路由的实现在混合输入上会丢失分支（同时提供背景与提问时只走其中一条）。

约束来自外围确定性代码而非提示词：

| 机制 | 位置 | 作用 |
|---|---|---|
| `FormGateway` Protocol | `src/agent/tools/form_gateway.py` | 唯一写入口，字段类型受限 |
| work 副本 + `_commit_work_deps` | `tool_agent.py` | 写入先在副本上生效，成功才提交；超时或异常整体丢弃 |
| `LeadInPhase` 状态机 | `src/agent/lead_in/state_machine.py` | IDLE / GATING / BLOCKED / EXTRACTING / AWAITING / PREDICTING / DONE / ERROR |
| `session_lock` | `src/agent/lead_in/session_lock.py` | 会话级并发串行化 |
| `UsageLimits` | pydantic-ai | 工具轮数上限 |

领域规则（字段取值域、越界过滤、召回时机）写在 `src/agent/tools/*.py` 的 docstring 中，由 pydantic-ai 注入 tool schema，不重复写入提示词。同一条规则出现两处会产生矛盾信号，边界由 `tests/unit/agent/test_agent_prompts_wired.py` 守住。

---

## 4. 实验

### 4.1 模型

XGBoost `binary:logistic`，`tree_method=hist`，`enable_categorical=True`，`n_estimators=590`，`max_depth=14`，`random_state=42`。特征 10 维。

留出集指标（`reports/model_evaluation.json`）：

| 指标 | 值 |
|---|---:|
| ROC-AUC | 0.7635 |
| Average Precision | 0.5947 |
| Brier | 0.1754 |
| Log Loss | 0.5305 |
| Precision / Recall (τ=0.24) | 0.4568 / 0.8180 |
| 混淆矩阵 | tn 6092 · fp 5066 · fn 948 · tp 4260 |

特征重要性（gain）：

```
target_major 0.2800   background_university 0.2048   target_university 0.0991
background_major 0.0965   gpa 0.0738   language_score 0.0619
award_count 0.0510   research_count 0.0476   paper_count 0.0437   internship_count 0.0417
```

`target_major` 的权重约为 `target_university` 的 2.8 倍：在同一学生画像下，目标专业维度对结果的区分度高于目标院校维度。

### 4.2 决策阈值

阈值取 **0.24**，非 F1 最优点。案例库正例率 31.8%，τ=0.24 对应预测正例率 0.570，即模型系统性高报。这一偏移是有意为之：漏判一所可申院校的代价高于多推一所冲刺院校。代价是精确率 0.4568。

阈值扫描结果（`best_f1_threshold_scan`）：τ=0.32 时 F1 0.5936 / P 0.5202 / R 0.6912。

### 4.3 校准

概率经 sigmoid 校准（`a=-2.8715, b=1.898`），抑制树模型在分布两端的过度自信。校准参数来自训练期拟合，记录于 `reports/v10_shap_explanation/shap_v10_report.json` 的 `metadata.calibration`。

### 4.4 延迟

热启动，3 目标校 × 3 目标专业：

```
相似度检索      0.156 s     62 万条预计算对查表
结果处理        0.047 s
XGBoost 推理    0.031 s     5 组合 × 10 特征，单线程
Tier 校准       0.000 s
pipeline 总计   0.547 s     自报值；差额约 0.31 s 为加载与预处理，未单独打点
```

冷启动首轮 2.81 s。检索是主导项而非推理，这是相似度矩阵必须离线预计算的原因。

---

## 5. 复现

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run main.py      # localhost:8501，访问码 SIGNALS2026
```

案例库与模型权重不在仓库内（53.4 MB）。环境变量、key 配置、数据放置位置见 [`docs/SETUP.md`](docs/SETUP.md)。

部署见 [`deploy/README.md`](deploy/README.md)。

---

## 6. 局限

- **评价协议**：留出集使用 `split_method = random`。申请结果本质上是时序过程，随机划分会引入同批次信息泄漏，指标偏乐观。时间序切分下的结果尚未系统评估。
- **选择偏差**：案例库正例率 31.8%，仅包含已提交申请者。未被申请过的学校-专业组合在训练数据中不存在，模型对该区域只能依赖 2.3 节的相似度外推与 2.4 节的领域先验。
- **缺失率**：7.4% 的案例缺关键字段（6,043 / 81,830，主因缺背景专业 5,291 条），这部分走 fallback 而非模型。
- **区间未做覆盖率验证**：2.2 节的 Wilson 区间未在留出集上做经验覆盖率检验，其标称水平与实测水平的偏差未知。
- **未容器化**，无水平扩展设计。`st.cache_resource` 使模型常驻内存，多副本各加载一份 17 MB 权重。
- **相似度矩阵更新为手动流程**，更换案例批次需重跑 `scripts/precompute_similarities.py`。
- **单元测试未随仓库分发**，第三方无法验证改动。本地执行 688 passed / 1 skipped。

---

## 7. 许可证

代码 Apache-2.0（`LICENSE`）。案例库、模型权重与学生信息不在仓库内，亦不属本许可证授权范围，详见 `NOTICE`。
