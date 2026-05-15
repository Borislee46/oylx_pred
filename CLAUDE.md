# CLAUDE.md

## Project

**Signals** — 香港/新加坡/澳门/马来西亚硕士留学录取概率预测系统，XDF 内部使用。同时包含 HR 看板、客服问卷、管理后台。

> **工作准则**：每次设计、实现、验证都按 Senior DS 标准——Why before How、验证先于声称、量化不确定性、显式trade-off、面试可交付、懂得不做什么、吃透了用大白话讲出来。详见 `TODO_ROUTE.md`。

## Architecture

```
pages/*.py          → 薄路由层
src/pages/*/        → 页面实现：prediction/ 是核心，hr_*/ 是 HR 模块
src/utils/          → 共享基础设施（auth / ui / session / page_init）
src/agent/          → LLM Agent 框架（DeepSeek，7个Agent）
src/machine_learning_models/ → XGBoost 离线训练 + 预训练模型
config/             → 所有配置的单一真相源
```

页面初始化链条：`init_page()` → Streamlit config → CSS → E2 登录守卫 → 水印 → user_info。

## DS 视角：数据全链路

从原始输入到最终概率，数据经过以下变换。每一步都在引入假设和偏差。

### 链路全景

```
原始表单输入（GPA、语言、院校、专业、经历文本）
    │
    ▼
[1] 特征工程 → XGBoost 推理 → Platt 校准概率
    │  问题：XGBoost 在稀疏区域外推不可控
    │  模型从不知道"这个组合只有3个样本"
    │
    ▼
[2] GPA 惩罚（二次函数，z-score）→ ×(1 - min(0.8, 0.15×z²))
    │  假设：GPA 低于均值时录取概率非线性下降
    │  问题：0.15 这个系数怎么来的？为什么是二次不是线性？
    │
    ▼
[3] 语言惩罚（阶梯函数，6级 L1-L5 + L3.5）→ ×(1 - 0.05~0.95)
    │  假设：语言成绩有硬门槛效应
    │  问题：GPA 用二次、语言用阶梯——两种函数形式的 DS 理由是什么？
    │  L3.5(0.20) 为 2026-05 新增，拆分 6.0-6.5 雅思密集区（89% 提交集中在此区间）
    │
    ▼
[4] 跨专业惩罚 → ×(0.5~1.0)，按相似度线性插值
    │  假设：相似度 <0.89 即触发
    │  问题：0.89 的阈值从哪来的？为什么不是 0.85 或 0.92？
    │
    ▼
[5] 跨学部惩罚 → ×0.3（硬编码学部规则）
    │  假设：理→文 ≈ 不可能，理→工 ≈ 无障碍
    │
    ▼
[6] 职业学位惩罚 → ×0.7（无实习申MBA）
    │
    ▼
[7] 仲裁器衰减叠加 → 总惩罚上限 70%
    │  问题：每层单独看都有道理，但没人设计过 5 层叠加后的联合效应
    │
    ▼
[8] TF-IDF 文本提升 → +0~15%
    │
    ▼
[9] 归一化 → floor 0.005 → 最终概率（0.005 ~ 0.72）
```

### 关键数据事实

| 事实 | DS 含义 |
|------|---------|
| 61,716 行，52,327 个唯一组合 | 平均每个组合 1.18 个样本 |
| 99.4% 组合 ≤4 个样本 | 模型在几乎所有输入上都是外推，不是内插 |
| 只有 1 个组合有 30+ 样本 | 不存在"统计显著"的个案预测 |
| 训练集 ECE 0.1155，偏差 -9pp | 内部都偏，外部更偏 |
| 外部 ApplySquare 偏差 -67pp | 调整链惩罚被相似度匹配差异放大 7 倍 |
| 模型从不输出 >0.72 | 仲裁器 70% 上限 + floor 效应 |
| 6% 输入完全预测失败 | 无 GPA + 冷门本科 → 冷启动，fallback.py 提供 Wilson CI 级联兜底 |
| Compass 17K 外部验证 | 第二外部数据集，质量存疑，已纳入 data_quality 测试套件 |
| ApplySquare 507 外部验证 | 偏差 -67pp，惩罚链在外部数据上问题加速恶化 |

## 已知问题 & 批判性审视

### 1. 调整链：合理但失控

5 层惩罚各自有 `DECISIONS.md` 的设计决策，但**联合效应从未被系统设计过**。DEC-007 加了衰减仲裁（0.85/layer）来缓解，但这是事后补救，不是事前设计。

- ECE=0.1155 > 0.10 → 严重失校准
- 分层偏差不均匀：C9 学生被低估 18pp，双非只被低估 6pp——惩罚对强者的伤害更大
- 外部数据 -67pp 证明：当相似度匹配变差时，更多惩罚被触发，问题指数级放大

**批判性问题**：5 层惩罚的设计逻辑是一层一层加上去的（"这里需要修正 GPA""这里需要修正语言"），但没有人在加了 5 层之后问"这些惩罚加在一起还是合理的吗？"

### 2. 无真正的 held-out 测试集

校准报告跑在训练数据上（cases.feather），外部数据验证只有 ApplySquare 507 行和 Compass 17K 行（质量存疑）。**我们不知道模型在真实新用户上的表现。** 这是最大的未知风险。

### 3. 缺失值处理：有信息不用

GPA/Language 缺失用 median imputation——DEC-010 说这是"保守"的。但缺失本身可能是信号：没填 GPA 的学生 GPA 可能确实低。用 median 反而给了这类学生一个"正常"的起点。

### 4. BlindEvalAgent：prompt 陷阱

LLM 盲评作为 benchmark 是有价值的，但：
- 语言分数格式 bug（归一化 0.72 → LLM 理解为 IELTS 0.72）已修，但类似的 prompt 理解偏差可能还存在
- LLM 从不自评 low confidence——高估自己的判断准确性
- 40 样本盲评的 Pearson r=0.44——排序一致性偏弱，LLM 和模型在"谁更容易录取"上意见分歧大

### 5. 全链路解释 ≠ 模型解释

当前 `_adjustment_trace` 展示的是调整链做了什么，不是模型本身为什么给出某个 base prob。`boundary_explainer_design.md` 设计了这个但未实现。用户看到的是"因为我们惩罚了你的 GPA"，而不是"因为历史上类似背景的学生有 X% 被录取"。

## Module Index

| 模块 | 路径 | 职责 |
|------|------|------|
| Prediction | `src/pages/prediction/` | 预测核心：表单→推理→调整→展示 |
| Agent | `src/agent/` | LLM Agent：lead-in / explain / blind_eval / application / pdf |
| ML Models | `src/machine_learning_models/` | XGBoost 训练 + 预训练模型 |
| HR Dashboard | `src/pages/hr_dashboard/` | HR 成本看板 |
| HR Structure | `src/pages/hr_structure_dashboard/` | HR 架构看板 |
| HR Profile | `src/pages/hr_profile/` | HR 绩效档案 |
| CS Survey | `src/pages/cs_survey/` | 客服问卷 |
| HK Dashboard | `src/pages/hk_dashboard/` | 香港运营仪表盘 |
| School View | `src/pages/school_view/` | 学校画像 + What-If 模拟 |
| Admin | `src/pages/admin/` | 管理后台 |
| Algorithm Lab | `src/pages/algorithm_lab/` | 统计算法实验 |
| Utils | `src/utils/` | 共享基础设施 |
| Config | `config/` | 配置单一真相源 |

每个模块有 `README.md` 作为路由入口。

## Prediction Pipeline

离线训练 → 模型部署 → 在线推理，三段式：

1. **离线**：`src/machine_learning_models/train.py` — XGBoost + CalibratedClassifierCV(sigmoid, prefit)，monotonic constraints，Optuna 可选
2. **在线推理**：`src/pages/prediction/flow/pipeline.py` — 规范化 → 特征构建 → XGBoost推理 → 调整链 → 合并去重
3. **概率调整链**：GPA惩罚 → 语言惩罚 → 跨专业(×0.5) → 跨学部(×0.3) → 职业学位 → 仲裁器 → 文本提升

预训练模型：`.ubj` 格式在 `src/machine_learning_models/pre-trained_models/`。

调整链参数修改后必须重跑：`pytest tests/data_quality/test_calibration_report.py -s`

## Agent System

不依赖 LangChain。Shared Context → Registry → Orchestrator 三层架构。所有 Agent 继承 `BaseAgent`。

```
LeadInAgent（碎片文本→结构化背景+追问）
    → apply_lead_in_to_form（桥接：背景→表单widget state）
    → [预测表单]
    → ExplainAgent（结果→自然语言解读）
```

8 个 Agent：BaseAgent、LeadInAgent、ExplainAgent、BoundaryCaseAgent、TextPreprocessingAgent、BackgroundFacultyAgent、**BlindEvalAgent**（AI盲评benchmark）、ApplicationAgent（申请策略生成）、PDFAIAgent（PDF报告文案）。

三级 JSON 容错：`json.loads` → 正则修复(<1ms) → API repair(~9s)。

## Auth

企业 E2 OAuth。`DEBUG_MODE: true` 绕过。权限三级：白名单 → 模块 → 细粒度（HR专用）。

## Key Commands

```bash
streamlit run main.py                              # 启动
pytest tests --ignore=tests/stress -q               # 常规测试
pytest tests/data_quality/ -v                       # 预测质量诊断（62 tests）
pytest tests/unit/ -v                               # 单元测试（12 tests）
python -m src.machine_learning_models.train \
  --data-path data/cases.feather                    # 离线训练
python scripts/precompute_similarities.py            # 预计算major相似度
python scripts/train_text_tfidf.py                   # 训练文本提升模型
python scripts/generate_correlation_matrix.py        # Monte Carlo相关性矩阵
```

## Data Quality Tests (62 tests)

| 文件 | 覆盖 |
|------|------|
| `test_calibration_report.py` | ECE + Brier + 分层校准 |
| `test_corner_cases.py` (22) | GPA/语言边界、极端组合、跨专业/学院 |
| `test_prediction_rigor.py` (32) | 单调性、稳定性、阈值、TOEFL等价、无悬崖 |
| `test_external_data_validation.py` (3) | Compass/ApplySquare校准 + 分布漂移 |
| `test_ai_blind_eval.py` | LLM盲评 vs 模型对比 |
| `test_model_outputs.py` (2) | 标准高低分范围 |
| `test_sparsity_stress.py` | 稀疏度扫描 + 压测 + LLM对比 |
| `test_fallback.py` | Fallback 级联兜底测试 |

## Test Suite

62 data_quality tests + 12 unit tests + 2 integration + 1 stress + 20+ root-level tests。

| 文件 | 覆盖 |
|------|------|
| `test_probability_adjuster.py` | GPA/语言惩罚全路径（2026-05 新增） |
| `test_hk_state_machine.py` | 步骤条状态机 + UI phase 转换（2026-05 新增） |
| `test_form_validator.py` | 表单校验全路径（2026-05 新增） |
| `test_experience_text_validator.py` | 背提文本校验（2026-05 新增） |
| `test_explain_profiles.py` | ExplainAgent 4 种 profile 分类（2026-05 新增） |
| `test_signal_scorer.py` | TF-IDF 信号评分（2026-05 新增） |
| `test_similarity_adjuster.py` | 相似度调整规则（2026-05 新增） |
| `test_competitiveness.py` | 竞争力评分（2026-05 新增） |
| `test_school_view.py` | 学校视图（2026-05 新增） |

## Important Constraints

- **没有数据库**：所有数据 `.feather` + Streamlit cache。st.cache_data(ttl=600~3600)，st.cache_resource 做模型缓存
- **Streamlit 无文件监听**：改代码需手动刷新
- **XGBoost monotonic constraints**：GPA、语言、四段经历全强制单调递增
- **生产 80 端口**：`server.address = "0.0.0.0"`，XSFR/CORS 关闭
- **session_state 唯一状态载体**：SessionManager 强类型存取
- **CSS 两层**：全局 `style.css` + Signals 品牌设计系统
- **实验性模块**：`school_combination_optimizer_algorithm/` 等 3 个目录从已删除 commit 恢复，所有文件有 `# !!EXPERIMENTAL:` 标记。详见 `src/pages/prediction/EXPERIMENTAL_ROUTE.md`

## Governance

### Rule 1: 文件不超过 300 行

接近 280 行拆分为子模块。豁免：第三方/vendored 代码、配置常量（标注 `# CONFIG_CONSTANTS_ALLOWED_LARGE`）。

### Rule 2: 公共 API 通过 __init__.py

```python
from .submodule import public_function
__all__ = ["public_function"]
```

### Rule 3: 每个模块必须有 README.md

一级模块 + 3+ 文件的子目录。模板：概述 → 目录结构 → 流程 → 组件 → 数据流 → 子模块 → 依赖。

### 超行文件（待处理）

| 文件 | 行数 |
|------|------|
| `hr_profile/ui/charts.py` | 730 |
| `cs_survey/ui/theme_css.py` | 680 |
| `headcount.py` | 623 |
| `json_api.py` | 441 |
| `form_state.py` | 396 |
| `orchestrator.py` | 393 |
| `processor.py` | 381 |
| `hr_dashboard/config.py` | 357 (标注豁免) |

## 相关文件

- `TODO_ROUTE.md` — 4 大 TODO + DS 行为准则 + Senior DS 面试视角
- `DECISIONS.md` — 14 个设计决策的正反论证
- `reports/prediction_diagnosis_20260513.md` — 全景诊断报告
- `MODEL_CARD.md` — 模型能力/局限声明
