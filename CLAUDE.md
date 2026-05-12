# CLAUDE.md

## Project

**Signals** — Streamlit 多页留学选校预测系统，XDF 内部使用。核心是香港/新加坡/澳门/马来西亚硕士申请录取概率预测。同时包含 HR 看板模块和客服问卷模块。

## Architecture

```
pages/*.py          → 薄路由层，只做 import 和调用 render/main 入口函数
src/pages/*/        → 页面实现：prediction/ 是核心，hr_*/ 是 HR 模块
src/utils/          → 共享基础设施：auth/ ui/ session_manager page_init（19项公共 API）
src/agent/          → LLM Agent 框架（DeepSeek，6 个 Agent：1 基类 + 3 业务 + 2 新业务）
src/machine_learning_models/ → 离线 XGBoost 训练流水线
config/             → 所有配置的单一真相源
src/machine_learning_models/ → 离线 XGBoost 训练流水线
config/             → 所有配置的单一真相源
.claude/            → settings.json（权限白名单）
```

页面初始化链条：`init_page()` → 加载 Streamlit config → 注入 CSS → E2 登录守卫 → 水印 → 返回 user_info。

## Module Index

每个模块有自己的 `README.md` 作为路由入口——打开任一模块的 README 即可理解：模块职责、代码位置、数据流向、公共 API。

| 模块 | 路径 | README |
|------|------|--------|
| Prediction（预测核心） | `src/pages/prediction/` | [README](src/pages/prediction/README.md) |
| HR Dashboard（成本看板） | `src/pages/hr_dashboard/` | [README](src/pages/hr_dashboard/README.md) |
| HR Structure Dashboard | `src/pages/hr_structure_dashboard/` | [README](src/pages/hr_structure_dashboard/README.md) |
| HR Profile（绩效档案） | `src/pages/hr_profile/` | [README](src/pages/hr_profile/README.md) |
| CS Survey（客服问卷） | `src/pages/cs_survey/` | [README](src/pages/cs_survey/README.md) |
| Admin（管理后台） | `src/pages/admin/` | [README](src/pages/admin/README.md) |
| Algorithm Lab（算法实验） | `src/pages/algorithm_lab/` | [README](src/pages/algorithm_lab/README.md) |
| Utils（共享基础设施） | `src/utils/` | [README](src/utils/README.md) |
| Agent（LLM Agent） | `src/agent/` | [README](src/agent/README.md) |
| ML Models（离线训练） | `src/machine_learning_models/` | [README](src/machine_learning_models/README.md) |
| Config（配置） | `config/` | [README](config/README.md) |

## Auth

企业 E2 OAuth。`src/utils/page_auth.py` 是每页都会调用的守卫。

- **生产环境**：session TTL 24 小时，过期重定向到 E2 登录页，回调验证签名后种 session
- **开发环境**：`config/dev_config.json` 设 `DEBUG_MODE: true` 绕过 E2，注入假用户
- 权限分三级：全局白名单 → 模块权限 → 细粒度功能权限（HR 专用）
- 管理员：`cuiting3`、`lijiapeng8`

## Config System

`APP_ENV` 环境变量选择 profile（默认 `test`），从 `config/app_config.json` 加载对应环境的配置。所有 API key、URL 都在那里。

关键配置文件：
- `app_config.json` — E2 OAuth key、OpenAI API key/URL
- `auth_config.json` — 白名单（~730人）、模块权限、维护模式开关
- `dev_config.json` — DEBUG_MODE + debug 用户身份
- `prediction_rules.json` — 23所学校难度排序、国家-学校映射
- `gpa_conversion_rules.json` — 北大/中科大/上交等校的 GPA 换算规则

## Prediction Pipeline (核心)

离线训练 → 模型部署 → 在线推理，三段式：

1. **离线**：`src/machine_learning_models/train.py` — XGBoost + CalibratedClassifierCV(sigmoid, prefit)，monotonic constraints，Optuna 可选调参
2. **在线推理**：`src/pages/prediction/flow/pipeline.py` — 表单规范化 → 特征构建 → 并行 XGBoost 推理 → 多阶段概率调整 → 合并去重 → 展示
3. **概率调整链**：GPA/语言偏差惩罚 → 跨专业惩罚(×0.5) → 跨学院惩罚(×0.3) → 职业学位降级 → TF-IDF 文本提升

预训练模型以 `.ubj` 格式存储在 `src/machine_learning_models/pre-trained_models/`。

## Agent System（全链路）

基于 Shared Context → Agent Registry → Orchestrator 三层架构。不依赖 LangChain，所有 Agent 继承 `BaseAgent`。通过 `Orchestrator.run()` 统一调用，Agent 懒注册。

### 架构

```
pages/hk.py
    │
    ├── render_lead_in_panel() → LeadInAgent
    │     └── apply_lead_in_to_form() → 表单自动填充 + expander 折叠/展开
    │
    ├── [Expander: 预测表单] → XGBoost 预测管道
    │
    └── _render_ai_explanation() → ExplainAgent
```

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `StudentContext` | `context.py` | 全链路共享上下文，从模糊到精确逐步填充 |
| `AgentRegistry` | `registry.py` | Agent 注册中心（dict wrapper），懒注册 |
| `AgentOrchestrator` | `orchestrator.py` | 编排路由：取 Agent → 调用 → 写回 Context + audit history |
| `BaseAgent` | `base_agent.py` | Agent 基类：OpenAI client、内存+文件缓存、重试、三级 JSON 修复（直接 parse → 轻量正则 → API repair）、结构化日志（REQ START/OK/TIMEOUT） |
| `LeadInAgent` | `lead_in_agent.py` | 前期 NLU：碎片文本 → 结构化背景 + 方向性评估 + 追问（支持多轮增量） |
| `ExplainAgent` | `explain_agent.py` | 中期解释：预测结果 + 背景 → 自然语言解读（overview / strengths / concerns / summary） |
| `apply_lead_in_to_form` | `form_bridge.py` | **桥接层**：extracted_background → 表单 widget state，含模糊匹配（精确→子串→difflib）+ 四种经历映射 |
| `BoundaryCaseAgent` | `boundary_case_agent.py` | 相似/跨专业推荐平衡决策（已有） |
| `TextPreprocessingAgent` | `text_preprocessing_agent.py` | 背提文本质量评估（已有） |
| `BackgroundFacultyAgent` | `background_faculty_agent.py` | 学部推断（已有） |

### 数据流

```
顾问自由文本
    │
    ▼
LeadInAgent.run(StudentContext)
    │  ├── extracted_background（university, major, gpa, language, target_schools, paper...）
    │  ├── quick_assessment（初步评估）
    │  └── suggested_questions（追问）
    │
    ▼
apply_lead_in_to_form(ctx, session_manager)
    │  ├── 模糊匹配院校/专业 → 设 widget state
    │  ├── 直接设 GPA / 语言 / 目标 / 经历 widget state
    │  └── 写 lead_in_form_summary → expander 标题
    │
    ▼
[Expander 预测表单] — 核验/修改 → 提交 → XGBoost 预测管道
    │
    ▼
ExplainAgent.run(StudentContext)
    │  ├── overview（整体评估）
    │  ├── strengths / concerns
    │  └── summary
```

### API 日志格式

所有 Agent 输出结构化日志，便于排查超时/解析失败：

```
[Agent名] REQ START | prompt=489chars ~196tk | max_tokens=300 | timeout=15s
[Agent名] → attempt 1/3
[Agent名] REQ OK | total=7667ms | attempts=1 | output=627chars ~263tk | usage: prompt=222 completion=281
[Agent名] PARSE: JSON decode failed | error=... → lightweight fix succeeded
```

三级 JSON 容错：`json.loads` → `_fix_json_lightweight`（正则修缺逗号/尾部逗号，<1ms）→ `_repair_json_once`（API 兜底，~9s）。

## Key Commands

```bash
# 运行应用
streamlit run main.py

# 运行测试（stress tests 需要 psutil，如未安装需 --ignore=tests/stress）
pytest tests --ignore=tests/stress -q

# 离线训练模型
python -m src.machine_learning_models.train --data-path data/cases.feather

# 预计算 major 相似度（需要 E5 模型）
python scripts/precompute_similarities.py

# 训练文本提升模型
python scripts/train_text_tfidf.py

# 生成录取相关性矩阵（Monte Carlo 用）
python scripts/generate_correlation_matrix.py
```

## Important Constraints

- **没有数据库**：所有数据都是 `.feather` 文件 + Streamlit cache。`st.cache_data(ttl=600~3600)` 做数据缓存，`st.cache_resource` 做模型缓存
- **Streamlit 无文件监听**：`.streamlit/config.toml` 设了 `fileWatcherType = "none"`，改代码后需要手动刷新
- **XGBoost monotonic constraints**：GPA、语言成绩、四段经历计数全部强制单调递增，训练时注意不要破坏
- **生产部署在 80 端口**：`server.port = 80`，`server.address = "0.0.0.0"`，XSFR/CORS 关闭（前面有反向代理）
- **session_state 是唯一状态载体**：通过 `SessionManager` 强类型存取，没有 disk persistence
- **CSS 分两层**：全局样式（`style.css`）+ Signals 品牌设计系统（`hk_style/00~40_*.css`，按加载顺序叠加）
- **实验性模块**：`src/pages/prediction/` 下有 3 个从已删除 commit 恢复的实验目录（school_combination_optimizer_algorithm/、admission_probability_calculator_components/、page_components/pdf_generation/），所有文件有 `# !!EXPERIMENTAL:` 标记，grep 可定位。详见 `src/pages/prediction/EXPERIMENTAL_ROUTE.md`

## Governance Rules

### Rule 1: Maximum File Size

- 单个 `.py` 文件不超过 **300 行**。
- 接近 ~280 行时必须拆分为更小的专注模块。
- **豁免**：
  - 第三方/vendored 代码（如 `src/pages/algorithm_lab/pymoo/`）
  - 配置文件常量（需在文件顶部标注 `# CONFIG_CONSTANTS_ALLOWED_LARGE`）
- **拆分方式**：将相关函数提取到新子模块（如 450行 `processor.py` → `processor_core.py` + `processor_helpers.py`），公共 API 从 `__init__.py` 暴露。

### Rule 2: Public API via __init__.py

- 每个模块（有 `__init__.py` 的目录）必须通过 `__init__.py` 的 `__all__` 暴露公共 API。
- "公共" = 任何包外代码 import 的名字。
- **模式**：
  ```python
  from .submodule import public_function, PublicClass
  __all__ = ["public_function", "PublicClass"]
  ```
- 内部辅助函数（`_` 前缀）不出现在 `__all__` 中。
- **已合规**：`admin`、`agent`、`cs_survey`、`hr_dashboard`、`hr_profile`、`hr_structure_dashboard`、`machine_learning_models`、`utils`

### Rule 3: Module README Required

- `src/pages/`、`src/utils/`、`src/agent/`、`src/machine_learning_models/`、`config/` 下的每个一级模块必须有 `README.md`。
- 含 3+ 个非 `__init__.py` 文件的子目录也应有 README。
- README 遵循 `src/pages/prediction/README.md` 模板：
  1. 模块概述（1段）
  2. 目录结构（ASCII tree）
  3. 端到端流程（ASCII 流程图）
  4. 核心组件（每个附简要说明）
  5. 数据流
  6. 子模块文档链接
  7. 依赖

### 当前超过300行的文件（已拆分 nivo.py，剩余待处理）

| 文件 | 行数 | 备注 |
|------|------|------|
| `hr_profile/ui/charts.py` | 730 | 按图表类型拆分 |
| `cs_survey/ui/theme_css.py` | 680 | CSS 模板常量 |
| `headcount.py` | 623 | KPI 计算拆分 |
| `json_api.py` | 441 | API 端点拆分 |
| `form_state.py` | 396 | 状态管理拆分 |
| `orchestrator.py` | 393 | 编排逻辑拆分 |
| `processor.py` | 381 | 处理器拆分 |
| `hr_dashboard/config.py` | 357 | 标注 CONFIG_CONSTANTS_ALLOWED_LARGE |
