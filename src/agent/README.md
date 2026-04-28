# Agent 模块技术文档

## 1. 模块概述

<<<<<<< HEAD
`src/agent` 是 LLM Agent 系统，基于 DeepSeek（OpenAI 兼容 API）。采用 Shared Context → Agent Registry → Orchestrator 三层架构，覆盖留学全链路：前期 NLU（碎片信息提取）→ 中期决策（边界 case + 预测解释）→ 待扩展后期（申请管理）。不依赖 LangChain，所有 Agent 继承 `BaseAgent`。
=======
`src/agent` 是 LLM Agent 系统，基于 DeepSeek v3.2（OpenAI 兼容 API）。采用 Shared Context → Agent Registry → Orchestrator 三层架构，覆盖留学全链路：前期 NLU（碎片信息提取）→ 中期决策（边界 case + 预测解释）→ 待扩展后期（申请管理）。不依赖 LangChain，所有 Agent 继承 `BaseAgent`。
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

## 2. 目录结构

```
agent/
<<<<<<< HEAD
├── __init__.py                     # 公共 API（14项导出，含懒注册）
├── base_agent.py                   # BaseAgent 基类（client、缓存、重试、JSON修复）
├── context.py                      # StudentContext（全链路共享上下文）
├── registry.py                     # AgentRegistry（懒加载注册中心）
├── orchestrator.py                 # AgentOrchestrator（单Agent + Pipeline编排）
├── schemas.py                      # TypedDict 数据契约（ExtractedBackground 等）
├── boundary_case_agent.py          # 边界案例决策 Agent
├── boundary_case_prompts.py        # 边界案例 Prompt 模板
├── text_preprocessing_agent.py     # 背提文本预处理 Agent
├── text_preprocessing_prompts.py   # 文本预处理 Prompt 模板
├── background_faculty_agent.py     # 背景学部推断 Agent
├── background_faculty_prompts.py   # 学部推断 Prompt 模板
├── lead_in_agent.py                # 前期 NLU Agent
├── lead_in_prompts.py              # LeadIn Prompt 模板
├── explain_agent.py                # 预测解释 Agent
=======
├── __init__.py                     # 公共 API（8项导出）
├── base_agent.py                   # BaseAgent 基类（client、缓存、重试、JSON修复）
├── context.py                      # StudentContext（全链路共享上下文）
├── registry.py                     # AgentRegistry（Agent 注册中心）
├── orchestrator.py                 # AgentOrchestrator（编排路由）
├── boundary_case_agent.py          # 边界案例决策 Agent（已有）
├── boundary_case_prompts.py        # 边界案例 Prompt 模板
├── text_preprocessing_agent.py     # 背提文本预处理 Agent（已有）
├── text_preprocessing_prompts.py   # 文本预处理 Prompt 模板
├── background_faculty_agent.py     # 背景学部推断 Agent（已有）
├── background_faculty_prompts.py   # 学部推断 Prompt 模板
├── lead_in_agent.py                # 前期 NLU Agent（新）
├── lead_in_prompts.py              # LeadIn Prompt 模板
├── explain_agent.py                # 预测解释 Agent（新）
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc
└── utils.py                        # Agent 共享工具
```

## 3. 架构

```
┌──────────────────────────────────────────────────────┐
│                AgentOrchestrator                      │
│                   (编排路由)                           │
├──────────────────────────────────────────────────────┤
│                AgentRegistry                          │
│        lead_in | explain | boundary | ...             │
├──────────┬──────────┬──────────┬─────────────────────┤
│ LeadIn   │ Explain  │ Boundary │ TextPrep / Faculty  │
│ Agent    │ Agent    │ Agent    │ (已有)              │
└────┬─────┴────┬─────┴────┬─────┴─────────────────────┘
     │          │          │
     ▼          ▼          ▼
┌──────────────────────────────────────────────────────┐
│                StudentContext                         │
│  raw_input → extracted_background → prediction_      │
│  results → application_plan                          │
│  + history（审计追踪）                                │
└──────────────────────────────────────────────────────┘
```

## 4. 端到端流程

```
顾问自由文本
    │
    ▼
LeadInAgent.run(StudentContext)
    ├── NLU：碎片文本 → 结构化背景
    ├── quick_assessment：方向性评估
    └── suggested_questions：追问建议
    │
    ▼
[表单自动填充] → [XGBoost 预测管道]
    │
    ▼
ExplainAgent.run(StudentContext)
    ├── overview：整体评估
    ├── strengths / concerns
    └── summary
```

### 已有 Agent 管线

```
BoundaryCaseAgent.evaluate_boundary_cases()
    ├── 输入：sim/cross 推荐列表 + 用户背景
    ├── 决策：relax / tighten
    └── 输出：调整后的推荐平衡

TextPreprocessingAgent → has_meaningful_experience_text 判定
BackgroundFacultyAgent → faculty 分类
```

## 5. 核心组件

### 5.1 BaseAgent

Agent 基类，所有 Agent 继承它：
- OpenAI 兼容 client（`config/app_config.json`）
- 内存缓存（`self._memory_cache`）+ 文件持久化缓存
- 自动重试（超时 + 网络错误）
- JSON 清理 + 修复（`_clean_json_content` / `_repair_json_once`）
- Token 估算

### 5.2 StudentContext

`context.py` — 全链路共享上下文 dataclass：
- **stage**：`lead_in | match | application`
- **前期字段**：`raw_input`、`extracted_background`、`quick_assessment`、`suggested_questions`
- **中期字段**：预测结果、背景信息、AI 解释
- **后期字段**：申请计划（待扩展）
- **audit**：`history` 列表记录每个 Agent 的调用

<<<<<<< HEAD
### 5.3 AgentRegistry（懒加载注册中心）

`registry.py` — Agent 工厂注册 + 懒实例化：

```python
AgentRegistry.register("lead_in", LeadInAgent)   # 注册类，不实例化
agent = AgentRegistry.get("lead_in")              # 首次调用时才 __init__
```

- 所有 Agent 在模块 import 时以**工厂函数**注册（非实例），import 开销 O(1)
- 首次 `get(name)` 触发实例化并缓存，后续命中缓存
- `list()` / `clear()` 辅助调试

### 5.4 AgentOrchestrator（单Agent + Pipeline 编排）

`orchestrator.py` — 两种调用模式：

**单 Agent 派发**：
```python
result = AgentOrchestrator.run("lead_in", context, user_input=text)
```

**Pipeline 链式编排**：
```python
steps = [
    {"agent": "lead_in", "kwargs": {"user_input": text}},
    {"agent": "form_validation", "kwargs": {"form_data": data}},
    {"agent": "explain", "kwargs": {}},
]
results = AgentOrchestrator.run_pipeline(steps, context)
```

Pipeline 通过共享 `StudentContext` 传递状态，任一步返回 `_error` 则提前终止。

### 5.5 数据契约（schemas.py）

`schemas.py` 提供 TypedDict 类型定义，替代裸 `dict[str, Any]`：

- `ExtractedBackground` — LeadInAgent 输出的结构化背景字段
- `LeadInResult` — LeadInAgent.run() 返回类型
- `PipelineStep` — `run_pipeline()` 的步骤定义

用于 IDE 自动补全、类型检查、以及作为 Agent 间数据契约的文档。

### 5.6 错误处理约定

所有 Agent 的 `run()` 返回 `dict` 中统一使用 `_error` 键标记失败：

| Agent | `_error` 值 | 触发条件 |
|-------|------------|---------|
| LeadInAgent | `"api_failed"` | API 调用失败 |
| ExplainAgent | `"api_failed"` | API 调用失败 |
| FormValidationAgent | `"api_failed"` | API 调用失败 |
| ApplicationAgent | `"api_failed"` / `"no_results"` | API 失败 / 无预测结果 |
| BoundaryCaseAgent | 通过 `api_errors` 计数 | 批量 API 调用中部分失败 |
| TextPreprocessingAgent | `"api_failed_or_invalid"` | API 失败或内容无效 |
| BackgroundFacultyAgent | `"no_faculties_resolved"` | 有输入但未能解析出学部 |

`run_pipeline()` 检测到 `_error` 后立即终止后续步骤。

### 5.7 LeadInAgent
=======
### 5.3 AgentRegistry + AgentOrchestrator

- **AgentRegistry**：`register(name, agent)` / `get(name)` — dict wrapper
- **AgentOrchestrator**：`run(agent_name, context, **kwargs)` → 取 Agent → 调用 → 写回 context

### 5.4 LeadInAgent（新）
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

前期 NLU Agent：
- 输入：顾问自由文本 + StudentContext
- 输出：`extracted_info`（JSON）+ `quick_assessment`（自然语言）+ `suggested_questions`
- 已集成到 `pages/hk.py`，通过 `render_lead_in_panel()`

<<<<<<< HEAD
### 5.8 ExplainAgent
=======
### 5.5 ExplainAgent（新）
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

预测解释 Agent：
- 输入：StudentContext（含 prediction_results）
- 输出：整体评估、推荐理由、优势/风险、总结建议
- 已集成到 `content_display.py`，预测完成后自动展示

<<<<<<< HEAD
### 5.9 已有 Agent
=======
### 5.6 已有 Agent
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

| Agent | 用途 |
|-------|------|
| BoundaryCaseAgent | 相似/跨专业推荐平衡决策 |
<<<<<<< HEAD
| FormValidationAgent | 表单数据合理性校验（含规则快速通道） |
| TextPreprocessingAgent | 背提文本质量评估 |
| BackgroundFacultyAgent | 背景专业 → 学部推断 |
| ApplicationAgent | 申请策略生成 |
=======
| TextPreprocessingAgent | 背提文本质量评估 |
| BackgroundFacultyAgent | 背景专业 → 学部推断 |
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

## 6. 数据流

```
config/app_config.json → BaseAgent → OpenAI client
    │
    ▼
Agent.run(StudentContext)
    ├── _build_prompt (子类实现)
    ├── _call_api (BaseAgent: 缓存 + 重试)
    ├── _parse_response (子类实现)
    └── context.record() (审计追踪)
```

## 7. 依赖

- `requests`（HTTP client）
- [Config](config/README.md) — `app_config.json`（API Key、Base URL、Model）
- [Utils](src/utils/README.md) — `load_app_config()`、`setup_logger()`
