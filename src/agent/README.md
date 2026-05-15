# Agent 模块技术文档

## 1. 模块概述

`src/agent` 是 LLM Agent 系统，基于 DeepSeek（OpenAI 兼容 API）。采用 Shared Context → Agent Registry → Orchestrator 三层架构，覆盖留学全链路：前期 NLU（碎片信息提取）→ 中期决策（边界 case + 预测解释）→ 后期（申请策略）。不依赖 LangChain，所有 Agent 继承 `BaseAgent`。

## DS 视角

LLM Agent 系统的工程指标很完善——4 级 JSON 修复覆盖 99.8% 的格式错误、缓存命中率、延迟追踪。但核心 DS 问题不是"JSON 能不能成功解析"，而是**Agent 做的决策对不对**。LeadInAgent 从碎片文本提取的结构化背景准确吗？ExplainAgent 的 profile 分类会不会把强学生分到 weak 组？BlindEvalAgent 作为独立 benchmark，但语言分数 prompt bug（0.72 被解读为 IELTS 0.72）已经证明 prompt 对数值输入的脆弱性——类似的维度理解偏差可能还有。

评估体系当前是纯工程指标（延迟、parse 成功率）加一次对比研究（40 样本，Pearson r=0.44），缺少持续的 Agent 决策质量评估。`eval/` 框架已搭好但还没灌真实数据。Agent 输出会级联影响下游——LeadIn 提取错了 → 表单自动填充出错 → XGBoost 输入偏差 → ExplainAgent 基于错误输入做解读。

## 2. 目录结构

```
agent/
├── __init__.py                     # 公共 API（24 项导出 + 集中注册）
├── base_agent.py                   # BaseAgent 基类（client、缓存、重试、JSON 修复、token 追踪）
├── context.py                      # StudentContext（全链路共享上下文）
├── registry.py                     # AgentRegistry（工厂注册，每次 get() 新建实例）
├── orchestrator.py                 # AgentOrchestrator（单 Agent + Pipeline 编排）
├── schemas.py                      # TypedDict 数据契约（12 个类型定义）
├── utils.py                        # Agent 共享工具（truncate、parse_bool 等）
├── form_bridge.py                  # Agent → 表单桥接（5层 fuzzy match + school_alias_resolver + 学位后缀剥离）
├── lead_in_agent.py                # 前期 NLU Agent
├── lead_in_prompts.py              # LeadIn Prompt 模板
├── explain_agent.py                # ExplainAgent（流式 + 同步双路径）
├── explain_profiles.py             # 4 种 Profile System Prompt（strong_elite/medium_mixed/weak_gaps/cross_major）
├── boundary_case_agent.py          # 边界案例决策 Agent
├── boundary_case_prompts.py        # 边界案例 Prompt 模板
├── text_preprocessing_agent.py     # 背提文本校验 Agent（含批量模式：单次 API 调用校验 4 字段）
├── text_preprocessing_prompts.py   # 文本校验 Prompt 模板（单字段 + 批量）
├── background_faculty_agent.py     # 背景学部推断 Agent（类级别 MD5 文件缓存）
├── background_faculty_prompts.py   # 学部推断 Prompt 模板
├── application_agent.py            # 申请策略生成 Agent（含 timeline + action_items）
├── application_prompts.py          # 申请策略 Prompt 模板
├── blind_eval_agent.py             # AI 专家盲评 Agent（独立概率评估，用于模型 benchmark）
├── blind_eval_prompts.py           # 盲评 Prompt 模板
├── pdf_agent.py                    # PDF 报告 AI 分析 Agent（DeepSeek 驱动文案生成）
├── pdf_prompts.py                  # PDF 分析 Prompt 模板
└── eval/                           # Agent 评估框架（F1 / coverage / regression test）
```

## 3. 架构

```
┌──────────────────────────────────────────────────────┐
│                AgentOrchestrator                      │
│                   (编排路由)                           │
├──────────────────────────────────────────────────────┤
│                AgentRegistry                          │
│  lead_in | explain | boundary_case | text_preprocessing | … │
├──────────┬──────────┬──────────┬──────────────────────────┤
│ LeadIn   │ Explain  │ Boundary │ TextPrep │ …            │
│ Agent    │ Agent    │ Agent    │ Agent    │              │
└────┬─────┴────┬─────┴──────────┴──────────┴──────────────┘
     │          │
     ▼          ▼
┌──────────────────────────────────────────────────────┐
│                StudentContext                         │
│  raw_input → extracted_background → prediction_      │
│  results → ai_explanation                             │
│  + history（审计追踪）                                │
└──────────────────────────────────────────────────────┘
```

## 4. 端到端流程

### 主流程

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
ExplainAgent.stream(ctx)
    ├── 流式 LLM 生成（25ms/6字双阈值节流）
    ├── 逐段揭示：overview → strengths → concerns → summary
    └── 输出 {overview, strengths, concerns, summary, school_notes, products}
    │
    ▼
ApplicationAgent.run(StudentContext)
    └── 申请策略 + 行动建议
```

### 辅助 Agent 管线

```
BoundaryCaseAgent.evaluate_boundary_cases()
    ├── 输入：sim/cross 推荐列表 + 用户背景
    ├── 决策：relax / tighten
    └── 输出：调整后的推荐平衡

TextPreprocessingAgent → has_meaningful_experience_text 判定
BackgroundFacultyAgent → faculty 分类
ApplicationAgent → 申请策略 + 行动建议
```

### 4.1 关键问题

- **LeadInAgent 提取准确率从未系统评估**：碎片文本→结构化背景是 Agent 链路的第一步，提取错了整个下游都偏。没有 baseline accuracy 测量。
- **ExplainAgent 的 `classify_profile()` 分四档——是数据驱动还是手调的？** strong_elite/medium_mixed/weak_gaps/cross_major 的分界阈值（avg_prob ≥0.55、cross ≥40% 等）有没有验证过分类准确率？
- **4 级 JSON 修复修不了语义错误**：格式错误的 JSON 能修复（99.8%），但格式正确内容错误的 JSON（比如 probability 字段输出了一个超出 0-1 的值）不会被检测到。
- **BlindEvalAgent 置信度偏差**：LLM 自评 confidence 100% 为 medium/high，从未自评 low——即使语言分数被误读时也自信满满。这表明 LLM 的 confidence 不适合作为可信度参考。

## 5. 核心组件

### 5.1 BaseAgent

Agent 基类，所有 Agent 继承它：
- OpenAI 兼容 client（`config/app_config.json`）
- 内存 LRU 缓存（`self._memory_cache`，128 条，1h TTL）+ 类级别文件持久化缓存
- 自动重试（默认 3 次，1s 间隔；流式 2 次）
- 四级 JSON 容错：`json.loads` → 轻量正则修复（<1ms）→ `json_repair` 库（<5ms）→ LLM 修复 API（~9s 兜底）
- 简化的 API 消息格式（`{"role": "user", "content": prompt}`，无嵌套 content 数组）
- 真实 Token 用量追踪（`self.last_usage` 存储 API 返回的 `prompt_tokens`/`completion_tokens`）

### 5.2 StudentContext

`context.py` — 全链路共享上下文 dataclass：
- **stage**：`lead_in | match | application`
- **前期字段**：`raw_input`、`extracted_background`、`quick_assessment`、`suggested_questions`
- **中期字段**：预测结果、背景信息、AI 解释
- **后期字段**：申请策略与行动建议（ApplicationAgent）
- **audit**：`history` 列表记录每个 Agent 的调用

### 5.3 AgentRegistry（工厂注册中心）

`registry.py` — Agent 工厂注册，每次 `get()` 创建新实例：

```python
AgentRegistry.register("lead_in", LeadInAgent)   # 注册工厂，不实例化
agent = AgentRegistry.get("lead_in")              # 每次调用都 __init__，新建实例
```

- 所有 Agent 在模块 import 时以**工厂函数**注册（非实例），import 开销 O(1)
- 每次 `get(name)` 创建新实例，保证 `_memory_cache` 和 `_session` 隔离
- 上游调用方（如 `experience_text_validator.py`）通过 `@cache_resource` 在 Streamlit 级别缓存 Agent 实例
- `list()` / `clear()` 辅助调试

### 5.4 AgentOrchestrator（单 Agent 编排）

`orchestrator.py` — 统一调用入口：

```python
result = AgentOrchestrator.run("lead_in", context, user_input=text)
```

通过共享 `StudentContext` 传递状态，结果通过 `context.record()` 写入审计追踪。

### 5.5 数据契约（schemas.py）

`schemas.py` 提供 TypedDict 类型定义：

- `ExtractedBackground` — LeadInAgent 输出的结构化背景字段
- `LeadInResult` — LeadInAgent.run() 返回类型
- `ExplainResult` — ExplainAgent 输出 schema

### 5.6 错误处理约定

所有 Agent 的 `run()` 返回 `dict` 中统一使用 `_error` 键标记失败：

| Agent | `_error` 值 | 触发条件 |
|-------|------------|---------|
| LeadInAgent | `"api_failed"` | API 调用失败 |
| ExplainAgent | `"api_failed"` | API 调用失败 |
| ApplicationAgent | `"api_failed"` / `"no_results"` | API 失败 / 无预测结果 |
| BoundaryCaseAgent | 通过 `api_errors` 计数 | 批量 API 调用中部分失败 |
| TextPreprocessingAgent | `"api_failed_or_invalid"` | API 失败或内容无效 |
| BackgroundFacultyAgent | `"no_faculties_resolved"` | 有输入但未能解析出学部 |

所有 Agent 通过 `AgentOrchestrator.run()` 统一调用，异常由 orchestrator 统一捕获。

### 5.7 LeadInAgent

前期 NLU Agent：
- 输入：顾问自由文本 + StudentContext
- 输出：`extracted_info`（JSON）+ `quick_assessment`（自然语言）+ `suggested_questions`
- **多轮对话支持**（2026-05）：顾问可对提取结果逐字段确认/修正，Agent 根据反馈重新提取，支持追问补充信息
- **动态院校约束**（2026-05）：`_build_system_prompt()` 从 `config/prediction_rules.json` 读取 `TARGET_COUNTRY_UNIVERSITY_MAP`，注入可用地区、院校列表。已集成到 `pages/hk.py`
- **特殊别名**（2026-05）：Prompt 已包含 37 个院校/专业缩写（北大/港大/HKU/NUS/CS/金融…），遇到 "985"/"211"/"港3" 等类别别名保留原值，由 `form_bridge` + `school_alias_resolver` 展开（`major_name_mapping.json` 处理专业别名归一化）

### 5.8 form_bridge — Agent → 表单桥接

`form_bridge.py` 将 LeadInAgent 提取的结构化背景映射到 Streamlit 表单 widget state。

**5 层模糊匹配**（`_fuzzy_match`）：
```
exact → substring → rapidfuzz partial_ratio → char-order → alias map
```

- `rapidfuzz` 替代 `difflib`：`partial_ratio` 做子串匹配，CJK 感知更好
- `_chars_in_order`：中文缩写匹配（"港大" → "香港大学"）
- `_UNIVERSITY_ALIAS_MAP`：英文缩写兜底（NUS/HKU/NTU...）
- **院校别名解析**（2026-05）：集成 `school_alias_resolver`。背景院校 "985"/"211"/"双一流" → 自动解析为该类别在 cases 中频次最高的学校；目标院校 "港3"/"港5"/"港8" → 展开为对应排名区间的港校列表。LeadInAgent prompt 中遇到类别别名保留原值，由 form_bridge + school_alias_resolver 展开

**目标专业模糊匹配**（`_fuzzy_match_major`）：
```
学位后缀剥离（硕士/博士/MSc...） → alias map（CS→Computer Science） → partial_ratio
```
- 50+ 专业别名映射（CS/金融/EE/统计/法律...）
- 自动加载 `school_major_details.feather` 全部英文专业名作为候选集
- **专业名映射**（2026-05）：集成 `config/major_name_mapping.json`，将 LLM 提取的专业别名归一化到标准专业名

### 5.9 ExplainAgent

预测解释 Agent：
- 输入：StudentContext（含 prediction_results、matched_products）
- 输出：`{"overview","strengths","concerns","summary","school_notes","products"}` JSON
- **双路径**：`stream()`（流式 yield chunk）和 `run()`（同步阻塞），流式失败时自动降级到同步
- **Profile 路由**：`classify_profile()` 根据预测结果特征（概率均值、惩罚类型、跨专业比例）选择 4 种 System Prompt 之一
- **Prompt 构建**：`_build_explain_prompt()` 将学生背景 + 预测结果 + 惩罚追踪 + 产品列表序列化为 LLM 输入
- 已集成到 `content_display.py:_render_ai_explanation()`，流式逐段揭示 + 卡片末尾 "AI解读中..." 脉冲动画

### 5.10 已有 Agent

| Agent | 用途 | 关键特性 |
|-------|------|----------|
| BoundaryCaseAgent | 相似/跨专业推荐平衡决策 | 类级别文件缓存、分块并行 |
| TextPreprocessingAgent | 背提文本质量评估 | **批量模式**：`validate_fields_batch()` 单次 API 调用校验 4 个字段 |
| BackgroundFacultyAgent | 背景专业 → 学部推断 | 类级别 MD5 文件缓存 |
| ApplicationAgent | 申请策略生成 | 全量背景 + 预测结果合并 |

**TextPreprocessingAgent 批量模式：**

```python
agent = TextPreprocessingAgent()
result = agent.validate_fields_batch({
    "research_details": "...",
    "award_details": "...",
    "internship_details": "...",
    "paper_details": "...",
})
# result: {"research_details": True, "award_details": False, ...}
```

## 6. 数据流

```
config/app_config.json → BaseAgent → DeepSeek API
    │
    ▼
Agent.run(StudentContext) / Agent.stream(StudentContext)
    ├── _build_prompt (子类实现)
    ├── _call_api (BaseAgent: 缓存 + 重试)
    │     ├── 内存缓存命中 → 直接返回
    │     ├── API 调用（扁平消息格式）
    │     └── 存储真实 token 用量到 self.last_usage
    ├── _parse_response (子类实现)
    │     └── 失败则走四级 JSON 修复
    └── context.record() (审计追踪)
```

## DS Known Issues

- **Agent 评估只有工程指标，没有决策质量指标**：延迟、parse 成功率都有追踪，但 Agent 的判断对不对——没人知道。`eval/` 框架设计了 F1/regression test 但还没数据。LeadInAgent 的提取准确率从未系统测量过。
- **Prompt 脆弱性**：BlindEvalAgent 的语言分数 bug（已修）说明数值字段在 prompt 中容易产生维度误读。类似问题可能还存在于 GPA 等其他数值字段的 prompt 表示中。LLM 自评 confidence 100% 为 medium/high，从未自评 low——confidence 不适合做可信度参考。
- **误差级联**：LeadIn → 表单自动填充 → XGBoost → ExplainAgent，整条链路没有确认检查点。LeadIn 提取错了背景，后面的所有决策都基于错误输入。
- **没有 A/B 测试框架**：prompt 改动后的效果评估全靠手动跑几组 case 看感觉，没有系统化的对照组对比。
- **缓存 key 敏感**：ExplainAgent 缓存用 MD5 of 原始输入——多一个空格就 cache miss，不必要的 API 重复调用。
- **BlindEvalAgent 40 样本 Pearson r=0.44**：LLM 和模型在"谁更容易录取"上排序一致性偏弱，分歧大——目前无法判断是模型偏还是 LLM 偏。

## 7. 依赖

- `requests`（HTTP client）
- [Config](config/README.md) — `app_config.json`（API Key、Base URL、Model）
- [Utils](src/utils/README.md) — `load_app_config()`、`setup_logger()`
