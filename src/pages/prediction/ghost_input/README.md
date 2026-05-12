# Ghost Input — Cursor-style inline autocomplete

## 1. 模块概述

`ghost_input` 是 Streamlit 自定义组件，为留学申请背景填写提供 Cursor 风格的灰字补全。基于 DeepSeek Prefix Continuation API，浏览器端直接调 API，无 Streamlit round-trip。Tab 接受，Esc 取消，分段逐出。

**核心理念**：缓存优先 + 规则兜底 + LLM 冷启动。热路径 <1ms 零 token，冷路径 ~500ms。

## 2. 目录结构

```
ghost_input/
├── __init__.py          # Streamlit component 声明 + 配置注入 + 遥测日志
├── frontend/
│   └── index.html       # 全部前端逻辑 (CSS + JS, 单文件)
└── README.md
```

## 3. 端到端流程

```
用户在 textarea 输入
    │
    ▼ debounce (120-450ms, 自适应)
    │
doFetch(text)
    │
    ├── Layer 1: localStorage 缓存 (精确 + 前缀回退)
    │   └── 命中 → 显示灰字 <1ms, 0 token
    │
    ├── Layer 2: 规则引擎 (POSTFIX_RULES)
    │   └── GPA/雅思/一段…等 10 个触发词 → 显示灰字 <1ms, 0 token
    │
    └── Layer 3: DeepSeek Prefix API
        ├── system prompt: 动态注入允许院校列表 (来自 prediction_rules.json)
        ├── 多级过滤: 禁区/黑名单/领域白名单/去重
        └── 命中 → 缓存 + 分段入队显示 (~500ms)
```

## 4. 核心组件

### 4.1 Python 端 (`__init__.py`)

| 组件 | 说明 |
|------|------|
| `ghost_text_area()` | Streamlit 组件入口，传递 config |
| `_load_allowed_universities()` | 从 `prediction_rules.json` 读取 23 所 + 别名，注入前端 system prompt |
| `_log_telemetry()` | 14 个计数器日志 + 错误率告警 (>50%) |

### 4.2 规则引擎 (JS)

| 规则 | 触发词 | 补全 |
|------|--------|------|
| 分数 | GPA / 均分 / 雅思 / 托福 / GRE / GMAT | 常用值 |
| 计数 | 一段 / 两段 / 一篇 | 科研 / 实习 / 论文 |

### 4.3 质量过滤链 (JS)

```
completion
  → NONSENSE_RE (纯标点/数字)
  → AGE_RE (年龄相关)
  → CONTACT_RE (联系方式)
  → POLITICAL_RE (政治)
  → _FORBIDDEN_REGIONS_RE (美国/英国/澳洲…)
  → meaningful check (有意义字符 ≥2)
  → duplicate check (已存在于输入文本)
  → DOMAIN_SIGNALS_RE (至少一个教育领域词)
```

### 4.4 分段队列

长补全按空格拆分为段，一次只显示一段。用户 Tab 接受当前段后自动出下一段，Esc 清空全部。目的：让用户掌控接受节奏，不一气全出。

### 4.5 遥测计数器 (14 项)

| 计数器 | 含义 |
|--------|------|
| `fetch_attempt` / `fetch_ok` / `fetch_fail` / `fetch_retry` | API 调用统计 |
| `cache_hit` / `cache_set` | 缓存命中/写入 |
| `rule_hit` / `rule_miss` | 规则引擎命中/未命中 |
| `suggestion_shown` / `suggestion_accepted` / `suggestion_dismissed` | 灰字交互 |
| `rate_limited` / `dedup_blocked` | 频控/去重拦截 |

## 5. 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `api_model` | deepseek-v4-flash | 前缀补全模型 |
| `rate_limit_max` | 30 | 每分钟最多 API 调用 |
| `rate_limit_window_seconds` | 60 | 滑动窗口 |
| `rate_limit_cooldown_seconds` | 15 | 超限冷却 |
| `allowed_universities` | prediction_rules.json | 系统支持院校 + 别名 |
| `allowed_regions` | 香港/新加坡/澳门/马来西亚 | 允许地区 |

## 6. 依赖

- [Config](../../../config/README.md) — `prediction_rules.json`（院校列表）
- Streamlit Custom Component (`st.components.v1.declare_component`)
- DeepSeek Chat Completions API (Beta, prefix continuation)
