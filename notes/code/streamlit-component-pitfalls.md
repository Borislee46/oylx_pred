# Streamlit 自定义组件踩坑记录

## 架构认知

**Streamlit 是 Python 驱动的服务端渲染框架，不是前后端分离架构。**

每一个页面交互（点击、输入、选单）都会触发 Python 脚本的完整重执行，然后 Streamlit 对比新旧 DOM 树，把差异 patch 到浏览器。这意味着：

- 整个页面是 Python 执行流的**副作用产物**——不存在"前端独立状态"
- 插入 React / Vue / 原生 JS 组件时，**必须显式处理双向通信**，框架不会帮你做
- 自定义组件的 iframe 在每次 Python 重跑时都会被重建——持久化状态只能存在 `st.session_state` 里

所以接入任何自定义前端框架时，核心问题只有一个：**你的 JS 数据和 Python 执行流怎么对齐？** `declare_component` 是 Streamlit 为这个问题提供的唯一正式通道。

## 核心结论

**任何需要 JS → Python 双向通信的组件，必须用 `st.components.v1.declare_component()`。**

两个常见的错误替代品：

- `st.components.v1.html()` — **已在 1.56.0 标记为 deprecated**，且被动重跑时返回值不稳定
- `st.html()` — 官方推荐的静态 HTML 替代品，但 JavaScript 默认被忽略（除非显式开 `unsafe_allow_javascript=True`），不能替代交互型组件

## `st.html()` vs `st.components.v1.html()` vs `declare_component()`

| | `st.html()` | `st.components.v1.html()` | `declare_component()` |
|---|---|---|---|
| 状态 | 官方推荐（1.56+） | **已弃用（1.56.0）**，将移除 | 官方自定义组件 API |
| JS 执行 | 默认禁用 | 不受限 | 不受限 |
| iframe 隔离 | 无（内联渲染） | 有（sandboxed iframe） | 有（sandboxed iframe） |
| 双向通信 | 不支持 | 不稳定（被动重跑返回 None） | **正式支持，始终送达** |
| 适合 | 静态样式、footer、badge | 不应再使用 | 交互型组件、编辑器、自定义输入 |

### 典型故障模式

```
用户输入 → JS setComponentValue → 值存入前端
    ↓
用户点击 st.button（外部 widget）→ Streamlit 全页重跑（被动）
    ↓
st.components.v1.html() 渲染 → 返回 None（值丢了！）
    ↓
session_state 拿不到文本 → AI 按钮"无效"
```

### 正确链路

```
用户输入 → JS setComponentValue → declare_component 双向通道
    ↓
用户点击（无论组件内 button 还是外部 widget）
    ↓
declare_component 始终返回最新值 ✅
```

## 官方 Custom Component 限制

来自 Streamlit 官方文档：
- 每个组件运行在**独立 sandboxed iframe** 中
- **不能与其他组件通信**——所以不能用组件来构建 grid layout 之类的东西
- **不能修改外部 CSS**——不能通过组件实现 dark mode 切换等全局样式变更
- **不能增删 Streamlit 元素**——不能通过组件隐藏菜单栏等
- **没有自动 debounce**——高频 `setComponentValue` 需要开发者自己在 JS 侧做频率控制

## 自定义框架（React 等）接入要点

在 Streamlit 里嵌入 React/Vue/Svelte 等框架时，除了用 `declare_component` 承载外，还有几个关键点：

1. **构建产物挂载**：把 React 构建产物（`build/`）放到 `frontend/` 目录下，`index.html` 负责加载 JS bundle。`declare_component` 的 `path` 参数指向这个目录。

2. **状态同步**：React 的 `useState` 在 Streamlit 场景下是**临时状态**——Python 重跑时 iframe 会被重建，所有 React 内部状态丢失。真正持久的状态必须通过 `sendToStreamlit` → Python `st.session_state` → 下次渲染时通过 `config` 传回 React。这就是"双向传递"的完整闭环。

3. **单向数据流**：推荐的模式是：

   ```
   React onChange → debounce → sendToStreamlit(value)
       ↓
   Python 重跑 → st.session_state 更新
       ↓
   下次渲染 → config.initial_value → React 恢复状态
   ```

   不要试图在 React 里维护"权威状态"——**st.session_state 是唯一的权威**。

4. **性能**：Python 每次重跑都会重建整个 iframe。如果 React 组件初始化很重（大量 DOM、网络请求），考虑在 JS 侧用 `localStorage` 做缓存，在 `init()` 阶段快速恢复。

5. **事件频率控制**：Streamlit 每次 Python 重跑有开销（~100ms+）。高频事件（`onChange`、`onMouseMove`）不能每次都 `sendToStreamlit`，必须在前端做 debounce/throttle，只在"完成"时同步（blur、提交按钮）。

## `declare_component` 正确写法

### Python 侧

```python
import streamlit as st

_component = st.components.v1.declare_component(
    "my_component",
    path=str(Path(__file__).parent / "frontend"),  # index.html 所在目录
)

def my_component(config: dict, key: str | None = None):
    result = _component(config=config, default=None, key=key)
    # result 是 JS setComponentValue 发来的值，始终可靠
    if isinstance(result, dict):
        return result
    return None
```

### JS 侧 (`frontend/index.html`)

`postMessage` **必须**包含 `isStreamlitMessage: true`：

```javascript
function sendToStreamlit(value) {
  window.parent.postMessage({
    isStreamlitMessage: true,           // 必须！
    type: "streamlit:setComponentValue",
    value: value,
    dataType: "json"
  }, "*");
}

function setComponentReady() {          // 必须！
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:componentReady",
    apiVersion: 1
  }, "*");
}

function setFrameHeight(height) {       // 推荐
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setFrameHeight",
    height: height || document.documentElement.scrollHeight
  }, "*");
}
```

## 事件去重

`declare_component` 可能因 Streamlit 内部机制重复触发。用 `event_id` 去重：

```javascript
// JS 侧
let _streamlitEventSeq = 0;
function sendToStreamlit(value) {
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setComponentValue",
    value: { ...value, event_id: sessionId + ":" + (++_streamlitEventSeq) },
    dataType: "json"
  }, "*");
}
```

```python
# Python 侧
if isinstance(result, dict):
    event_id = str(result.get("event_id") or "")
    if event_id:
        last_key = f"_my_component_last_event:{key or 'default'}"
        if st.session_state.get(last_key) == event_id:
            return ""           # 已处理过，跳过
        st.session_state[last_key] = event_id
```

## `@st.fragment` 的坑

1. **不能定义在嵌套函数内部**——每次父函数执行创建新函数对象，Streamlit 无法识别为同一 fragment，退化为全页重跑。
2. **fragment 重跑是"被动重跑"**——对 `st.components.v1.html()` 来说，fragment 重跑和全页重跑没区别，组件返回值都丢。对 `declare_component` 来说 fragment 重跑也是被动重跑，但因为组件通信机制不同，declare_component 不受影响。
3. **即使 fragment 正常工作**，它只是隔离了页面其他区域的渲染，不解决组件通信问题。
4. **Widget 只能放在 fragment 的主体内**——官方明确：fragment 不能把 widget 渲染到外部创建的 container 里（比如 `st.container()` 或 `st.columns()` 创建的容器）。widget 必须在 fragment 的顶层直接调用。
5. **调用 `st.sidebar` 在 fragment 内不支持**——需要 sidebar 的话，在外部用 `with st.sidebar:` 包裹 fragment 调用。

官方 fragment 行为要点（来自 Streamlit 文档）：
- 调用 `st.rerun()` 从 fragment 内部触发 app 重跑
- 调用 `st.rerun(scope="fragment")` 仅重跑 fragment 自身
- fragment 与外部共享 `st.session_state`——这是 fragment 内部数据的唯一传出方式
- fragment 元素在每次 fragment 重跑时会被清除并重绘，其余 app 保持不变

正确用法：fragment 定义在模块层，且只包裹不依赖组件返回值的纯 Streamlit widget。

## 排查清单

碰到"按钮点击无反应"时，按顺序排查：

1. **日志里有 `COMPONENT_RAW | result_type=NoneType`？** → 组件返回值丢了，换 `declare_component`
2. **日志里有 `result_type=dict` 但 `text` 为空？** → JS 侧 `setComponentValue` 没发对，检查 `postMessage` 格式
3. **`analyze=True` 但 `raw=""`？** → 外部 widget 触发的被动重跑，组件值未送达
4. **页面日志反复出现"页面加载"？** → fragment 没生效（嵌套定义 or 版本问题）
5. **AI 调用卡住？** → 实际在等 API 超时（15-45s），不是在"卡死"，加个 timeout 提示

## `st.cache_data` vs `st.cache_resource`

| | `st.cache_data` | `st.cache_resource` |
|---|---|---|
| 用途 | 可序列化的数据（DataFrame、array、list、str、int） | 不可序列化的全局资源（ML 模型、DB 连接） |
| 返回值 | 每次调用创建**副本**（防 mutation） | 直接返回缓存对象（所有 session 共享同一实例） |
| 线程安全 | ✅（副本隔离） | ⚠️ 需自行处理并发 |
| TTL | 通过 `ttl` 参数控制 | 通过 `ttl` 参数控制 |
| 最大条目 | `max_entries`（LRU 淘汰） | `max_entries`（LRU 淘汰） |
| 典型用法 | `@st.cache_data(ttl=600)` 加载数据文件 | `@st.cache_resource` 加载 XGBoost 模型 |

关键行为：
- 缓存 key 由**函数参数值 + 函数代码**共同决定——改代码 = 自动刷新缓存
- 开发模式下，源码变更自动 invalidate 缓存
- 不确定用哪个时，先用 `st.cache_data`——它适用绝大多数场景

## 适用场景速查

| 场景 | 用什么 |
|------|--------|
| 纯展示 HTML（footer、badge、分割线） | `st.html()` |
| 静态样式注入 | `st.html(f"<style>{css}</style>")` |
| 自定义表单输入、编辑器、富文本 | `declare_component` |
| 内嵌复杂 JS 交互（React、图表、地图、代码编辑器） | `declare_component` |
| 加载数据文件（CSV、Feather、Parquet） | `@st.cache_data` |
| 加载 ML 模型、DB 连接 | `@st.cache_resource` |
| Streamlit 原生 widget 够用 | 直接用 `st.button`、`st.text_area` 等，别折腾 |
