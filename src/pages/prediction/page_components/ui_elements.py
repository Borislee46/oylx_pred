"""
HK 页面 UI 原子组件。

职责：渲染页面级别的 UI 元素（header、footer、反馈、思考气泡、lead-in）。
这些组件不含业务逻辑，只负责渲染。复杂组件（如 ghost_input、form）在其他模块中。

组件分类：
  - 品牌层：logo → base64, header（双盾牌效果）, footer
  - 交互层：feedback（👍👎）, back_to_homepage
  - 进度层：thought_bubble（静态）, thought_bubble_with_wait_pulse（动画）
  - Lead-In 层：ghost textarea + AI 提取按钮 + 字段 chip 展示
"""

import base64
import html
from pathlib import Path

import streamlit as st

from src.agent.context import StudentContext          # Agent 输入数据模型
from src.agent.form_bridge import apply_lead_in_to_form  # Agent 输出 → 表单 widget state 桥接
from src.agent.orchestrator import AgentOrchestrator  # Agent 编排器（lead_in / explain）
from src.pages.prediction.ghost_input import ghost_text_area  # 浏览器端 DeepSeek 自动补全组件
from src.utils.env_config_loader import load_app_config  # 读取 OPEN_AI_API_KEY
from src.utils.logger import setup_logger
from src.utils.ui.hk_shield_v2 import mount_hk_shield_v2  # 双盾牌 CSS/JS 效果

logger = setup_logger("page3", "prediction")

# DeepSeek API key 的模块级缓存（从 config 读取一次，避免重复 IO）
_api_key_cache: str | None = None


def _get_ghost_api_key() -> str:
    """延迟加载 DeepSeek API key（从 app_config.json 读取，缓存到模块变量）。"""
    global _api_key_cache
    if _api_key_cache is not None:
        return _api_key_cache
    try:
        cfg = load_app_config()
        _api_key_cache = cfg.get("OPEN_AI_API_KEY", "")
    except Exception:
        _api_key_cache = ""
    return _api_key_cache


def _get_project_root() -> Path:
    """从当前文件向上遍历，通过 pyproject.toml 定位项目根目录。

    为什么不用 __file__ 写死相对路径？
      部署时 CWD 不可控，pyproject.toml 是稳定的锚点。
    """
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[4]  # fallback


# ── Logo 加载（带 Streamlit 缓存）─────────────────────────
@st.cache_data(show_spinner=False)
def get_product_logo_image_as_base64(path: str) -> str:
    """读取图片文件并转为 base64 字符串（data URI 格式）。

    @st.cache_data 确保文件只读一次，后续重跑直接返回缓存值。
    适用于所有页面的 logo/shield 图片加载。
    """
    full_path = _get_project_root() / path
    data = full_path.read_bytes()
    if not data:
        return ""
    return base64.b64encode(data).decode()


# ── 品牌 Header ────────────────────────────────────────────
def render_header(logo_base64: str) -> None:
    """渲染页面顶部品牌 header：双盾牌 logo + 标题 + 分割线。

    双盾牌效果（mount_hk_shield_v2）：
      - 底层：金属纹理（shield_metal.png）作为 mask
      - 上层：产品 logo（product_logo.png）
      - CSS 层叠实现金属质感 + 光晕动画
    """
    mask_base64 = get_product_logo_image_as_base64("assets/shield_metal.png")
    metal_base64 = get_product_logo_image_as_base64("assets/shield_mask.png")

    if not logo_base64:
        st.title("Signals")
        return

    mount_hk_shield_v2()  # 注入双盾牌 CSS + JS

    html_block = f"""
        <div class="hk-header">
            <div class="hk-logo-container"
                 style="--logo-mask-url: url(data:image/png;base64,{mask_base64}); --metal-tex-url: url(data:image/png;base64,{metal_base64})">
                <div class="hk-metal-layer"></div>
                <img class="hk-header-logo" src="data:image/png;base64,{logo_base64}" alt="logo">
            </div>
            <div>
                <p class="hk-header-title">Signals</p>
                <p class="hk-header-subtitle">
                    <span class="hk-header-subtitle-en">Admission Intelligence</span>
                    <span class="hk-header-subtitle-zh">留学择校系统</span>
                </p>
            </div>
        </div>
        <hr class="hk-header-divider">
    """
    st.html(html_block)


# ── 用户反馈区 ─────────────────────────────────────────────
def display_feedback_section(session_id: str) -> None:
    """渲染 👍👎 反馈按钮。

    去重机制：toast_key 记录上次发送的反馈值，相同值不重复弹 toast。
    Streamlit 的 st.feedback 组件在每次重跑时可能返回已存储的值，
    用 toast_key 比较避免用户刷新页面时反复弹 toast。
    """
    key = f"feedback_{session_id}"
    toast_key = f"{key}_toast_sent"

    if (val := st.feedback("thumbs", key=key)) is not None:
        val = int(val)
        if st.session_state.get(toast_key) != val:
            st.toast(
                "感谢您的肯定！我们会继续努力！" if val == 1 else "收到您的反馈，我们会持续改进！"
            )
            st.session_state[toast_key] = val
            logger.info(f"用户反馈: {'满意' if val == 1 else '不满意'}, session: {session_id}")


# ── 返回首页 ───────────────────────────────────────────────
def display_back_to_homepage() -> None:
    """渲染"返回首页"链接，带 scroll_to 锚点参数。"""
    st.page_link("main.py", label="返回首页", query_params={"scroll_to": "main-page-header-anchor"})


# ── 思考气泡（进度展示）───────────────────────────────────
_THOUGHT_BUBBLE_CLASS = "hk-thought-bubble"


def render_thought_bubble(logs: list[str], placeholder: st.delta_generator.DeltaGenerator) -> None:
    """渲染静态思考气泡：展示完整的进度日志列表。

    用于 finally 块中：预测完成后将动画气泡替换为静态完整日志。
    placeholder 是 st.empty() 创建的占位符，多次调用会覆盖之前的内容。
    """
    if not logs:
        return

    thought_content = f"""
    <div class="{_THOUGHT_BUBBLE_CLASS}">
        {"<br>".join(logs)}
    </div>
    """
    placeholder.markdown(thought_content, unsafe_allow_html=True)


def render_thought_bubble_with_wait_pulse(
    logs: list[str], placeholder: st.delta_generator.DeltaGenerator
) -> None:
    """渲染动画思考气泡：展示进度日志 + 最后一行附加"处理中..."脉冲动画。

    用于 progress_cb 回调中：每新增一条日志，重新渲染整个气泡，
    最后一行附加三个点的 CSS 动画（hk-thought-wait-d1/d2/d3 依次淡入）。
    html.escape 防止日志中的 HTML 标签被解释。
    """
    if not logs:
        return
    escaped = [html.escape(x) for x in logs]
    # 三个点的逐次延迟动画，产生"思考中"的动感
    pulse = (
        ' <span class="hk-thought-wait">处理中'
        '<span class="hk-thought-wait-d1">.</span>'
        '<span class="hk-thought-wait-d2">.</span>'
        '<span class="hk-thought-wait-d3">.</span>'
        "</span>"
    )
    escaped[-1] = escaped[-1] + pulse
    inner = "<br>".join(escaped)
    thought_content = f'<div class="{_THOUGHT_BUBBLE_CLASS}">{inner}</div>'
    placeholder.markdown(thought_content, unsafe_allow_html=True)


# ── Lead-In：Ghost Textarea ───────────────────────────────
def render_lead_in_ghost(session_manager) -> str:
    """渲染 ghost textarea（浏览器端 DeepSeek 自动补全组件）。

    架构关键：必须在 @st.fragment 外部渲染。
      原因：ghost_text_area 内部使用 st.components.v1.declare_component 创建自定义
      iframe 组件。如果放在 fragment 内，fragment 重跑会销毁并重建 iframe，
      导致用户正在编辑的文本丢失。

    数据流：
      用户输入 → ghost_text_area 返回文本 → session_state["lead_in_ghost_text"]
      → 用户点击 iframe 内的 AI 按钮 → ghost_text_area 设置 _ghost_analyze_text
      → render_lead_in_actions 消费
    """
    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())
    ghost_key = "lead_in_ghost_text"
    if ghost_key not in st.session_state:
        st.session_state[ghost_key] = ctx.raw_input or ""

    lead_in_consumed = session_manager.get("lead_in_consumed", False)
    placeholder = "✓ 已完成。粘贴新文本开始下一轮预测..." if lead_in_consumed else ""

    api_key = _get_ghost_api_key()
    returned = ghost_text_area(
        api_key=api_key,
        api_base_url="https://api.deepseek.com/beta",
        api_model="deepseek-v4-flash",  # 轻量模型，快速补全
        placeholder=placeholder,
        initial_text=st.session_state[ghost_key],
        height=148,
        key="lead_in_ghost_component",
    )
    if returned:
        st.session_state[ghost_key] = returned
    return st.session_state[ghost_key]


# ── Lead-In：字段 Chip 展示 ───────────────────────────────
# AI 提取完成后，将提取的字段以 chip 形式展示在 lead-in 区域。
# 字段标签 → 中文映射，只展示有值的字段。
_FIELD_LABELS: dict[str, str] = {
    "background_university": "本科院校",
    "background_major": "本科专业",
    "gpa": "GPA",
    "language_type": "语言",
    "language_score": "成绩",
    "target_schools": "目标院校",
    "target_majors": "目标专业",
    "target_country": "目标地区",
    "standardized_test_type": "标化",
    "standardized_test_score": "分数",
}


def _format_field_value(key: str, val) -> str:
    """格式化字段值用于 chip 展示。

    - list → 前 4 项用顿号连接
    - float → 小数点后 1 位（GPA/语言成绩）
    - 其他 → str
    """
    if isinstance(val, list):
        return "、".join(str(v) for v in val[:4] if v)
    if isinstance(val, float):
        return f"{val:.1f}" if key in ("gpa", "language_score") else str(val)
    return str(val)


def _build_field_chips(applied: dict[str, object]) -> str:
    """构建字段 chip 的 HTML 字符串（hk-field-chip-grid 布局）。

    只渲染有值的字段，跳过 None/空字符串/空列表。
    """
    chips: list[str] = []
    for key, label in _FIELD_LABELS.items():
        val = applied.get(key)
        if val is None or val == "" or (isinstance(val, list) and not val):
            continue
        display_val = _format_field_value(key, val)
        chips.append(
            f'<span class="hk-field-chip">'
            f'<span class="hk-chip-label">{label}</span>'
            f'<span class="hk-chip-value">{html.escape(display_val)}</span>'
            f"</span>"
        )
    if not chips:
        return ""
    return f'<div class="hk-field-chip-grid">{"".join(chips)}</div>'


# ── Lead-In：AI 提取按钮 + 结果展示 ──────────────────────
def render_lead_in_actions(session_manager) -> None:
    """处理 AI 提取触发 + 展示提取结果。

    触发链：
      1. ghost_input iframe 内用户点击 AI 按钮
      2. ghost_text_area 检测到 action="analyze" → 设置 st.session_state["_ghost_analyze_text"]
      3. 本函数通过 pop 消费 _ghost_analyze_text
      4. AgentOrchestrator.run("lead_in") 调用 LeadInAgent
      5. apply_lead_in_to_form 将 Agent 输出写入表单 widget state
      6. 设置 _lead_in_processed → _render_lead_in_section 检测 → auto_submit
    """
    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())

    # pop 确保只消费一次（Streamlit 重跑不会重新触发）
    analyze_text = st.session_state.pop("_ghost_analyze_text", None)

    if analyze_text and analyze_text.strip():
        logger.info("AI_AGENT 开始调用 | text=%s", analyze_text[:120])
        AgentOrchestrator.run("lead_in", ctx, user_input=analyze_text)
        applied = apply_lead_in_to_form(ctx, session_manager)
        st.session_state["lead_in_ctx"] = ctx

        # 展示提取结果 chip 网格
        chips_html = _build_field_chips(applied)
        if chips_html:
            st.html(chips_html)

        field_count = len(applied)
        st.session_state["_lead_in_processed"] = True
        logger.info("AI_AGENT 调用完成 | fields=%d", field_count)
