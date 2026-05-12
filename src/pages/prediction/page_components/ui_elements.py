import base64
import html
from pathlib import Path

import streamlit as st

from src.agent.context import StudentContext
from src.agent.form_bridge import apply_lead_in_to_form
from src.agent.orchestrator import AgentOrchestrator
from src.pages.prediction.ghost_input import ghost_text_area
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger
from src.utils.ui.hk_shield_v2 import mount_hk_shield_v2

logger = setup_logger("page3", "prediction")

_api_key_cache: str | None = None


def _get_ghost_api_key() -> str:
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
    """Walk upward from this file to locate the project root via pyproject.toml."""
    start = Path(__file__).resolve()
    for p in (start, *start.parents):
        if (p / "pyproject.toml").exists():
            return p
    return start.parents[4]


@st.cache_data(show_spinner=False)
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = _get_project_root() / path
    data = full_path.read_bytes()
    if not data:
        return ""
    return base64.b64encode(data).decode()


def render_header(logo_base64: str) -> None:
    mask_base64 = get_product_logo_image_as_base64("assets/shield_metal.png")
    metal_base64 = get_product_logo_image_as_base64("assets/shield_mask.png")

    if not logo_base64:
        st.title("Signals")
        return

    mount_hk_shield_v2()

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


def display_feedback_section(session_id: str) -> None:
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


def display_back_to_homepage() -> None:
    st.page_link("main.py", label="返回首页", query_params={"scroll_to": "main-page-header-anchor"})


_THOUGHT_BUBBLE_CLASS = "hk-thought-bubble"


def render_thought_bubble(logs: list[str], placeholder: st.delta_generator.DeltaGenerator) -> None:
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
    if not logs:
        return
    escaped = [html.escape(x) for x in logs]
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


def render_lead_in_ghost(session_manager) -> str:
    """Render the ghost textarea at page level.

    Must stay outside any ``@st.fragment`` so its iframe survives fragment
    reruns. Returns the current raw text (via session_state).
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
        api_model="deepseek-v4-flash",
        placeholder=placeholder,
        initial_text=st.session_state[ghost_key],
        height=148,
        key="lead_in_ghost_component",
    )
    if returned:
        st.session_state[ghost_key] = returned
    return st.session_state[ghost_key]


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
    if isinstance(val, list):
        return "、".join(str(v) for v in val[:4] if v)
    if isinstance(val, float):
        return f"{val:.1f}" if key in ("gpa", "language_score") else str(val)
    return str(val)


def _build_field_chips(applied: dict[str, object]) -> str:
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


def _scroll_to(target_selector: str) -> None:
    """Inject JS to smoothly scroll to a DOM element in the parent frame."""
    st.components.v1.html(
        f"""<script>
        setTimeout(function() {{
            var el = window.parent.document.querySelector('{target_selector}');
            if (el) {{ el.scrollIntoView({{behavior:'smooth', block:'start'}}); }}
        }}, 150);
        </script>""",
        height=0,
    )


def render_lead_in_actions(session_manager) -> None:
    """Render post-analysis summary.  AI trigger comes from the iframe button."""
    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())

    analyze_text = st.session_state.pop("_ghost_analyze_text", None)

    if analyze_text and analyze_text.strip():
        logger.info("AI_AGENT 开始调用 | text=%s", analyze_text[:120])
        AgentOrchestrator.run("lead_in", ctx, user_input=analyze_text)
        applied = apply_lead_in_to_form(ctx, session_manager)
        st.session_state["lead_in_ctx"] = ctx

        chips_html = _build_field_chips(applied)
        if chips_html:
            st.html(chips_html)

        field_count = len(applied)
        st.session_state["_lead_in_processed"] = True
        logger.info("AI_AGENT 调用完成 | fields=%d", field_count)
