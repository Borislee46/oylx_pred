import base64
import html
from pathlib import Path

import streamlit as st

from src.agent.context import StudentContext
from src.agent.form_bridge import apply_lead_in_to_form
from src.agent.orchestrator import AgentOrchestrator
from src.utils.logger import setup_logger
from src.utils.ui.hk_shield_v2 import mount_hk_shield_v2

logger = setup_logger("page3", "prediction")


@st.cache_data(show_spinner=False)
def get_product_logo_image_as_base64(path: str) -> str:
    full_path = Path.cwd() / path
    return base64.b64encode(full_path.read_bytes()).decode()


def render_header(logo_base64: str) -> None:
    mask_base64 = get_product_logo_image_as_base64("assets/shield_metal.png")
    metal_base64 = get_product_logo_image_as_base64("assets/shield_mask.png")

    if not logo_base64:
        st.title("Signals 选校预测系统")
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
                <p class="hk-header-subtitle">留学择校系统</p>
            </div>
        </div>
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


THOUGHT_BUBBLE_STYLE = """
    border-left: 1.5px solid #efefef;
    padding-left: 0.8rem;
    margin-top: -12px;
    margin-bottom: 8px;
    color: #a0a0a0;
    font-style: italic;
    font-size: 0.82em;
    line-height: 1.3;
    font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
"""


def render_thought_bubble(logs: list[str], placeholder: st.delta_generator.DeltaGenerator) -> None:
    if not logs:
        return

    thought_content = f"""
    <div style="{THOUGHT_BUBBLE_STYLE}">
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
    thought_content = f'<div style="{THOUGHT_BUBBLE_STYLE}">{inner}</div>'
    placeholder.markdown(thought_content, unsafe_allow_html=True)


def render_lead_in_panel(session_manager) -> None:
    if "lead_in_ctx" not in st.session_state:
        st.session_state["lead_in_ctx"] = StudentContext()

    ctx: StudentContext = st.session_state["lead_in_ctx"]
    has_lead_in = bool(ctx.quick_assessment)
    has_predicted = session_manager.get("has_predicted", False)
    form_active = (
        has_lead_in
        or bool(session_manager.get("lead_in_form_summary", ""))
        or session_manager.get("form_expanded", False)
    )

    def _step_class(step: int) -> str:
        if has_predicted:
            return " hk-step-done"
        if step == 1:
            return " hk-step-done" if has_lead_in else " hk-step-active"
        if step == 2:
            return " hk-step-active" if form_active else ""
        return ""

    st.html(
        '<div class="hk-step-row">'
        f'<div class="hk-step-item{_step_class(1)}">'
        '<div class="hk-step-num">1</div>'
        '<div class="hk-step-label">背景整理</div>'
        "</div>"
        '<div class="hk-step-connector"></div>'
        f'<div class="hk-step-item{_step_class(2)}">'
        '<div class="hk-step-num">2</div>'
        '<div class="hk-step-label">表单核验</div>'
        "</div>"
        '<div class="hk-step-connector"></div>'
        f'<div class="hk-step-item{_step_class(3)}">'
        '<div class="hk-step-num">3</div>'
        '<div class="hk-step-label">预测结果</div>'
        "</div>"
        "</div>"
    )

    if not has_lead_in:
        st.html(
            '<p style="color:var(--hk-slate-400);font-size:0.8rem;margin:0 0 0.5rem 0">'
            "输入学生背景，系统会整理并填入表单</p>"
        )

    with st.container(border=True):
        raw = st.text_area(
            "输入学生背景",
            value=ctx.raw_input,
            placeholder="例如：北航 CS GPA 3.2 雅思7.0 2段科研经历 想去港三 CS",
            height=68,
            key="lead_in_raw",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 4])
        with c1:
            analyze = st.button("整理信息", key="lead_in_btn", width="stretch")

        if analyze and raw.strip():
            progress = st.empty()
            _render_lead_in_progress(progress, "正在整理背景")
            AgentOrchestrator.run("lead_in", ctx, user_input=raw)
            _render_lead_in_progress(progress, "已整理")
            progress.empty()
            apply_lead_in_to_form(ctx, session_manager)
            st.session_state["lead_in_ctx"] = ctx
            st.rerun()

    if has_lead_in:
        _render_lead_in_summary(ctx, c2, session_manager)


def _render_lead_in_summary(ctx, action_col, session_manager) -> None:
    info = {k: v for k, v in (ctx.extracted_background or {}).items() if v}
    summary = ctx.quick_assessment[:88] + ("…" if len(ctx.quick_assessment) > 88 else "")

    st.html(
        '<div class="hk-insight-card hk-insight-accent" style="margin-top:0.5rem">'
        f'<p style="margin:0;font-size:0.82rem;line-height:1.5;color:var(--hk-slate-600)">{summary}</p>'
        "</div>"
    )

    with action_col:
        if st.button("重新输入", key="lead_in_clear"):
            st.session_state["lead_in_ctx"] = StudentContext()
            session_manager.set(lead_in_form_summary="", lead_in_form_filled=False)
            st.rerun()

    with st.expander(f"查看已整理信息 · {len(info)} 项"):
        FIELD_LABELS = {
            "university": "院校",
            "major": "专业",
            "gpa": "GPA",
            "language_type": "语言类型",
            "language_score": "语言成绩",
            "country": "目标地区",
            "grade": "年级",
            "target_schools": "目标院校",
            "target_majors": "目标专业",
            "research": "科研经历",
            "internship": "实习经历",
            "paper": "论文",
            "award": "获奖",
        }
        parts = ['<div class="hk-field-chip-grid">']
        for k, v in info.items():
            label = FIELD_LABELS.get(k, k)
            val = _fmt_field_value(k, v)
            parts.append(
                '<span class="hk-field-chip">'
                f'<span class="hk-chip-label">{label}</span>'
                f'<span class="hk-chip-value">{val}</span>'
                "</span>"
            )
        parts.append("</div>")
        st.html("\n".join(parts))

        if ctx.suggested_questions:
            pills = "".join(
                f'<span class="hk-question-pill">{q}</span>' for q in ctx.suggested_questions
            )
            st.html(f'<div style="margin-top:0.5rem">{pills}</div>')


def _render_lead_in_progress(placeholder, text: str) -> None:
    placeholder.markdown(
        f'<div style="border-left:1.5px solid var(--hk-cyan);padding:0.4rem 0.7rem;'
        f'color:var(--hk-slate-500);font-size:0.82rem;font-style:italic">'
        f"{text}"
        '<span class="hk-thought-wait">'
        '<span class="hk-thought-wait-d1">.</span>'
        '<span class="hk-thought-wait-d2">.</span>'
        '<span class="hk-thought-wait-d3">.</span>'
        "</span>"
        "</div>",
        unsafe_allow_html=True,
    )


def _fmt_field_value(key: str, val) -> str:
    if isinstance(val, list):
        return ", ".join(str(v) for v in val[:4])
    if isinstance(val, float):
        if key in ("gpa",):
            return f"{val:.2f}"
        if key == "language_score":
            return f"{val:.1f}" if val == int(val) else f"{round(val)}"
    return str(val)
