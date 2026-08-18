from __future__ import annotations

import html
import re

from src.pages.prediction.ui.lead_in_progress_copy import (
    PIPELINE,
    entries_to_detail_lines,
    humanize_detail,
    stage_index,
    status_hint,
)

__all__ = ["entries_to_detail_lines", "humanize_detail", "render_progress_html"]


def _inline_bold_markdown(line: str) -> str:
    out: list[str] = []
    pos = 0
    for m in re.finditer(r"\*\*(.+?)\*\*", line):
        out.append(html.escape(line[pos : m.start()]))
        out.append(f"<strong>{html.escape(m.group(1))}</strong>")
        pos = m.end()
    out.append(html.escape(line[pos:]))
    return "".join(out)


def _format_agent_feedback_html(text: str) -> str:
    paragraphs: list[str] = []
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        paragraphs.append(f'<p class="hk-lead-in-feedback-p">{_inline_bold_markdown(line)}</p>')
    return "".join(paragraphs) if paragraphs else f"<p>{html.escape(text)}</p>"


def _rail_html(steps: list[str], *, active: bool) -> str:
    if not active:
        return ""
    cur = stage_index(steps)
    if cur < 0:
        cur = 0
    n = len(PIPELINE)
    pct = 0.0 if n <= 1 else (cur / (n - 1)) * 100.0
    short_map = {
        "解析学生背景": "读背景",
        "结构化提取": "提取",
        "写入表单": "填表",
        "完成": "完成",
    }
    nodes: list[str] = []
    for i, name in enumerate(PIPELINE):
        if i < cur:
            state = "done"
        elif i == cur:
            state = "cur"
        else:
            state = ""
        mark = "" if state == "done" else str(i + 1)
        check = '<span class="hk-li-check"></span>' if state == "done" else mark
        short = short_map.get(name, name)
        nodes.append(
            f'<div class="hk-li-node {state}">'
            f'<div class="hk-li-orb">{check}</div>'
            f'<span class="lbl">{html.escape(short)}</span></div>'
        )
    return (
        f'<div class="hk-li-rail">'
        f'<div class="hk-li-rail-track">'
        f'<div class="hk-li-rail-progress" style="width:{pct:.1f}%"></div>'
        f"</div>"
        f'<div class="hk-li-nodes">{"".join(nodes)}</div>'
        f"</div>"
    )


def _details_html(details: list[str], *, active: bool, max_show: int = 4) -> str:
    if not details:
        return ""
    shown = details[-max_show:] if active else details[-4:]
    hidden = max(0, len(details) - len(shown))
    rows: list[str] = []
    for i, line in enumerate(shown):
        is_cur = active and i == len(shown) - 1
        cls = "hk-li-row cur" if is_cur else "hk-li-row"
        rows.append(
            f'<div class="{cls}"><span class="mk"></span>'
            f"<span>{html.escape(humanize_detail(line))}</span></div>"
        )
    more = f'<div class="hk-li-hint">更早 {hidden} 步已折叠</div>' if hidden and active else ""
    return f'<div class="hk-li-details">{"".join(rows)}{more}</div>'


def render_progress_html(
    steps: list[str],
    partial_text: str,
    *,
    elapsed: float = 0,
    retry: int = 0,
    variant: str = "default",
    details: list[str] | None = None,
    active: bool = True,
    path_hint: str = "",
) -> str:
    blocked = variant == "intent_blocked"
    detail_lines = [str(x).strip() for x in (details or []) if str(x).strip()]

    if blocked:
        partial = html.escape(str(partial_text or "").strip())
        return (
            '<div class="hk-li-card">'
            '<div class="hk-li-title-l" style="color:var(--hk-orange,#f97316);'
            'text-transform:none;letter-spacing:0.02em">暂时没法直接处理这个问题</div>'
            f'<div class="hk-li-live" style="color:var(--hk-slate-400)">{partial}</div></div>'
        )

    hint = status_hint(elapsed, active=active, retry=retry)
    foot_bits: list[str] = []
    if active and elapsed >= 1:
        foot_bits.append(f"{elapsed:.0f}s")
    if retry > 0:
        foot_bits.append(f"续跑 {retry}/3")
    foot = f'<div class="hk-li-foot">{" · ".join(foot_bits)}</div>' if foot_bits else ""

    strategy = ""
    ph = (path_hint or "").strip()
    if active and ph:
        strategy = f'<div class="hk-li-strategy">{html.escape(ph)}</div>'

    live = ""
    if active and partial_text and not detail_lines:
        live = (
            f'<div class="hk-li-live">'
            f"{_format_agent_feedback_html(humanize_detail(partial_text.strip()))}"
            "</div>"
        )

    empty = ""
    if active and not steps and not detail_lines and not partial_text:
        empty = '<div class="hk-li-hint">正在接通…</div>'

    card_cls = "hk-li-card" if active else "hk-li-card done"
    hint_html = f'<div class="hk-li-hint">{html.escape(hint)}</div>' if hint else ""

    if active:
        title = "页面闪断，正在自动续跑" if retry > 0 else "解读中"
        header = (
            f'<div class="hk-li-title">'
            f'<div class="hk-li-title-l"><span class="hk-li-dot"></span>'
            f"{html.escape(title)}</div>{foot}</div>"
        )
        return (
            f'<div class="{card_cls}">'
            f"{header}{strategy}"
            f"{_rail_html(steps, active=True)}"
            f"{_details_html(detail_lines, active=True)}"
            f"{live}{empty}{hint_html}</div>"
        )

    if not detail_lines:
        return ""
    return f'<div class="{card_cls}">{_details_html(detail_lines, active=False)}</div>'


_render_progress_html = render_progress_html
