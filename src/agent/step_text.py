"""步骤行文案：把一次工具调用变成**一行给人看的话**。

## 为什么放在 agent 层

实时进度由 agent 侧产出（`tool_agent._trace_detail_lines` 每 ~0.3s 轮询 trace），
完成后回看由 UI 侧产出（`lead_in_progress_copy.entries_to_detail_lines`）。
两边必须共用这一份实现，否则"进行中"和"完成后"的措辞会不一致
——历史上就是两套，正是这次去冗余要消灭的东西。

本模块**不依赖 Streamlit**，只有字符串处理（对 `format_trace_step_summary` 是延迟导入）。
"""

from __future__ import annotations

import os
import re

# 极客层开关：开着时步骤行附带原始工具名与耗时，并提供折叠的"原始调用明细"
# （参数 / 返回 JSON）。默认开 —— 折叠起来用户不会看，但对排障非常有用。
# 想给终端用户一个纯人话界面：设 `LEAD_IN_GEEK_UI=0`。
_GEEK_ENV = "LEAD_IN_GEEK_UI"
_GEEK_FALSY = ("0", "false", "no", "off")


def geek_enabled() -> bool:
    return os.environ.get(_GEEK_ENV, "1").strip().lower() not in _GEEK_FALSY


# 工具 → 用户看得懂的标签。步骤行里显示它，让用户知道"用了啥"，
# 而不是把 `lookup_admission_stats` 这种函数名摆到界面上。
# 极客模式下原始函数名会作为附注同时出现（见 `step_line`）。
TOOL_LABEL: dict[str, str] = {
    "read_form": "读表单",
    "write_form": "写表单",
    "get_form_options": "查候选名",
    "check_scope": "核验范围",
    "submit_prediction": "触发预测",
    "expand_form": "展开表单",
    "start_new_profile": "新建档案",
    "standardize_background_university": "对齐校名",
    "standardize_background_major": "对齐专业",
    "standardize_target_major": "对齐目标专业",
    "standardize_fields": "对齐字段",
    "lookup_program": "查项目要求",
    "lookup_school": "查院校层次",
    "list_supported_targets": "查支持范围",
    "lookup_admission_stats": "查历史录取率",
    "estimate_admission_chance": "估录取可能",
    "lookup_similar_admitted_cases": "找相似先例",
}

_INTENT_LABEL = {
    "profile": "背景提取",
    "question": "答疑",
    "vague": "信息偏少",
    "off_topic": "偏题",
}
_CONF_LABEL = {"high": "较高", "medium": "中等", "low": "偏低"}

_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"^已收到输入，准备结构化提取$", "已读完输入，开始拆字段"),
    (r"^已收到输入，正在阅读并提取…?$", "已收到，先快速扫一遍背景…"),
    (r"^模型正在提取院校 / 专业 / 成绩等字段$", "正在拆解院校、专业与成绩"),
    (r"^模型正在提取字段…?$", "正在拆字段…"),
    (r"^正在阅读输入…?$", "正在快速扫一遍背景"),
    (r"^正在匹配院校专业并写入表单…?$", "正在对齐院校库并填表…"),
    (r"^识别到：(.+)$", r"已抓住：\1"),
    (r"^核心字段齐全，即将生成录取方案$", "关键信息齐了，准备出方案"),
    (r"^核心字段未齐，等待补充后再预测$", "还缺几项关键信息，先展开表单核对"),
    (r"^置信偏低或非背景提取，展开表单供确认$", "信息不够稳，先请你确认表单"),
    (r"^未写入新字段（可能与表单已有值一致）$", "表单已是最新，无需重复写入"),
    (r"^已写入表单 (\d+) 项$", r"已填入表单 \1 项"),
    (r"^正在连接模型，请稍候…?$", "正在接通顾问引擎…"),
    (r"^模型继续分析中…?$", "还在交叉核对细节…"),
    (r"^已写入部分字段，正在收尾…?$", "字段已写入大半，正在收尾"),
    (r"^意图识别通过，继续处理$", "意图没问题，继续往下"),
    (r"^正在识别用户意图…?$", "先判断这是背景还是答疑…"),
    (r"^正在检索院校专业库…?$", "正在对齐院校 / 专业库…"),
    (r"^正在核验申请范围…?$", "正在核对是否在支持范围内…"),
    (r"^正在写入关键字段…?$", "正在把关键信息填进表单…"),
    (r"^正在生成录取方案…?$", "关键信息齐了，正在出方案…"),
    (r"^预测已触发，页面将生成录取方案$", "方案马上出来"),
    (r"^申请范围核验通过$", "申请范围没问题"),
)


def humanize_detail(line: str) -> str:
    """把机器措辞换成口语。"""
    t = (line or "").strip()
    if not t:
        return ""
    m = re.match(r"^意图=(\w+)\s*[·•]\s*置信=(\w+)$", t)
    if m:
        intent = _INTENT_LABEL.get(m.group(1), m.group(1))
        conf = _CONF_LABEL.get(m.group(2), m.group(2))
        return f"判断：{intent} · 把握{conf}"
    for pat, repl in _REPLACEMENTS:
        newt = re.sub(pat, repl, t)
        if newt != t:
            return newt
    return t


def step_line(entry, *, geek: bool | None = None) -> str:
    """一个 trace 条目 → 一行展示文案：`**工具标签** · 一句话结果`。

    **默认不 dump 原始 JSON**：11 个后加的工具以前没有摘要分支，
    fallthrough 直接把 JSON 打到了界面上，这是"界面像测试日志"的根因。

    极客模式（`LEAD_IN_GEEK_UI=1`，默认）额外在行尾附上原始函数名与耗时：
        **查项目要求** · Master of Finance：雅思 6.0、托福 80 `lookup_program · 355ms`
    参数与返回的原始 JSON 由 `StepLog.raw_details` 放在**折叠区**里，不占版面。
    """
    from src.agent.lead_in.tool_agent import _tool_running_hint, format_trace_step_summary

    tool = str(getattr(entry, "tool", "") or "")
    preview = str(getattr(entry, "result_preview", "") or "")

    if tool == "router_stage" and preview:
        # 历史遗留：router 阶段的合成条目自带完整文案，直接用
        base = humanize_detail(preview)
    elif preview:
        base = humanize_detail(format_trace_step_summary(entry))
    else:
        base = humanize_detail(_tool_running_hint(tool))

    label = TOOL_LABEL.get(tool)
    line = f"**{label}** · {base}" if label and base else (label or base)

    if not line:
        return ""
    if (geek if geek is not None else geek_enabled()) and tool and tool != "router_stage":
        ms = float(getattr(entry, "duration_ms", 0.0) or 0.0)
        if ms <= 0:
            timing = ""
        elif ms < 1:
            timing = " · <1ms"
        else:
            timing = f" · {ms:.0f}ms"
        line = f"{line} `{tool}{timing}`"
    return line
