from __future__ import annotations

import re
from typing import Any

PIPELINE: tuple[str, ...] = ("解析学生背景", "结构化提取", "写入表单", "完成")
PIPELINE_ALIASES: dict[str, str] = {
    "逐步填表": "结构化提取",
    "结构化提取中": "结构化提取",
    "匹配院校": "写入表单",
    "写入表单中": "写入表单",
    "触发预测": "完成",
    "生成方案": "完成",
    "识别用户意图": "解析学生背景",
    "判断意图": "解析学生背景",
    "编写编排代码": "结构化提取",
    "编排步骤": "结构化提取",
    "检索院校专业库": "结构化提取",
    "核验申请范围": "结构化提取",
    "核对表单": "结构化提取",
    "写入关键字段": "写入表单",
    "写入语言成绩": "写入表单",
    "写入目标院校": "写入表单",
    "写入经历背景": "写入表单",
    "整理待补充项": "写入表单",
    "生成录取方案": "完成",
    "读取方案摘要": "完成",
    "生成顾问解读": "完成",
    "检查解读状态": "完成",
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
    (r"^预测已触发，页面将生成录取方案$", "已触发预测，方案马上出来"),
    (r"^申请范围核验通过$", "申请范围没问题"),
)


def humanize_detail(line: str) -> str:
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


def entries_to_detail_lines(entries: list[Any] | None, *, limit: int = 10) -> list[str]:
    if not entries:
        return []
    from src.agent.harness import TraceEntry
    from src.agent.lead_in.tool_agent import _tool_running_hint, format_trace_step_summary

    lines: list[str] = []
    for raw in entries:
        if isinstance(raw, TraceEntry):
            entry = raw
        elif isinstance(raw, dict):
            entry = TraceEntry(
                seq=int(raw.get("seq") or 0),
                tool=str(raw.get("tool") or ""),
                args_preview=dict(raw.get("args_preview") or {}),
                result_preview=str(raw.get("result_preview") or ""),
                ts=float(raw.get("ts") or 0.0),
                ok=bool(raw.get("ok", True)),
            )
        else:
            continue
        if entry.tool == "router_stage" and entry.result_preview:
            line = humanize_detail(entry.result_preview)
        elif entry.result_preview:
            line = humanize_detail(format_trace_step_summary(entry))
        else:
            line = humanize_detail(_tool_running_hint(entry.tool))
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return lines[-limit:]


def normalize_steps(steps: list[str]) -> list[str]:
    out: list[str] = []
    for s in steps or []:
        key = PIPELINE_ALIASES.get(s, s)
        if key not in out:
            out.append(key)
    return out


def stage_index(steps: list[str]) -> int:
    norm = normalize_steps(steps)
    idx = -1
    for s in norm:
        if s in PIPELINE:
            idx = max(idx, PIPELINE.index(s))
    if idx < 0 and norm:
        return min(len(norm) - 1, len(PIPELINE) - 1)
    return idx


def status_hint(elapsed: float, *, active: bool, retry: int) -> str:
    if not active:
        return ""
    if retry > 0:
        return "网络偏慢，已自动续跑，无需重复提交"
    if elapsed < 18:
        return ""
    if elapsed < 40:
        return "仍在处理，可先核对下方表单"
    return "耗时偏长，完成后会自动填表"
