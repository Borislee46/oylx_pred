from __future__ import annotations

from typing import Any

# 措辞与工具标签的唯一实现在 agent 层（实时路径也要用），这里只做转发。
from src.agent.step_text import TOOL_LABEL, humanize_detail  # noqa: F401

PIPELINE: tuple[str, ...] = ("解析学生背景", "结构化提取", "写入表单", "完成")

# 4 段流水线的短标签（rail 上显示）。**只在这里定义一份**——
# 以前 `lead_in_dispatch` 和 `lead_in_wait` 各抄了一份，改一处就会不一致。
STAGE_SHORT: dict[str, str] = {
    "解析学生背景": "读背景",
    "结构化提取": "提取",
    "写入表单": "填表",
    "完成": "完成",
}
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



def entries_to_detail_lines(entries: list[Any] | None, *, limit: int = 12) -> list[str]:
    """把一轮的 trace 变成**给人看的**展示行。

    措辞与实时路径共用 `step_text.step_line`（agent 侧也用它），
    所以"进行中"和"完成后"看到的是同一批说法。
    本函数只负责：条目归一化（dict → TraceEntry）、去连续重复、截尾。
    """
    if not entries:
        return []
    from src.agent.harness import TraceEntry
    from src.agent.step_text import step_line

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
                duration_ms=float(raw.get("duration_ms") or 0.0),
            )
        else:
            continue

        line = step_line(entry)
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
