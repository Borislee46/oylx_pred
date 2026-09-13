"""Lead-in 智能体（ReAct 循环）。

## 设计要点

- **单一执行体**：不再有 router-first 分流。所有输入交给同一个 pydantic-ai `Agent`，
  由它自己决定「直接回答 / 追问 / 查数据 / 写表单 / 触发预测」。信息齐全时它不调工具，
  一轮 LLM 就能回复，所以并没有牺牲快路径。
- **运行时**：所有异步调用提交到进程级常驻循环（`runtime.event_loop`）。
  绝不能在这里用 `asyncio.run`——模型层的 `httpx.AsyncClient` 是进程级缓存，
  连接池绑定宿主循环，换循环复用会抛 `RuntimeError: Event loop is closed`。
- **进度来源唯一**：工具步骤一律取自 `ToolTurnTrace`（`tools.tool_trace.traced` 写入），
  文本回复取自模型流式 delta。不要再另建一套"从事件名猜步骤"的并行实现。
- **不要取消事件迭代器**：`asyncio.wait_for(aiter.__anext__())` 超时会关闭异步生成器，
  后续迭代抛 `StopAsyncIteration` → 流被静默截断（已复现）。心跳由独立轮询任务负责。
- **隔离**：所有写入先落在 work 副本上，成功才 `_commit_work_deps` 提交；
  超时/异常一律丢弃，绝不留下半截状态。
"""

import asyncio
import contextlib
import copy
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic_ai import Agent, AgentRunResultEvent, UsageLimits
from pydantic_ai.exceptions import UsageLimitExceeded
from pydantic_ai.messages import (
    PartDeltaEvent,
    TextPartDelta,
)

from src.agent.context import StudentContext
from src.agent.field_labels import FIELD_LABEL
from src.agent.harness import HarnessDeps, ToolTurnTrace, TraceEntry
from src.agent.lead_in.agent_prompts import build_agent_system_prompt
from src.agent.lead_in.session_lock import get_session_lock
from src.agent.runtime.event_loop import submit
from src.agent.runtime.model_factory import (
    LEAD_IN_TOOL_STREAM_TIMEOUT,
    build_model_with_fallback,
)
from src.agent.tools.form_tools import FORM_TOOLS
from src.agent.tools.probe_tools import PROBE_TOOLS
from src.agent.tools.recall_tools import RECALL_TOOLS

_log = logging.getLogger("lead_in_tool_agent")

# 轮询间隔：只用于把 trace 变化翻译成 UI 进度，不影响模型/工具本身。
_POLL_INTERVAL_SECONDS = 0.3
# 无变化时的心跳间隔。
_HEARTBEAT_SECONDS = 5.0
# 配置读不到时的兜底轮数上限。
_DEFAULT_MAX_TOOL_ROUNDS = 12


def _build_tool_user_prompt(text: str, deps: HarnessDeps) -> str:
    ctx = deps.ctx
    if getattr(ctx, "session_continuity", "continue") == "fresh":
        return text
    try:
        from src.agent.lead_in.session_continuity import (
            build_continuity_user_prompt,
            has_prior_context,
        )

        sm = getattr(getattr(deps, "gateway", None), "_sm", None)
        if has_prior_context(sm, ctx):
            return build_continuity_user_prompt(text, ctx, sm)
    except Exception:
        _log.debug("_build_tool_user_prompt fallback", exc_info=True)
    turn_note = f"[第{deps.turn}轮对话] " if deps.turn > 1 else ""
    return f"{turn_note}{text}"


def _resolve_session_id(deps: HarnessDeps) -> str | None:
    try:
        gw = getattr(deps, "gateway", None)
        sm = getattr(gw, "_sm", None)
        if sm is not None:
            return getattr(sm, "session_id", None)
    except Exception:
        pass
    return None


def _effective_turn_timeout() -> float:
    """单轮预算：优先取 HarnessConfig.turn_budget_ms，读不到才用常量。"""
    try:
        from src.agent.harness import load_harness_config

        ms = int(getattr(load_harness_config(), "turn_budget_ms", 0) or 0)
        if ms > 0:
            return ms / 1000.0
    except Exception:
        _log.debug("TURN_BUDGET | config unavailable, fallback to constant", exc_info=True)
    return float(LEAD_IN_TOOL_STREAM_TIMEOUT)


def _max_tool_rounds() -> int:
    try:
        from src.agent.harness import load_harness_config

        rounds = int(getattr(load_harness_config(), "max_tool_rounds", 0) or 0)
        if rounds > 0:
            return rounds
    except Exception:
        _log.debug("MAX_ROUNDS | config unavailable, fallback to default", exc_info=True)
    return _DEFAULT_MAX_TOOL_ROUNDS


def _usage_limits() -> UsageLimits:
    rounds = _max_tool_rounds()
    return UsageLimits(request_limit=rounds, tool_calls_limit=rounds)


_TOOL_STEP_PHRASE = {
    "read_form": "解析学生背景",
    "get_form_options": "检索院校专业库",
    "check_scope": "核验申请范围",
    "write_form": "写入关键字段",
    "submit_prediction": "生成录取方案",
    "expand_form": "整理待补充项",
    "standardize_background_university": "标准化背景字段",
    "standardize_background_major": "标准化背景字段",
    "standardize_target_major": "标准化目标字段",
    "standardize_fields": "标准化背景字段",
}

_GATEWAY_FIELD_STEPS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("university", "major", "gpa", "gpa_scale"), "写入关键字段"),
    (("language_type", "language_score"), "写入语言成绩"),
    (("target_schools", "target_majors", "country"), "写入目标院校"),
    (("research", "internship", "paper", "award"), "写入经历背景"),
)


def _try_parse_json(text: str) -> Any | None:
    text = (text or "").strip()
    if not text or text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return None


def _format_field_snippet(key: str, val: Any) -> str:
    if val in (None, "", [], "None"):
        return FIELD_LABEL.get(key, key)
    if isinstance(val, list):
        joined = "、".join(str(v) for v in val[:3] if v)
        return joined or FIELD_LABEL.get(key, key)
    if key in ("gpa", "language_score") and isinstance(val, (int, float)):
        return f"{FIELD_LABEL.get(key, key)} {val}"
    s = str(val).strip()
    if len(s) > 24:
        s = s[:24] + "…"
    label = FIELD_LABEL.get(key, key)
    if key in ("research", "internship", "paper", "award"):
        return f"{label}：{s}"
    return s


def _tool_running_hint(tool: str) -> str:
    phrase = _TOOL_STEP_PHRASE.get(tool or "")
    return f"正在{phrase}…" if phrase else f"正在执行 {tool}…"


def feedback_from_trace(trace) -> str:
    if trace is None or not trace.entries:
        return ""
    parts: list[str] = []
    for entry in trace.entries:
        if not entry.ok:
            continue
        summary = format_trace_step_summary(entry)
        if summary and summary not in parts:
            parts.append(summary)
    if not parts:
        return ""
    if any("预测已触发" in p for p in parts):
        return "已识别您的背景并触发录取方案生成，请查看下方结果。"
    return parts[-1]


def format_trace_step_summary(entry: TraceEntry) -> str:
    tool = entry.tool or ""
    preview = (entry.result_preview or "").strip()
    if not entry.ok:
        msg = preview.removeprefix("Error:").strip()
        return f"步骤失败：{msg[:120]}" if msg else "步骤执行失败"

    data = _try_parse_json(preview)

    if tool == "read_form" and isinstance(data, dict):
        missing = [str(x) for x in (data.get("missing") or []) if x]
        if missing:
            return "仍缺：" + "、".join(missing)
        present = data.get("present") or {}
        if present:
            keys = [FIELD_LABEL.get(k, k) for k in list(present.keys())[:6]]
            return "已有字段：" + "、".join(keys)
        return "表单为空，待填写"

    if tool == "write_form" and isinstance(data, dict):
        applied = [str(k) for k in (data.get("applied") or []) if k]
        missing = [str(x) for x in (data.get("missing") or []) if x]
        args = entry.args_preview or {}
        parts = [_format_field_snippet(k, args.get(k)) for k in applied]
        line = "已写入：" + " · ".join(parts[:8]) if parts else "已更新表单"
        if missing:
            line += "；仍缺：" + "、".join(missing)
        return line

    if tool == "get_form_options":
        field = str((entry.args_preview or {}).get("field") or "字段")
        field_label = {
            "university": "本科院校",
            "major": "本科专业",
            "target_major": "目标专业",
        }.get(field, field)
        if isinstance(data, list):
            n = len(data)
            sample = "、".join(str(x) for x in data[:3])
            if n > 3:
                sample += f" 等{n}项"
            return f"{field_label}候选：{sample}" if sample else f"{field_label}暂无候选"
        if preview:
            return preview[:160]

    if tool == "check_scope" and isinstance(data, dict):
        if data.get("in_scope"):
            return "申请范围核验通过"
        issues = data.get("issues") or data.get("unsupported_schools") or []
        if issues:
            return "范围外：" + "、".join(str(x) for x in issues[:4])
        return "部分目标不在支持范围内"

    if tool == "submit_prediction":
        if isinstance(data, dict):
            if data.get("ok"):
                return "预测已触发，页面将生成录取方案"
            missing = [str(x) for x in (data.get("missing") or []) if x]
            if missing:
                return "预测未触发，仍缺：" + "、".join(missing)
            reason = str(data.get("blocked_reason") or "").strip()
            if reason:
                return f"预测未触发：{reason[:80]}"
        if preview.startswith("已触发"):
            return "预测已触发，页面将生成录取方案"
        if preview.startswith("未触发"):
            return preview[:160]

    if tool == "expand_form":
        # 标签已经写了"展开表单"，这里不要重复动词
        return "供用户补充或核对"

    if tool == "start_new_profile":
        return "已切换为新学生（上一位的档案已清空）"

    # ── 字段对齐 ──────────────────────────────────────────────────
    if tool in (
        "standardize_background_university",
        "standardize_background_major",
        "standardize_target_major",
    ) and isinstance(data, dict):
        raw, std = str(data.get("raw") or ""), str(data.get("standardized") or "")
        if not std or std == raw:
            return f"对齐「{raw}」：保持原值"
        return f"对齐：{raw} → {std}"

    if tool == "standardize_fields" and isinstance(data, dict):
        changed = [
            f"{v.get('raw')} → {v.get('standardized')}"
            for v in data.values()
            if isinstance(v, dict) and v.get("standardized") and v.get("standardized") != v.get("raw")
        ]
        return "对齐：" + "、".join(changed) if changed else "字段名无需对齐"

    # ── 探查类（这些以前直接 fallthrough 到 JSON dump，是最"raw"的地方）──
    if tool == "lookup_program" and isinstance(data, dict):
        if not data.get("found"):
            reason = data.get("reason")
            if reason == "unknown_school":
                return f"项目库里没有「{data.get('input_university') or ''}」"
            if reason == "major_required":
                return f"{data.get('resolved_university') or ''}：有 {data.get('available_major_count')} 个项目，需指定专业"
            return "没查到该项目，已给出相近方向"
        lang = data.get("language_requirements") or {}
        bits = [f"{k} {v}" for k, v in list(lang.items())[:2] if v]
        name = str(data.get("major") or "")
        return f"{name}：{'、'.join(bits)}" if bits else f"已查到 {name} 的项目要求"

    if tool == "lookup_school" and isinstance(data, dict):
        if data.get("is_level_alias"):
            return "这是院校层次别名，按原话写入"
        if not data.get("found"):
            return f"层次库里没有「{data.get('input') or ''}」，按原话写入"
        return f"{data.get('resolved_school') or ''}：{data.get('school_level') or '未知'}"

    if tool == "list_supported_targets":
        if isinstance(data, dict) and data.get("found"):
            if data.get("universities"):
                return f"{data.get('country')}：{len(data['universities'])} 所院校"
            countries = data.get("supported_countries") or []
            return f"支持 {len(countries)} 个地区：{'、'.join(str(c) for c in countries[:4])}"
        return "该地区不在支持范围内"

    if tool == "lookup_admission_stats" and isinstance(data, dict):
        if not data.get("found"):
            return "录取统计里没有这个组合"
        rate = data.get("admit_rate")
        n = data.get("sample_size")
        if rate is None:
            return f"样本仅 {n} 条，不给录取率"
        return f"历史录取率 {float(rate) * 100:.0f}%（样本 {n} 条）"

    if tool == "estimate_admission_chance" and isinstance(data, dict):
        if not data.get("found"):
            return "缺少本科背景，无法做分层估计"
        if data.get("insufficient_stratum"):
            broader = data.get("broader_reference") or {}
            base = broader.get("admit_rate")
            if base is not None:
                return f"分层样本不足，退回项目整体 {float(base) * 100:.0f}%"
            return "分层样本不足，不给个性化数字"
        prob = data.get("probability")
        if prob is None:
            return "分层样本不足，不给个性化数字"
        n = data.get("stratum_sample_size")
        return f"按你的背景约 {float(prob) * 100:.0f}%（分层样本 {n} 条）"

    if tool == "lookup_similar_admitted_cases" and isinstance(data, dict):
        if not data.get("found"):
            return "案例库里没有可比对的先例"
        n = data.get("similar_case_count")
        sim = (data.get("similarity") or {}).get("min")
        if sim is None:
            return f"找到 {n} 条背景相近的成功先例"
        return f"找到 {n} 条背景相近的成功先例（相似度 ≥{sim}）"

    if preview:
        plain = preview.replace("\n", " ")
        return plain[:160] + ("…" if len(plain) > 160 else "")
    return _tool_running_hint(tool)


def _trace_detail_lines(trace) -> list[str]:
    """把 trace 快照翻译成展示行（最近 12 条）。

    与完成后回看共用 `step_text.step_line` —— 措辞、工具标签、去 JSON 化都在那一份实现里，
    所以"进行中"和"完成后"看到的是同一批说法（历史上是两套，回看那份还带原始参数 dump）。
    """
    if trace is None or not getattr(trace, "entries", None):
        return []
    from src.agent.step_text import step_line

    lines: list[str] = []
    for entry in trace.entries:
        line = step_line(entry)
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return lines[-12:]


def _invoke_progress_cb(
    progress_cb: Callable[..., None] | None,
    steps: list[str],
    text: str,
    details: list[str] | None = None,
) -> None:
    if progress_cb is None:
        return
    detail_lines = list(details or [])
    try:
        progress_cb(steps, text, detail_lines)
    except TypeError:
        progress_cb(steps, text)


def _append_step(steps: list[str], phrase: str | None) -> bool:
    if phrase and phrase not in steps:
        steps.append(phrase)
        return True
    return False


def _append_tool_step(steps: list[str], tool_name: str) -> bool:
    return _append_step(steps, _TOOL_STEP_PHRASE.get(tool_name or ""))


def _append_steps_from_present(steps: list[str], present: dict) -> bool:
    changed = False
    for fields, phrase in _GATEWAY_FIELD_STEPS:
        if any(present.get(f) not in (None, "", []) for f in fields):
            if _append_step(steps, phrase):
                changed = True
    return changed


@dataclass
class _ProgressState:
    steps: list[str] = field(default_factory=list)
    details: list[str] = field(default_factory=list)
    model_text: str = ""
    seen_step_seq: set[int] = field(default_factory=set)
    abort: BaseException | None = None
    started: float = field(default_factory=time.monotonic)
    last_emit: float = field(default_factory=time.monotonic)
    # 有新内容待推送。循环线程只置位，真正的 UI 写入由脚本线程的 drain 完成。
    pending: bool = True

    def display_text(self) -> str:
        if self.model_text.strip():
            return self.model_text
        if self.details:
            return self.details[-1]
        return ""


def _sync_from_trace(deps: HarnessDeps, state: _ProgressState) -> bool:
    """把 trace 的新条目翻译进进度状态。返回是否有变化。"""
    trace = getattr(deps, "trace", None)
    if trace is None:
        return False
    changed = False
    for entry in trace.entries:
        if entry.seq not in state.seen_step_seq:
            state.seen_step_seq.add(entry.seq)
            if _append_tool_step(state.steps, entry.tool):
                changed = True
    lines = _trace_detail_lines(trace)
    if lines != state.details:
        state.details = lines
        changed = True
    return changed


def _mark_progress(state: _ProgressState | None) -> None:
    """在**事件循环线程**上标记"有新进度"。

    这里刻意不碰任何 Streamlit API：从非脚本线程写 `st.*` / `st.session_state` 会被
    Streamlit 静默丢弃（没有 ScriptRunContext）。真正的 UI 写入交给
    `_drain_progress`，它由 `submit(on_tick=...)` 在**脚本线程**上调用。
    """
    if state is not None:
        state.pending = True
        state.last_emit = time.monotonic()


def _drain_progress(
    progress_cb: Callable[..., None] | None, state: _ProgressState | None
) -> None:
    """把待推送的进度交给 progress_cb。**必须在脚本线程上调用。**

    progress_cb 抛出的异常（典型是 UI 的 `LeadInCancelled`）会一路传给
    `submit`，由它取消循环上的任务。
    """
    if progress_cb is None or state is None or not state.pending:
        return
    state.pending = False
    _invoke_progress_cb(progress_cb, state.steps, state.display_text(), state.details)


async def _poll_progress(
    deps: HarnessDeps,
    state: _ProgressState,
) -> None:
    """独立心跳：只读 trace 并置位 pending，绝不碰模型/工具的事件迭代器。

    注意这里不做任何 UI 调用 —— 本协程跑在事件循环线程上。
    """
    while True:
        _sync_from_trace(deps, state)
        _mark_progress(state)
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)


def _build_work_deps(deps: HarnessDeps) -> HarnessDeps:
    ctx = deps.ctx
    if ctx is None:
        return deps
    try:
        work_ctx = ctx.model_copy(deep=True)
        gw = copy.copy(deps.gateway)
        if not hasattr(gw, "_ctx"):
            _log.warning(
                "_build_work_deps | gateway 无 _ctx 属性，降级为不隔离（仅测试 Fake 应触发）"
            )
            return deps
        gw._ctx = work_ctx
        for attr in ("_applied", "_dropped_schools", "_low_confidence"):
            if hasattr(gw, attr):
                setattr(gw, attr, {} if attr == "_applied" else [])
        for attr in ("_submit_requested", "_expand_requested", "_reset_requested"):
            if hasattr(gw, attr):
                setattr(gw, attr, False)
    except Exception:
        _log.warning("_build_work_deps | 隔离构造失败，降级为不隔离", exc_info=True)
        return deps
    return HarnessDeps(
        gateway=gw,
        ctx=work_ctx,
        prediction=deps.prediction,
        explain=deps.explain,
        memory=deps.memory,
        turn=deps.turn,
        trace=ToolTurnTrace(),
    )


def _commit_work_deps(deps: HarnessDeps, work_deps: HarnessDeps) -> None:
    if work_deps is deps or deps.ctx is None:
        return
    deps.ctx.extracted_background = work_deps.ctx.extracted_background
    for attr in (
        "_applied",
        "_dropped_schools",
        "_low_confidence",
        "_submit_requested",
        "_expand_requested",
        "_reset_requested",
        "_university_alias",
        "_university_display_label",
    ):
        if hasattr(work_deps.gateway, attr) and hasattr(deps.gateway, attr):
            setattr(deps.gateway, attr, getattr(work_deps.gateway, attr))
    deps.trace.entries.extend(work_deps.trace.entries)


class LeadInToolAgent:
    """Lead-in 单智能体：ReAct 循环决定回答 / 追问 / 查数据 / 写表 / 触发预测。"""

    def __init__(self, model=None) -> None:
        self._model = model or build_model_with_fallback()
        self._agent = Agent(
            self._model,
            tools=[*FORM_TOOLS, *PROBE_TOOLS, *RECALL_TOOLS],
            output_type=str,
            instructions=build_agent_system_prompt(),
        )
        self.agent_name = "LeadInToolAgent"

    def run(
        self,
        user_input: str,
        deps: HarnessDeps,
        message_history=None,
        progress_cb: Callable[..., None] | None = None,
    ):
        text = (user_input or "").strip()
        if not text:
            return "请提供学生信息：本科院校、专业、GPA、语言成绩、目标院校。", message_history

        prompt = _build_tool_user_prompt(text, deps)
        lock = get_session_lock(_resolve_session_id(deps))
        work_deps = _build_work_deps(deps)
        timeout = _effective_turn_timeout()
        # 循环线程把 state 挂进来，脚本线程的 tick 才能排空进度
        holder: dict[str, Any] = {}

        with lock:
            try:
                final, msgs = submit(
                    self._run_async(prompt, work_deps, message_history, holder),
                    timeout=timeout,
                    on_tick=(
                        (lambda: _drain_progress(progress_cb, holder.get("state")))
                        if progress_cb is not None
                        else None
                    ),
                )
            except BaseException:
                # 超时/取消：work 副本整体丢弃，真实 ctx 保持零写入。
                deps.run_aborted = True
                raise
            # 收尾时再排空一次（此时仍在脚本线程上）
            _drain_progress(progress_cb, holder.get("state"))
            _commit_work_deps(deps, work_deps)
            return final, msgs

    def run_streaming(
        self,
        user_input: str,
        deps: HarnessDeps,
        progress_cb: Callable[..., None],
        message_history=None,
    ):
        """兼容入口：新版 run 已统一支持 progress_cb。"""
        return self.run(user_input, deps, message_history, progress_cb)

    async def _run_async(
        self,
        prompt: str,
        deps: HarnessDeps,
        message_history,
        holder: dict[str, Any],
    ) -> tuple[str, Any]:
        state = _ProgressState()
        # 让脚本线程的 tick 能拿到进度状态（引用赋值，不需要锁）
        holder["state"] = state
        _sync_from_trace(deps, state)
        _mark_progress(state)

        poll_task = asyncio.create_task(_poll_progress(deps, state))
        final = ""
        msgs = message_history
        run_t0 = time.monotonic()
        _log.info(
            "LEAD_IN_TOOL | run start | turn=%s prompt_len=%d history=%s budget=%.0fs rounds=%d",
            deps.turn,
            len(prompt),
            "yes" if message_history else "no",
            _effective_turn_timeout(),
            _max_tool_rounds(),
        )
        try:
            async with self._agent.run_stream_events(
                prompt,
                deps=deps,
                message_history=message_history,
                usage_limits=_usage_limits(),
            ) as stream:
                # 注意：这里绝不能用 wait_for 包 __anext__()——超时会关闭异步生成器，
                # 后续迭代直接 StopAsyncIteration，流被静默截断。心跳交给 poll_task。
                async for event in stream:
                    if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                        state.model_text += event.delta.content_delta
                        _mark_progress(state)
                    elif isinstance(event, AgentRunResultEvent):
                        final = str(event.result.output)
                        msgs = event.result.all_messages()
        except UsageLimitExceeded:
            _log.warning(
                "LEAD_IN_TOOL | 轮数上限 %d 用尽，按当前进展收尾",
                _max_tool_rounds(),
            )
        finally:
            poll_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await poll_task

        _sync_from_trace(deps, state)
        if final:
            state.model_text = final
        _mark_progress(state)
        _log.info(
            "LEAD_IN_TOOL | run done | elapsed=%.1fs tools=%d output_len=%d",
            time.monotonic() - run_t0,
            len(getattr(getattr(deps, "trace", None), "entries", []) or []),
            len(final or state.model_text),
        )
        return (final or state.model_text), msgs


def build_deps(session_manager, ctx: StudentContext, turn: int = 1) -> HarnessDeps:
    from src.agent.harness import build_harness_deps

    return build_harness_deps(
        session_manager,
        ctx,
        turn=turn,
    )
