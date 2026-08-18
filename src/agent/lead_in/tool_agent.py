import asyncio
import copy
import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic_ai import Agent, AgentRunResultEvent
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    TextPartDelta,
)

from src.agent.context import StudentContext
from src.agent.field_labels import FIELD_LABEL
from src.agent.harness import HarnessDeps, ToolTurnTrace, TraceEntry
from src.agent.lead_in.router_prompts import build_tool_agent_system_prompt
from src.agent.lead_in.session_lock import get_session_lock
from src.agent.runtime.model_factory import (
    LEAD_IN_TOOL_STREAM_TIMEOUT,
    build_model_with_fallback,
)
from src.agent.tools.form_tools import FORM_TOOLS

_log = logging.getLogger("lead_in_tool_agent")


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


_HEARTBEAT_SECONDS = 8

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
    if parts:
        return parts[-1]
    return "已识别部分背景并写入表单，请核对后补充。"


def prediction_triggered_in_trace(trace) -> bool:
    if trace is None:
        return False
    for entry in trace.entries:
        if entry.tool != "submit_prediction" or not entry.ok:
            continue
        preview = (entry.result_preview or "").strip()
        if preview.startswith("已触发"):
            return True
        data = _try_parse_json(preview)
        if isinstance(data, dict) and data.get("ok"):
            return True
    return False


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
        return preview or "已展开表单供核对"

    if preview:
        plain = preview.replace("\n", " ")
        return plain[:160] + ("…" if len(plain) > 160 else "")
    return _tool_running_hint(tool)


def _sync_trace_to_progress(
    trace,
    steps: list[str],
    seen_step_seq: set[int],
    summarized_seq: set[int],
) -> str:
    buffer = ""
    for entry in trace.entries:
        if entry.seq not in seen_step_seq:
            seen_step_seq.add(entry.seq)
            if _append_tool_step(steps, entry.tool):
                buffer = _tool_running_hint(entry.tool)
        if entry.result_preview and entry.seq not in summarized_seq:
            summarized_seq.add(entry.seq)
            buffer = format_trace_step_summary(entry)
    return buffer


def _trace_detail_lines(trace) -> list[str]:
    if trace is None or not getattr(trace, "entries", None):
        return []
    lines: list[str] = []
    for entry in trace.entries:
        line = (
            format_trace_step_summary(entry)
            if entry.result_preview
            else _tool_running_hint(entry.tool)
        )
        line = (line or "").strip()
        if line and (not lines or lines[-1] != line):
            lines.append(line)
    return lines[-12:]


def _invoke_progress_cb(
    progress_cb: Callable[[list[str], str], None] | None,
    steps: list[str],
    text: str,
    trace=None,
) -> None:
    if progress_cb is None:
        return
    details = _trace_detail_lines(trace)
    try:
        progress_cb(steps, text, details)
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


def _handle_stream_event(event, steps: list[str], progress_cb, buffer: str) -> str:
    if isinstance(event, FunctionToolCallEvent):
        if _append_tool_step(steps, event.part.tool_name):
            progress_cb(steps, buffer)
    elif isinstance(event, FunctionToolResultEvent):
        part = event.part
        tool_name = getattr(part, "tool_name", "") or ""
        if _append_tool_step(steps, tool_name):
            progress_cb(steps, buffer)
    elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
        buffer += event.delta.content_delta
        progress_cb(steps, buffer)
    return buffer


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
        for attr in ("_submit_requested", "_expand_requested"):
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
        "_university_alias",
        "_university_display_label",
    ):
        if hasattr(work_deps.gateway, attr) and hasattr(deps.gateway, attr):
            setattr(deps.gateway, attr, getattr(work_deps.gateway, attr))
    deps.trace.entries.extend(work_deps.trace.entries)


class LeadInToolAgent:
    def __init__(self, model=None) -> None:
        self._model = model or build_model_with_fallback()
        self._agent = Agent(
            self._model,
            tools=FORM_TOOLS,
            output_type=str,
            instructions=build_tool_agent_system_prompt(),
        )
        self.agent_name = "LeadInToolAgent"

    def run(
        self,
        user_input: str,
        deps: HarnessDeps,
        message_history=None,
        progress_cb: Callable[[list[str], str], None] | None = None,
    ):
        text = (user_input or "").strip()
        if not text:
            return "请提供学生信息：本科院校、专业、GPA、语言成绩、目标院校。", message_history
        prompt = _build_tool_user_prompt(text, deps)
        lock = get_session_lock(_resolve_session_id(deps))
        work_deps = _build_work_deps(deps)
        if progress_cb is not None:
            with lock:
                try:
                    final, msgs = asyncio.run(
                        self._async_run_with_trace(prompt, work_deps, message_history, progress_cb)
                    )
                except BaseException:
                    deps.run_aborted = True
                    raise
                _commit_work_deps(deps, work_deps)
                return final, msgs
        with lock:
            try:
                result = asyncio.run(
                    asyncio.wait_for(
                        asyncio.to_thread(
                            self._agent.run_sync,
                            prompt,
                            deps=work_deps,
                            message_history=message_history,
                        ),
                        timeout=LEAD_IN_TOOL_STREAM_TIMEOUT,
                    )
                )
            except BaseException:
                deps.run_aborted = True
                raise
            _commit_work_deps(deps, work_deps)
        return str(result.output), result.all_messages()

    async def _async_run_with_trace(
        self,
        prompt: str,
        deps: HarnessDeps,
        message_history,
        progress_cb: Callable[[list[str], str], None],
    ):
        steps: list[str] = []
        buffer = ""
        seen_step_seq: set[int] = set()
        summarized_seq: set[int] = set()
        stop_poll = asyncio.Event()
        run_t0 = time.monotonic()
        _log.info(
            "LEAD_IN_TOOL | async_run start | turn=%d prompt_len=%d history=%s",
            deps.turn,
            len(prompt),
            "yes" if message_history else "no",
        )

        async def _poll_trace():
            nonlocal buffer
            last_hb = time.monotonic()
            while not stop_poll.is_set():
                trace = getattr(deps, "trace", None)
                if trace is not None and trace.entries:
                    prev = buffer
                    new_buf = _sync_trace_to_progress(trace, steps, seen_step_seq, summarized_seq)
                    if new_buf:
                        buffer = new_buf
                    if buffer != prev:
                        _invoke_progress_cb(progress_cb, steps, buffer, trace)
                now = time.monotonic()
                if now - last_hb >= 5.0:
                    last_hb = now
                    elapsed = now - run_t0
                    n_tools = len(trace.entries) if trace else 0
                    last = trace.entries[-1] if trace and trace.entries else None
                    if last is None:
                        phase = "starting"
                        hb_text = "正在连接模型，请稍候…"
                    elif not last.result_preview:
                        phase = f"tool_running:{last.tool}"
                        hb_text = buffer or _tool_running_hint(last.tool)
                    else:
                        phase = f"llm_pending:after_{last.tool}"
                        if elapsed >= max(20.0, LEAD_IN_TOOL_STREAM_TIMEOUT * 0.6):
                            hb_text = "已写入部分字段，正在收尾…"
                        else:
                            hb_text = buffer or "模型继续分析中…"
                    if not steps:
                        steps.append("解析学生背景")
                    _invoke_progress_cb(progress_cb, steps, hb_text, trace)
                    _log.info(
                        "LEAD_IN_TOOL | heartbeat | elapsed=%.0fs tools=%d phase=%s",
                        elapsed,
                        n_tools,
                        phase,
                    )
                try:
                    await asyncio.wait_for(stop_poll.wait(), timeout=0.3)
                    break
                except TimeoutError:
                    continue

        poll_task = asyncio.create_task(_poll_trace())
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._agent.run_sync,
                    prompt,
                    deps=deps,
                    message_history=message_history,
                ),
                timeout=LEAD_IN_TOOL_STREAM_TIMEOUT,
            )
            final = str(result.output)
            msgs = result.all_messages()

            trace = getattr(deps, "trace", None)
            if trace is not None:
                _sync_trace_to_progress(trace, steps, seen_step_seq, summarized_seq)
            _invoke_progress_cb(progress_cb, steps, final, trace)

            _log.info(
                "LEAD_IN_TOOL | async_run done | elapsed=%.1fs tools=%d output_len=%d",
                time.monotonic() - run_t0,
                len(trace.entries) if trace else 0,
                len(final),
            )
            return final, msgs
        except TimeoutError:
            _log.warning(
                "LEAD_IN_TOOL | async_run timeout | elapsed=%.0fs limit=%ds tools=%d",
                time.monotonic() - run_t0,
                LEAD_IN_TOOL_STREAM_TIMEOUT,
                len(getattr(deps.trace, "entries", []) or []),
            )
            raise
        finally:
            stop_poll.set()
            poll_task.cancel()
            try:
                await poll_task
            except asyncio.CancelledError:
                pass

    def run_streaming(
        self,
        user_input: str,
        deps: HarnessDeps,
        progress_cb: Callable[[list[str], str], None],
        message_history=None,
    ):
        text = (user_input or "").strip()
        if not text:
            return "请提供学生信息：本科院校、专业、GPA、语言成绩、目标院校。", message_history
        prompt = _build_tool_user_prompt(text, deps)
        lock = get_session_lock(_resolve_session_id(deps))
        work_deps = _build_work_deps(deps)
        with lock:
            try:
                final, msgs = asyncio.run(
                    self._stream(prompt, work_deps, message_history, progress_cb)
                )
            except BaseException:
                deps.run_aborted = True
                raise
            _commit_work_deps(deps, work_deps)
            return final, msgs

    async def _stream(self, prompt, deps, message_history, progress_cb):
        steps: list[str] = []
        buffer = ""
        final = ""
        msgs = message_history
        try:
            seen_present: set[str] = set((deps.gateway.read().get("present") or {}).keys())
        except Exception:
            seen_present = set()
        stop_poll = asyncio.Event()

        async def _poll_gateway_steps():
            nonlocal buffer
            while not stop_poll.is_set():
                try:
                    present = deps.gateway.read().get("present") or {}
                    new_keys = set(present.keys()) - seen_present
                    if new_keys:
                        seen_present.update(new_keys)
                        if _append_steps_from_present(steps, present):
                            progress_cb(steps, buffer)
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(stop_poll.wait(), timeout=0.35)
                    break
                except TimeoutError:
                    continue

        async def _consume():
            nonlocal final, msgs, buffer
            poll_task = asyncio.create_task(_poll_gateway_steps())
            try:
                async with self._agent.run_stream_events(
                    prompt, deps=deps, message_history=message_history
                ) as stream:
                    stream_aiter = stream.__aiter__()
                    while True:
                        try:
                            event = await asyncio.wait_for(
                                stream_aiter.__anext__(),
                                timeout=_HEARTBEAT_SECONDS,
                            )
                        except StopAsyncIteration:
                            break
                        except TimeoutError:
                            if buffer:
                                _invoke_progress_cb(progress_cb, steps, buffer)
                            continue

                        buffer = _handle_stream_event(event, steps, progress_cb, buffer)
                        if isinstance(event, AgentRunResultEvent):
                            final = str(event.result.output)
                            msgs = event.result.all_messages()
            finally:
                stop_poll.set()
                poll_task.cancel()
                try:
                    await poll_task
                except asyncio.CancelledError:
                    pass
                present = deps.gateway.read().get("present") or {}
                if _append_steps_from_present(steps, present):
                    progress_cb(steps, buffer)

        await asyncio.wait_for(_consume(), timeout=LEAD_IN_TOOL_STREAM_TIMEOUT)
        return (final or buffer), msgs


def build_deps(session_manager, ctx: StudentContext, turn: int = 1) -> HarnessDeps:
    from src.agent.harness import build_harness_deps

    return build_harness_deps(
        session_manager,
        ctx,
        turn=turn,
    )
