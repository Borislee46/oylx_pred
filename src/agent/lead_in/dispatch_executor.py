import logging
from typing import Any

from src.agent import get_lead_in_router_agent, get_lead_in_tool_agent
from src.agent.context import StudentContext
from src.agent.lead_in.dispatch_constants import (
    MSGS_KEY,
    TOOLS_FAILED_KEY,
    TURN_KEY,
    LeadInCancelled,
)
from src.agent.lead_in.dispatch_helpers import (
    _append_conversation_turn,
    _build_router_history,
    _detail_lines_as_trace,
    _looks_extractable_profile,
    _looks_structured,
    _serialize_trace,
    _summarize_extracted_fields,
    should_auto_predict,
)
from src.agent.lead_in.dispatch_result import DispatchResult
from src.agent.lead_in.session_continuity import (
    apply_fresh_session,
    should_reset_for_new_profile,
)
from src.agent.lead_in.state_machine import LeadInPhase
from src.agent.runtime.model_factory import LEAD_IN_TOOL_STREAM_TIMEOUT

_log = logging.getLogger("LeadInDispatcher")


class _DispatchExecutorMixin:
    def _finalize_predict(
        self, session_manager: Any, ctx: StudentContext, *, had_api_error: bool
    ) -> bool:
        predict = should_auto_predict(ctx, had_api_error=had_api_error)
        session_manager.set(_lead_in_processed=predict)

        if predict and self._sm is not None:
            try:
                self._sm.transition(LeadInPhase.PREDICTING)
            except ValueError:
                _log.debug("_finalize_predict: PREDICTING transition skipped", exc_info=True)

        return predict

    def _resolve_partial_tools(
        self,
        session_manager,
        ctx,
        applied: dict,
        *,
        trace_fb: str,
        default_feedback: str,
        deps,
    ) -> DispatchResult:
        merged = {**(ctx.extracted_background or {}), **(applied or {})}
        ctx.extracted_background = merged
        from src.agent.tools.form_gateway import compute_missing_required

        missing = compute_missing_required(merged)
        if missing:
            miss_txt = "、".join(missing)
            feedback = trace_fb or default_feedback
            ask = f"还缺：{miss_txt}。请补充后继续。"
            if ask not in feedback:
                feedback = f"{feedback}\n{ask}" if feedback else ask
            return DispatchResult(
                handled=True,
                path="tools",
                feedback=feedback,
                applied_fields=applied,
                should_expand_form=True,
                should_auto_predict=False,
                trace_entries=_serialize_trace(deps),
            )
        predict = self._finalize_predict(session_manager, ctx, had_api_error=False)
        return DispatchResult(
            handled=True,
            path="tools",
            feedback=trace_fb or "已识别您的背景，请查看表单与录取方案。",
            applied_fields=applied,
            should_expand_form=True,
            should_auto_predict=predict,
            trace_entries=_serialize_trace(deps),
        )

    def _dispatch_new(
        self,
        session_manager,
        ctx,
        analyze_text,
        mode: str,
        session_state: dict | None,
    ) -> DispatchResult:
        from src.pages.prediction.ui.lead_in_echo import (
            path_narrative,
        )

        if should_reset_for_new_profile(analyze_text, ctx, session_manager):
            apply_fresh_session(session_manager, ctx, state_machine=self._sm)
            ctx.session_continuity = "fresh"
            self._progress_path_hint = path_narrative("fresh")
        else:
            ctx.session_continuity = "continue"
        session_manager.set(lead_in_consumed=False)
        ctx.intent_gate_last = None

        if self._sm is not None:
            self._sm.transition(LeadInPhase.EXTRACTING)

        self._emit_progress(
            ["解析学生背景"],
            "",
            variant="default",
            path_hint=self._progress_path_hint or path_narrative("default"),
            details=["已收到输入，准备结构化提取"],
        )

        if mode == "harness":
            _log.warning("DISPATCH | mode=harness 已废弃（3b 删除），回退 → router")
            return self._try_router(session_manager, ctx, analyze_text)

        if mode == "tools":
            prefer_router = _looks_structured(analyze_text) or _looks_extractable_profile(
                analyze_text
            )
            if prefer_router:
                _log.info("DISPATCH | fast path → router (extractable profile)")
                self._emit_progress(
                    ["解析学生背景", "结构化提取"],
                    "",
                    path_hint=path_narrative("router"),
                    details=["已收到输入，准备结构化提取", "模型正在提取院校 / 专业 / 成绩等字段"],
                )
                result = self._try_router(session_manager, ctx, analyze_text)
                if result.handled:
                    if session_state is not None:
                        session_state.pop(TOOLS_FAILED_KEY, None)
                    return result
                _log.warning("DISPATCH | fast path router 未产出，继续 tools")

            skip_tools = bool(
                (self._sm is not None and self._sm.get_state().tools_failed)
                or (session_state and session_state.get(TOOLS_FAILED_KEY))
            )
            if skip_tools:
                _log.info("DISPATCH | tools 已失败，重试直接走 router")
            else:
                self._emit_progress(
                    ["解析学生背景", "逐步填表"],
                    "",
                    path_hint=path_narrative("tools"),
                    details=["信息较散，改为逐步对齐填表"],
                )
                result = self._try_tools(session_manager, ctx, analyze_text)
                if result.handled:
                    if session_state is not None:
                        session_state.pop(TOOLS_FAILED_KEY, None)
                    if self._sm is not None:
                        self._sm.update(tools_failed=False)
                    return result
                if session_state is not None:
                    session_state[TOOLS_FAILED_KEY] = True
                if self._sm is not None:
                    self._sm.update(tools_failed=True)
                _log.warning("DISPATCH | tools 未产出字段，fallback → router")
            result = self._try_router(session_manager, ctx, analyze_text)
            if result.handled:
                if session_state is not None:
                    session_state.pop(TOOLS_FAILED_KEY, None)
                if self._sm is not None:
                    self._sm.update(tools_failed=False)
                return result
        else:
            result = self._try_router(session_manager, ctx, analyze_text)
            if result.handled:
                return result

        _log.error("DISPATCH | tools + router 均失败")
        return DispatchResult(handled=False, path=mode, error="all_paths_failed")

    def _try_tools(self, session_manager, ctx, analyze_text) -> DispatchResult:
        deps = None
        history = None
        try:
            from src.agent.lead_in.tool_agent import build_deps
            from src.utils.logger import ensure_lead_in_console_logging

            ensure_lead_in_console_logging()
            _log.info(
                "DISPATCH | tools start | turn=%d text_len=%d continuity=%s",
                int(session_manager.get(TURN_KEY, 0)) + 1,
                len(analyze_text or ""),
                getattr(ctx, "session_continuity", "continue"),
            )

            turn = int(session_manager.get(TURN_KEY, 0)) + 1
            session_manager.set(**{TURN_KEY: turn})

            ctx.raw_input = analyze_text
            deps = build_deps(session_manager, ctx, turn=turn)
            history = session_manager.get(MSGS_KEY)

            tool_agent = get_lead_in_tool_agent()
            try:
                if self.on_progress is not None:
                    try:
                        raw = tool_agent.run_streaming(
                            analyze_text,
                            deps,
                            message_history=history,
                            progress_cb=self.on_progress,
                        )
                    except TimeoutError:
                        raise
                    except LeadInCancelled:
                        raise
                    except Exception:
                        _log.warning(
                            "DISPATCH | run_streaming 失败，回退非流式 run | len=%d",
                            len(analyze_text or ""),
                            exc_info=True,
                        )
                        raw = tool_agent.run(
                            analyze_text,
                            deps,
                            message_history=history,
                        )
                else:
                    raw = tool_agent.run(
                        analyze_text,
                        deps,
                        message_history=history,
                    )
                if isinstance(raw, dict) and raw.get("_error"):
                    raise RuntimeError(str(raw["_error"]))
                feedback, new_msgs = raw
            finally:
                if not getattr(self, "_cancelled", False):
                    deps.gateway.flush()
                    if deps.prediction is not None:
                        deps.prediction.flush()

            applied = deps.gateway.applied_fields()
            if not applied:
                return DispatchResult(
                    handled=False,
                    path="tools",
                    error="no_fields_produced",
                )

            session_manager.set(**{MSGS_KEY: new_msgs})

            trace_entries = _serialize_trace(deps)

            _append_conversation_turn(ctx, "user", analyze_text, sm=self._sm)
            _append_conversation_turn(ctx, "assistant", feedback or "", sm=self._sm)

            from src.pages.prediction.ui.lead_in_progress import entries_to_detail_lines

            tool_details = entries_to_detail_lines(trace_entries)
            if self._sm is not None:
                self._sm.transition(LeadInPhase.AWAITING)
                self._sm.update(
                    last_path="tools",
                    last_applied_fields=applied,
                    last_trace=trace_entries,
                    progress_details=tool_details,
                    conversation_turns=list(ctx.conversation_turns or []),
                    turn=int(session_manager.get(TURN_KEY, turn)),
                )
            if tool_details:
                self._emit_progress(
                    ["解析学生背景", "逐步填表", "完成"],
                    "",
                    details=tool_details,
                )

            predict = self._finalize_predict(session_manager, ctx, had_api_error=False)

            return DispatchResult(
                handled=True,
                path="tools",
                feedback=feedback or "",
                applied_fields=applied,
                low_confidence_fields=session_manager.get("lead_in_low_confidence_fields"),
                should_expand_form=True,
                should_auto_predict=predict,
                trace_entries=trace_entries,
            )
        except LeadInCancelled:
            raise
        except TimeoutError:
            from src.agent.lead_in.tool_agent import feedback_from_trace

            _log.warning(
                "DISPATCH | tools 超时 (%ds)，检查 partial work",
                LEAD_IN_TOOL_STREAM_TIMEOUT,
            )
            if deps is None or getattr(deps, "run_aborted", False):
                return DispatchResult(handled=False, path="tools", error="timeout")
            applied = deps.gateway.applied_fields()
            if applied:
                _log.info(
                    "DISPATCH | timeout 但已写入 %d 字段，按 partial success 返回", len(applied)
                )
                if history is not None:
                    session_manager.set(**{MSGS_KEY: history})
                return self._resolve_partial_tools(
                    session_manager,
                    ctx,
                    applied,
                    trace_fb=feedback_from_trace(getattr(deps, "trace", None)),
                    default_feedback="已识别部分背景，请核对表单。",
                    deps=deps,
                )
            return DispatchResult(handled=False, path="tools", error="timeout")
        except Exception:
            from src.agent.lead_in.tool_agent import feedback_from_trace

            _log.exception("DISPATCH | tools 路径异常，检查 partial work")
            if deps is None or getattr(deps, "run_aborted", False):
                return DispatchResult(handled=False, path="tools", error="exception")
            applied = deps.gateway.applied_fields()
            if applied:
                _log.info("DISPATCH | 异常但已写入 %d 字段，按 partial success 返回", len(applied))
                if history is not None:
                    session_manager.set(**{MSGS_KEY: history})
                return self._resolve_partial_tools(
                    session_manager,
                    ctx,
                    applied,
                    trace_fb=feedback_from_trace(getattr(deps, "trace", None)),
                    default_feedback="已识别部分背景，请核对表单。",
                    deps=deps,
                )
            return DispatchResult(handled=False, path="tools", error="exception")

    def _try_router(self, session_manager, ctx, analyze_text) -> DispatchResult:
        try:
            from src.pages.prediction.ui.lead_in_echo import path_narrative

            details: list[str] = ["已收到输入，准备结构化提取"]
            self._emit_progress(
                ["解析学生背景"],
                "",
                details=details,
                path_hint=path_narrative("router"),
            )

            router_agent = get_lead_in_router_agent()
            history = _build_router_history(ctx, session_manager)
            session_id = getattr(session_manager, "session_id", None)

            details.append("模型正在提取院校 / 专业 / 成绩等字段")
            self._emit_progress(
                ["解析学生背景", "结构化提取"],
                "",
                details=details,
            )

            decision = router_agent.run(analyze_text, history=history, session_id=session_id)

            extracted = decision.to_extracted_info()
            if not extracted:
                return DispatchResult(
                    handled=False,
                    path="router",
                    error="empty_extraction",
                )

            details.append(_summarize_extracted_fields(extracted))
            details.append(f"意图={decision.intent} · 置信={decision.confidence}")
            self._emit_progress(
                ["解析学生背景", "结构化提取", "写入表单"],
                "",
                details=details,
            )

            ctx.raw_input = analyze_text
            ctx.extracted_background = {
                **(ctx.extracted_background or {}),
                **extracted,
            }
            ctx.quick_assessment = decision.feedback

            from src.pages.prediction.form_bridge import apply_lead_in_to_form

            applied = apply_lead_in_to_form(ctx, session_manager)
            details.append(
                f"已写入表单 {len(applied)} 项"
                if applied
                else "未写入新字段（可能与表单已有值一致）"
            )

            extra = (
                decision.clarifying_question if decision.next_action == "ask_clarification" else ""
            )
            combined = decision.feedback
            if extra and extra not in combined:
                combined = f"{combined}\n{extra}" if combined else extra

            should_expand = decision.next_action in ("fill_only", "ask_clarification")
            if decision.intent != "profile" or decision.confidence == "low":
                predict = False
                session_manager.set(_lead_in_processed=False)
                should_expand = True
                details.append("置信偏低或非背景提取，展开表单供确认")
            else:
                if self._sm is not None and self._sm.is_extracting():
                    self._sm.transition(LeadInPhase.AWAITING)
                predict = self._finalize_predict(session_manager, ctx, had_api_error=False)
                if not predict:
                    should_expand = True
                    details.append("核心字段未齐，等待补充后再预测")
                else:
                    details.append("核心字段齐全，即将生成录取方案")

            self._emit_progress(
                ["解析学生背景", "结构化提取", "写入表单", "完成"],
                "",
                details=details,
            )

            questions: list[str] = []
            if extra:
                questions.append(extra)

            _append_conversation_turn(ctx, "user", analyze_text, sm=self._sm)
            _append_conversation_turn(ctx, "assistant", combined, sm=self._sm)

            router_trace = _detail_lines_as_trace(details)
            if self._sm is not None:
                if self._sm.is_extracting():
                    self._sm.transition(LeadInPhase.AWAITING)
                self._sm.update(
                    last_path="router",
                    last_applied_fields=applied,
                    clarifying_questions=questions,
                    conversation_turns=list(ctx.conversation_turns or []),
                    progress_details=details,
                    last_trace=router_trace,
                )

            return DispatchResult(
                handled=True,
                path="router",
                feedback=combined,
                applied_fields=applied,
                low_confidence_fields=session_manager.get("lead_in_low_confidence_fields"),
                should_expand_form=should_expand,
                should_auto_predict=predict,
                clarifying_questions=questions,
                trace_entries=router_trace,
            )
        except LeadInCancelled:
            raise
        except Exception:
            _log.exception("DISPATCH | router 路径异常")
            if self._sm is not None:
                self._sm.transition(LeadInPhase.ERROR)
                self._sm.update(last_path="router", last_error="exception")
            return DispatchResult(handled=False, path="router", error="exception")
