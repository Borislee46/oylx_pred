import logging
from typing import Any

from src.agent import get_lead_in_tool_agent
from src.agent.context import StudentContext
from src.agent.lead_in.dispatch_constants import (
    MSGS_KEY,
    TOOLS_FAILED_KEY,
    TURN_KEY,
    LeadInCancelled,
)
from src.agent.lead_in.dispatch_helpers import (
    _append_conversation_turn,
    _serialize_trace,
    should_auto_predict,
)
from src.agent.lead_in.dispatch_result import DispatchResult
from src.agent.lead_in.session_continuity import (
    apply_fresh_session,
    should_reset_for_new_profile,
)
from src.agent.lead_in.state_machine import LeadInPhase

_log = logging.getLogger("LeadInDispatcher")

PATH = "agent"

# 「本人背景」类字段：只有写入这些才说明"用户在提供自己的背景"。
# 目标类字段（target_*）不算 —— 用户问"港大港中文的雅思要求"时模型也会写 target_schools，
# 但那不是在陈述自己的背景，不该据此触发预测（实测线上 turn 6 就是这么误触发的）。
_BACKGROUND_APPLIED_KEYS = frozenset(
    {
        "gpa",
        "gpa_scale",
        "language_type",
        "language_score",
        "standardized_test_type",
        "standardized_test_score",
        "research",
        "research_count",
        "internship",
        "internship_count",
        "paper",
        "paper_count",
        "award",
        "award_count",
    }
)


def _is_background_application(applied: dict[str, Any] | None) -> bool:
    for key in applied or {}:
        name = str(key)
        if name.startswith("background_") or name in _BACKGROUND_APPLIED_KEYS:
            return True
    return False


def _trim_message_history(messages: Any, *, keep_user_turns: int = 4) -> Any:
    """只保留最近 N 个用户回合，避免 message_history 无限增长。

    实测未截断时 prompt_len 逐轮增长（23 → 190 → 284 → 403 → 522 → 570），
    延迟和成本都在涨（turn 4/5 都要 35s+）。
    只在**用户回合边界**切，避免留下孤立的工具返回（那会让下一次调用直接报错）。
    """
    if not messages:
        return messages
    try:
        from pydantic_ai.messages import ModelRequest, UserPromptPart
    except Exception:
        return messages

    starts = [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in (getattr(msg, "parts", None) or []))
    ]
    if len(starts) <= keep_user_turns:
        return messages
    return list(messages)[starts[-keep_user_turns] :]


class _DispatchExecutorMixin:
    def _finalize_predict(
        self,
        session_manager: Any,
        ctx: StudentContext,
        *,
        had_api_error: bool,
        applied_this_turn: bool = True,
        submit_requested: bool = False,
        hold_requested: bool = False,
    ) -> bool:
        """是否触发预测：两把钥匙 + 一个刹车。

        **钥匙1（意图，只有 LLM 能判）**：本轮真的把字段写进了表单。
        智能体被要求：只有"用户本人在提供自己的背景"时才写入；转述别人的情况（"我同学…"）
        或纯咨询时不写。所以"写入"本身就是意图信号。

        **钥匙2（数据，必须客观判）**：四核心（院校/专业/GPA/语言成绩）齐全。
        不能让 LLM 数数——它会漏、会算错。

        **刹车（显式否定）**：智能体调了 `expand_form` 却没调 `submit_prediction`，
        表示"先让用户核对/别急着出方案"（例如用户说"先别算"，或数据自相矛盾）。

        `predict = 钥匙1 AND 钥匙2 AND NOT 刹车`。

        为什么不把 `submit_requested` 当闸门：模型忘了调工具时用户就永远拿不到方案，
        而"漏出方案"比"多出一次方案"更贵。所以模型沉默时用客观兜底，不引入新的漏触发风险。
        """
        if had_api_error or not applied_this_turn:
            predict = False
        elif hold_requested and not submit_requested:
            predict = False
        else:
            predict = should_auto_predict(ctx, had_api_error=False)

        _log.info(
            "PREDICT DECISION | applied=%s submit=%s hold=%s → predict=%s",
            applied_this_turn,
            submit_requested,
            hold_requested,
            predict,
        )

        session_manager.set(_lead_in_processed=predict)

        if predict and self._sm is not None:
            try:
                self._sm.transition(LeadInPhase.PREDICTING)
            except ValueError:
                _log.debug("_finalize_predict: PREDICTING transition skipped", exc_info=True)

        return predict

    def _dispatch_new(
        self,
        session_manager,
        ctx,
        analyze_text,
        session_state: dict | None,
    ) -> DispatchResult:
        from src.pages.prediction.ui.lead_in_echo import path_narrative

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
            details=["已收到输入，交给顾问智能体判断"],
        )

        # 单一执行体：由智能体自己决定「回答 / 追问 / 查数据 / 写表单 / 触发预测」。
        # 不再有 router-first 分流——分流会让"提问型输入"既拿不到回答、又白跑一轮工具循环。
        result = self._try_agent(session_manager, ctx, analyze_text)
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
        return DispatchResult(
            handled=False,
            path=PATH,
            error=result.error or "all_paths_failed",
        )

    def _reset_for_new_profile(self, session_manager: Any, ctx: StudentContext) -> None:
        """智能体判定「换了个学生」→ 清掉上一位的档案。

        时序很重要：agent.run 内部的 `_commit_work_deps` 已经把本轮新写的字段提交到
        `ctx.extracted_background` 了，所以这里先把它们取出来，重置完再放回去 ——
        否则会把本轮刚识别到的字段一起清掉。
        重置必须发生在 `gateway.flush()` **之前**，这样 flush 只写新档案。
        """
        from src.agent.lead_in.session_continuity import apply_fresh_session

        carried = dict(ctx.extracted_background or {})
        apply_fresh_session(session_manager, ctx, state_machine=self._sm)
        ctx.extracted_background = carried
        _log.info(
            "DISPATCH | reset for new profile | carried_fields=%s",
            sorted(carried.keys()),
        )

    def _try_agent(
        self,
        session_manager,
        ctx,
        analyze_text,
    ) -> DispatchResult:
        deps = None
        try:
            from src.agent.lead_in.tool_agent import build_deps
            from src.utils.logger import ensure_lead_in_console_logging

            ensure_lead_in_console_logging()
            turn = int(session_manager.get(TURN_KEY, 0)) + 1
            session_manager.set(**{TURN_KEY: turn})
            _log.info(
                "DISPATCH | agent start | turn=%d text_len=%d continuity=%s",
                turn,
                len(analyze_text or ""),
                getattr(ctx, "session_continuity", "continue"),
            )

            ctx.raw_input = analyze_text
            deps = build_deps(session_manager, ctx, turn=turn)
            history = session_manager.get(MSGS_KEY)
            agent = get_lead_in_tool_agent()

            raw = agent.run(
                analyze_text,
                deps,
                message_history=history,
                progress_cb=self.on_progress,
            )
            if isinstance(raw, dict) and raw.get("_error"):
                raise RuntimeError(str(raw["_error"]))
            feedback, new_msgs = raw

            # 智能体判定"换了个学生" → 先清旧档案，再落地本轮字段
            if getattr(deps.gateway, "_reset_requested", False):
                self._reset_for_new_profile(session_manager, ctx)

            if not getattr(self, "_cancelled", False):
                deps.gateway.flush()
                if deps.prediction is not None:
                    deps.prediction.flush()

            applied = deps.gateway.applied_fields()
            submit_requested = bool(getattr(deps.gateway, "_submit_requested", False))
            expand_requested = bool(getattr(deps.gateway, "_expand_requested", False))
            feedback = feedback or ""

            # 纯答疑/追问：没写字段但有回复 —— 这就是一次成功交付。
            # 旧实现把它判成 no_fields_produced → all_paths_failed，用户提问拿不到任何回答。
            if not applied and not feedback.strip():
                return DispatchResult(
                    handled=False,
                    path=PATH,
                    error="empty_output",
                )

            if new_msgs is not None:
                session_manager.set(**{MSGS_KEY: _trim_message_history(new_msgs)})

            trace_entries = _serialize_trace(deps)
            _append_conversation_turn(ctx, "user", analyze_text, sm=self._sm)
            _append_conversation_turn(ctx, "assistant", feedback, sm=self._sm)

            from src.pages.prediction.ui.lead_in_progress_copy import entries_to_detail_lines

            tool_details = entries_to_detail_lines(trace_entries)
            if self._sm is not None:
                self._sm.transition(LeadInPhase.AWAITING)
                self._sm.update(
                    last_path=PATH,
                    last_applied_fields=applied,
                    last_trace=trace_entries,
                    progress_details=tool_details,
                    conversation_turns=list(ctx.conversation_turns or []),
                    turn=int(session_manager.get(TURN_KEY, turn)),
                )
            if tool_details:
                self._emit_progress(
                    ["解析学生背景", "结构化提取", "完成"],
                    "",
                    details=tool_details,
                )

            predict = self._finalize_predict(
                session_manager,
                ctx,
                had_api_error=False,
                applied_this_turn=_is_background_application(applied),
                submit_requested=submit_requested,
                hold_requested=expand_requested,
            )

            return DispatchResult(
                handled=True,
                path=PATH,
                feedback=feedback,
                applied_fields=applied,
                low_confidence_fields=session_manager.get("lead_in_low_confidence_fields"),
                should_expand_form=bool(applied) or expand_requested,
                should_auto_predict=predict,
                trace_entries=trace_entries,
            )
        except LeadInCancelled:
            raise
        except TimeoutError:
            _log.warning(
                "DISPATCH | agent 超时，work 副本已丢弃（隔离语义：不留半截状态）",
            )
            return DispatchResult(handled=False, path=PATH, error="timeout")
        except Exception:
            _log.exception("DISPATCH | agent 路径异常")
            return DispatchResult(handled=False, path=PATH, error="exception")
