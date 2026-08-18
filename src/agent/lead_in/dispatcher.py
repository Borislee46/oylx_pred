import hashlib
import logging
import time
from collections.abc import Callable
from typing import Any

from src.agent.context import StudentContext
from src.agent.lead_in.dispatch_constants import (
    DISMISS_KEY,
    IN_PROGRESS_KEY,
    INTENT_BLOCKED_KEY,
    MAX_RETRIES,
    MAX_RETRY_ELAPSED,
    PENDING_KEY,
    PROGRESS_STEPS_KEY,
    PROGRESS_TEXT_KEY,
    PROGRESS_VARIANT_KEY,
    RETRY_COOLDOWN,
    RETRY_COUNT_KEY,
    RUNNING_HASH_KEY,
    RUNNING_TS_KEY,
    TOOLS_FAILED_KEY,
)
from src.agent.lead_in.dispatch_executor import _DispatchExecutorMixin
from src.agent.lead_in.dispatch_result import DispatchResult
from src.agent.lead_in.state_machine import (
    LeadInPhase,
    LeadInTurnStateMachine,
)

_log = logging.getLogger("LeadInDispatcher")


class LeadInDispatcher(_DispatchExecutorMixin):
    def __init__(
        self,
        on_progress: Callable[[list[str], str], None] | None = None,
        state_machine: LeadInTurnStateMachine | None = None,
    ):
        self.on_progress = on_progress
        self._progress_variant = "default"
        self._progress_path_hint = ""
        self._sm: LeadInTurnStateMachine | None = state_machine

    def _emit_progress(
        self,
        steps: list[str],
        text: str,
        *,
        variant: str = "default",
        path_hint: str | None = None,
        details: list[str] | None = None,
    ) -> None:
        self._progress_variant = variant
        if path_hint is not None:
            self._progress_path_hint = path_hint
        detail_lines = list(details or [])
        if self._sm is not None:
            self._sm.update(
                progress_steps=list(steps),
                progress_text=text,
                progress_variant=variant,
                progress_details=detail_lines,
            )
        if self.on_progress:
            try:
                self.on_progress(steps, text, detail_lines)
            except TypeError:
                self.on_progress(steps, text)

    def _sync_flat(
        self,
        session_state: dict,
        *,
        pending: str | None = None,
        in_progress: bool | None = None,
        running_hash: str | None = None,
        running_ts: float | None = None,
        retry_count: int | None = None,
        tools_failed: bool | None = None,
        clear_pending: bool = False,
        clear_tools_failed: bool = False,
        clear_trace: bool = False,
    ) -> None:
        if clear_pending:
            session_state.pop(PENDING_KEY, None)
        elif pending is not None:
            session_state[PENDING_KEY] = pending
        if in_progress is not None:
            session_state[IN_PROGRESS_KEY] = in_progress
        if running_hash is not None:
            session_state[RUNNING_HASH_KEY] = running_hash
        if running_ts is not None:
            session_state[RUNNING_TS_KEY] = running_ts
        if retry_count is not None:
            session_state[RETRY_COUNT_KEY] = retry_count
        if clear_tools_failed:
            session_state.pop(TOOLS_FAILED_KEY, None)
        elif tools_failed is not None:
            session_state[TOOLS_FAILED_KEY] = tools_failed
        if clear_trace:
            session_state.pop("_lead_in_last_trace", None)

    def dispatch(
        self,
        session_manager: Any,
        ctx: StudentContext,
        analyze_text: str,
        session_state: dict | None = None,
    ) -> DispatchResult:
        mode = self._mode()
        _log.info(
            "DISPATCH | mode=%s len=%d text=%s",
            mode,
            len(analyze_text),
            analyze_text[:80],
        )
        result = self._dispatch_new(session_manager, ctx, analyze_text, mode, session_state)
        _log.info(
            "DISPATCH RESULT | path=%s handled=%s fields=%d should_predict=%s error=%s",
            result.path,
            result.handled,
            len(result.applied_fields),
            result.should_auto_predict,
            result.error,
        )
        return result

    def enqueue(self, new_text: str | None, session_state: dict) -> str | None:
        if not (new_text and str(new_text).strip()):
            return None
        text = str(new_text).strip()
        session_state[DISMISS_KEY] = False
        session_state.pop(INTENT_BLOCKED_KEY, None)
        session_state.pop(PROGRESS_VARIANT_KEY, None)
        session_state.pop(PROGRESS_STEPS_KEY, None)
        session_state.pop(PROGRESS_TEXT_KEY, None)

        if self._sm is not None:
            state = self._sm.get_state()
            saved_turn = state.turn
            saved_messages = list(state.pydantic_messages)
            saved_conversation = list(state.conversation_turns)
            state.reset_for_new_turn()
            self._sm.update(
                phase=LeadInPhase.IDLE,
                pending_text=text,
                turn=saved_turn,
                pydantic_messages=saved_messages,
                conversation_turns=saved_conversation,
                feedback_dismissed=False,
                progress_variant="default",
                progress_steps=[],
                progress_text="",
                intent_blocked=False,
                last_path="",
                last_error=None,
            )
        self._sync_flat(session_state, pending=text, in_progress=False)
        return text

    def dequeue(self, session_state: dict) -> str | None:
        if self._sm is not None:
            pending = (self._sm.get_state().pending_text or "").strip()
            if pending:
                return pending
        pending = session_state.get(PENDING_KEY)
        if not (pending and str(pending).strip()):
            return None
        return str(pending).strip()

    def is_recovering(self, session_state: dict) -> bool:
        pending = self.dequeue(session_state)
        if not pending:
            return False
        pending_hash = hashlib.sha256(str(pending).encode()).hexdigest()[:8]
        if self._sm is not None:
            state = self._sm.get_state()
            if not self._sm.is_busy():
                return False
            return bool(state.running_hash) and state.running_hash == pending_hash
        if not session_state.get(IN_PROGRESS_KEY, False):
            return False
        return session_state.get(RUNNING_HASH_KEY) == pending_hash

    def should_retry(self, session_state: dict) -> bool:
        if not self.is_recovering(session_state):
            return False

        if self._sm is not None:
            state = self._sm.get_state()
            retry_count = state.retry_count
            running_ts = state.running_ts
        else:
            retry_count = session_state.get(RETRY_COUNT_KEY, 0)
            running_ts = session_state.get(RUNNING_TS_KEY, 0)

        if retry_count >= MAX_RETRIES:
            _log.error("DISPATCH | 重试 %d 次仍失败，放弃", retry_count)
            self._abandon_retry(session_state)
            return False
        elapsed = time.time() - (running_ts or 0)
        if elapsed > MAX_RETRY_ELAPSED:
            _log.error(
                "DISPATCH | 已耗时 %.0fs > %ds 上限，放弃重试",
                elapsed,
                MAX_RETRY_ELAPSED,
            )
            self._abandon_retry(session_state)
            return False
        if elapsed < RETRY_COOLDOWN:
            return False
        new_count = retry_count + 1
        if self._sm is not None:
            self._sm.update(retry_count=new_count)
        self._sync_flat(session_state, retry_count=new_count)
        _log.info("DISPATCH | 重试 %d/%d | elapsed=%.0fs", new_count, MAX_RETRIES, elapsed)
        return True

    def _abandon_retry(self, session_state: dict) -> None:
        if self._sm is not None:
            if self._sm.is_extracting() or self._sm.is_gating():
                self._sm.transition(LeadInPhase.ERROR)
            self._sm.update(retry_count=0, tools_failed=False, pending_text="")
        self._sync_flat(
            session_state,
            in_progress=False,
            retry_count=0,
            clear_pending=True,
            clear_tools_failed=True,
        )

    def mark_running(self, session_state: dict, analyze_text: str) -> None:
        text_hash = hashlib.sha256(analyze_text.encode()).hexdigest()[:8]
        now = time.time()
        self._sync_flat(
            session_state,
            in_progress=True,
            running_hash=text_hash,
            running_ts=now,
            clear_trace=True,
        )

        if self._sm is not None:
            if self._sm.current != LeadInPhase.IDLE:
                self._sm.update(
                    progress_steps=[],
                    progress_text="",
                    progress_variant="default",
                    intent_blocked=False,
                    intent_gate_result=None,
                    last_path="",
                    last_error=None,
                    last_applied_fields={},
                    last_trace=[],
                    tools_failed=False,
                    feedback_dismissed=False,
                )
            if self._sm.current == LeadInPhase.GATING:
                self._sm.transition(LeadInPhase.EXTRACTING)
            elif self._sm.current != LeadInPhase.EXTRACTING:
                self._sm.transition(LeadInPhase.EXTRACTING)
            self._sm.update(
                pending_text=analyze_text,
                running_hash=text_hash,
                running_ts=now,
                last_trace=[],
            )

    def mark_done(self, session_state: dict) -> None:
        if self._sm is not None:
            if self._sm.is_blocked():
                self._sm.transition(LeadInPhase.IDLE)
            elif self._sm.is_awaiting():
                self._sm.transition(LeadInPhase.DONE)
            elif self._sm.is_extracting() or self._sm.is_gating():
                self._sm.transition(LeadInPhase.AWAITING)
                self._sm.transition(LeadInPhase.DONE)
            elif self._sm.is_error():
                self._sm.transition(LeadInPhase.IDLE)
            elif not self._sm.is_idle() and not self._sm.is_done():
                self._sm.transition(LeadInPhase.DONE)
            self._sm.update(
                progress_steps=[],
                progress_text="",
                progress_variant="default",
                pending_text="",
                retry_count=0,
                tools_failed=False,
            )
        self._sync_flat(
            session_state,
            in_progress=False,
            retry_count=0,
            clear_pending=True,
            clear_tools_failed=True,
        )

    @staticmethod
    def _mode() -> str:
        try:
            from src.agent.harness import load_harness_config

            return load_harness_config().stages.get("lead_in", "tools")
        except Exception:
            _log.warning("failed to load harness config, defaulting to tools mode", exc_info=True)
            return "tools"
