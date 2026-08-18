import logging
import time as _time_module
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from src.utils.config_models import HarnessConfig
from src.utils.env_config_loader import load_app_config

_log = logging.getLogger(__name__)


def load_harness_config() -> HarnessConfig:
    cfg = load_app_config()
    return HarnessConfig.from_app_config(cfg)


@dataclass
class TraceEntry:
    seq: int
    tool: str
    args_preview: dict[str, Any] = field(default_factory=dict)
    result_preview: str = ""
    ts: float = 0.0
    ok: bool = True


@dataclass
class ToolTurnTrace:
    entries: list[TraceEntry] = field(default_factory=list)
    _seq: int = field(default=0, init=False)

    def record(
        self,
        tool: str,
        *,
        args_preview: dict[str, Any] | None = None,
        result_preview: str = "",
        ok: bool = True,
    ) -> TraceEntry:
        entry = TraceEntry(
            seq=self._seq,
            tool=tool,
            args_preview=args_preview or {},
            result_preview=result_preview,
            ts=_time_module.time(),
            ok=ok,
        )
        self._seq += 1
        self.entries.append(entry)
        return entry


@dataclass
class HarnessDeps:
    gateway: Any
    ctx: Any | None = None
    prediction: Any | None = None
    explain: Any | None = None
    memory: Any = None
    turn: int = 1
    trace: ToolTurnTrace = field(default_factory=ToolTurnTrace)


def build_harness_deps(
    session_manager: Any,
    ctx: Any,
    *,
    turn: int = 1,
) -> HarnessDeps:
    from src.agent.tools.form_gateway import StreamlitFormGateway

    form_gateway = StreamlitFormGateway(session_manager, ctx)

    return HarnessDeps(
        gateway=form_gateway,
        ctx=ctx,
        prediction=None,
        explain=None,
        turn=turn,
    )


@dataclass
class HarnessTurnResult:
    handled: bool = False
    path: str = ""
    feedback: str = ""
    applied_fields: dict[str, Any] = field(default_factory=dict)
    low_confidence_fields: dict[str, Any] | None = None
    should_expand_form: bool = False
    should_auto_predict: bool = False
    clarifying_questions: list[str] = field(default_factory=list)
    prediction_triggered: bool = False
    explain_result: dict | None = None
    error: str | None = None
    validation_issues: list[str] = field(default_factory=list)


class ConsultationHarness:
    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or load_harness_config()

    def run_lead_in_turn(
        self,
        session_manager: Any,
        ctx: Any,
        user_input: str,
        *,
        on_progress: Callable | None = None,
    ) -> HarnessTurnResult:
        import streamlit as st

        from src.agent.lead_in.dispatcher import LeadInDispatcher
        from src.agent.lead_in.state_machine import LeadInTurnStateMachine

        sm = LeadInTurnStateMachine(session_manager)
        dispatcher = LeadInDispatcher(on_progress=on_progress, state_machine=sm)
        dispatch_result = dispatcher.dispatch(
            session_manager, ctx, user_input, session_state=st.session_state
        )

        return HarnessTurnResult(
            handled=dispatch_result.handled,
            path=dispatch_result.path,
            feedback=dispatch_result.feedback,
            applied_fields=dispatch_result.applied_fields,
            low_confidence_fields=dispatch_result.low_confidence_fields,
            should_auto_predict=dispatch_result.should_auto_predict,
            should_expand_form=dispatch_result.should_expand_form,
            clarifying_questions=dispatch_result.clarifying_questions,
            validation_issues=dispatch_result.validation_issues,
            error=dispatch_result.error,
        )
