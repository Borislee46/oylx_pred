from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.utils.session_manager import SessionManager


class LeadInPhase:
    IDLE: str = "idle"
    GATING: str = "gating"
    BLOCKED: str = "blocked"
    EXTRACTING: str = "extracting"
    AWAITING: str = "awaiting"
    PREDICTING: str = "predicting"
    DONE: str = "done"
    ERROR: str = "error"


@dataclass
class LeadInTurnState:
    phase: str = LeadInPhase.IDLE
    pending_text: str = ""
    running_hash: str = ""
    running_ts: float = 0.0
    retry_count: int = 0
    tools_failed: bool = False
    turn: int = 0

    progress_steps: list[str] = field(default_factory=list)
    progress_text: str = ""
    progress_variant: str = "default"
    progress_details: list[str] = field(default_factory=list)

    feedback_dismissed: bool = False
    last_applied_fields: dict[str, Any] = field(default_factory=dict)
    low_confidence_display: dict[str, Any] = field(default_factory=dict)
    clarifying_questions: list[str] = field(default_factory=list)
    last_trace: list[dict[str, Any]] = field(default_factory=list)

    intent_blocked: bool = False
    intent_gate_result: dict[str, Any] | None = None

    pydantic_messages: list[Any] = field(default_factory=list)
    conversation_turns: list[dict[str, Any]] = field(default_factory=list)

    last_path: str = ""
    last_error: str | None = None

    form_expander_open: bool = False

    def reset_for_new_turn(self) -> None:
        self.phase = LeadInPhase.IDLE
        self.pending_text = ""
        self.running_hash = ""
        self.running_ts = 0.0
        self.retry_count = 0
        self.tools_failed = False
        self.progress_steps = []
        self.progress_text = ""
        self.progress_variant = "default"
        self.progress_details = []
        self.intent_blocked = False
        self.intent_gate_result = None
        self.last_path = ""
        self.last_error = None
        self.form_expander_open = False

    def reset_all(self) -> None:
        self.reset_for_new_turn()
        self.turn = 0
        self.pydantic_messages = []
        self.conversation_turns = []
        self.feedback_dismissed = False
        self.last_applied_fields = {}
        self.low_confidence_display = {}
        self.clarifying_questions = []
        self.last_trace = []


class LeadInTurnStateMachine:
    _STATE_KEY = "_lead_in_turn_state"

    _TRANSITIONS: dict[str, set[str]] = {
        LeadInPhase.IDLE: {LeadInPhase.GATING, LeadInPhase.EXTRACTING},
        LeadInPhase.GATING: {LeadInPhase.EXTRACTING, LeadInPhase.BLOCKED, LeadInPhase.ERROR},
        LeadInPhase.BLOCKED: {LeadInPhase.IDLE, LeadInPhase.GATING},
        LeadInPhase.EXTRACTING: {LeadInPhase.AWAITING, LeadInPhase.ERROR, LeadInPhase.GATING},
        LeadInPhase.AWAITING: {
            LeadInPhase.IDLE,
            LeadInPhase.EXTRACTING,
            LeadInPhase.DONE,
            LeadInPhase.PREDICTING,
        },
        LeadInPhase.PREDICTING: {LeadInPhase.DONE, LeadInPhase.ERROR},
        LeadInPhase.DONE: {LeadInPhase.IDLE, LeadInPhase.EXTRACTING},
        LeadInPhase.ERROR: {LeadInPhase.IDLE, LeadInPhase.EXTRACTING, LeadInPhase.GATING},
    }

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager

    def _ensure_state(self) -> LeadInTurnState:
        state = self._sm.get(self._STATE_KEY)
        if state is None:
            state = LeadInTurnState()
            self._sm.set(**{self._STATE_KEY: state})
        return state

    @property
    def current(self) -> str:
        return self._ensure_state().phase

    def transition(self, to: str) -> None:
        if to == self.current:
            return
        allowed = self._TRANSITIONS.get(self.current, set())
        if to not in allowed:
            raise ValueError(f"非法 LeadIn 状态转换: {self.current} → {to}")
        state = self._ensure_state()
        state.phase = to
        self._sm.set(**{self._STATE_KEY: state})

    def is_idle(self) -> bool:
        return self.current == LeadInPhase.IDLE

    def is_gating(self) -> bool:
        return self.current == LeadInPhase.GATING

    def is_blocked(self) -> bool:
        return self.current == LeadInPhase.BLOCKED

    def is_extracting(self) -> bool:
        return self.current == LeadInPhase.EXTRACTING

    def is_awaiting(self) -> bool:
        return self.current == LeadInPhase.AWAITING

    def is_predicting(self) -> bool:
        return self.current == LeadInPhase.PREDICTING

    def is_done(self) -> bool:
        return self.current == LeadInPhase.DONE

    def is_error(self) -> bool:
        return self.current == LeadInPhase.ERROR

    def is_active(self) -> bool:
        return self.current not in (
            LeadInPhase.IDLE,
            LeadInPhase.DONE,
            LeadInPhase.ERROR,
            LeadInPhase.BLOCKED,
        )

    def is_busy(self) -> bool:
        return self.current in (LeadInPhase.GATING, LeadInPhase.EXTRACTING)

    def get_state(self) -> LeadInTurnState:
        return self._ensure_state()

    def update(self, **kwargs: Any) -> None:
        state = self._ensure_state()
        for key, value in kwargs.items():
            if hasattr(state, key):
                setattr(state, key, value)
        self._sm.set(**{self._STATE_KEY: state})
