from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.session_manager import SessionManager


class HKPagePhase:
    IDLE: str = "idle"
    RUNNING: str = "running"
    AWAITING_CONFIRM: str = "awaiting_confirm"
    DONE: str = "done"
    ERROR: str = "error"


class PageStateMachine:
    _TRANSITIONS: dict[str, set[str]] = {
        HKPagePhase.IDLE: {HKPagePhase.RUNNING},
        HKPagePhase.RUNNING: {HKPagePhase.DONE, HKPagePhase.AWAITING_CONFIRM, HKPagePhase.ERROR},
        HKPagePhase.AWAITING_CONFIRM: {HKPagePhase.RUNNING, HKPagePhase.IDLE, HKPagePhase.ERROR},
        HKPagePhase.DONE: {HKPagePhase.IDLE, HKPagePhase.RUNNING},
        HKPagePhase.ERROR: {HKPagePhase.IDLE, HKPagePhase.RUNNING},
    }

    def __init__(self, session_manager: SessionManager) -> None:
        self._sm = session_manager
        self._key = "hk_ui_phase"

    @property
    def current(self) -> str:
        return self._sm.get(self._key, HKPagePhase.IDLE)

    def transition(self, to: str) -> None:
        if to == self.current:
            return
        if to not in self._TRANSITIONS.get(self.current, set()):
            raise ValueError(f"非法状态转换: {self.current} → {to}")
        self._sm.set(**{self._key: to})

    def is_idle(self) -> bool:
        return self.current == HKPagePhase.IDLE

    def is_running(self) -> bool:
        return self.current == HKPagePhase.RUNNING

    def is_awaiting_confirm(self) -> bool:
        return self.current == HKPagePhase.AWAITING_CONFIRM

    def is_done(self) -> bool:
        return self.current == HKPagePhase.DONE

    def is_error(self) -> bool:
        return self.current == HKPagePhase.ERROR
