from __future__ import annotations

import atexit
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

_logger = logging.getLogger(__name__)

FLUSH_INTERVAL_SECONDS: float = 30.0
BATCH_SIZE: int = 50
MAX_QUEUE_SIZE: int = 1000
LOG_DIR: Path = Path("logs") / "analytics"


@dataclass
class AnalEvent:
    event_type: str
    session_id: str
    user_email: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    data: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "session_id": self.session_id,
            "user_email": self.user_email,
            "timestamp": self.timestamp,
            "data_json": json.dumps(self.data, ensure_ascii=False, default=str),
        }


_EVENT_QUEUE: list[AnalEvent] = []
_QUEUE_LOCK = threading.Lock()
_FLUSH_LOCK = threading.Lock()
_FLUSH_THREAD: threading.Thread | None = None
_SHUTDOWN_FLAG = threading.Event()
_THREAD_STARTED = False


def _get_session_id() -> str:
    try:
        import streamlit as st

        sid = st.session_state.get("session_id")
        if sid:
            return str(sid)[:8]
    except Exception:
        pass
    return "NO_SESSION"


def _get_user_email() -> str:
    try:
        import streamlit as st

        udm = st.session_state.get("user_data_model")
        if udm:
            info = getattr(udm, "user_info", {}) or {}
            email = info.get("user_email", "")
            if not email:
                email = st.session_state.get("e2_user_email", "")
            if email:
                return _mask_email(str(email))
    except Exception:
        pass
    return "UNKNOWN"


def _mask_email(email: str) -> str:
    if "@" not in email:
        return email[:3] + "***"
    local, domain = email.split("@", 1)
    if len(local) <= 3:
        return f"{local}***@{domain}"
    return f"{local[:3]}***@{domain}"


def _today_file() -> Path:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return LOG_DIR / f"{date.today().isoformat()}.feather"


_DEDUP_SCOPES: dict[str, str] = {
    "session_start": "session",
    "page_view": "session",
    "lead_in_start": "session",
    "lead_in_complete": "session",
    "prediction_start": "prediction",
    "dual_major_used": "prediction",
    "cross_faculty_popup": "prediction",
    "prediction_complete": "prediction",
    "prediction_empty": "prediction",
    "prediction_error": "prediction",
    "explain_requested": "prediction",
    "portfolio_tab_viewed": "prediction",
    "manual_adjust_applied": "prediction",
}


def _has_streamlit_runtime() -> bool:
    try:
        import streamlit as st

        runtime = getattr(st, "runtime", None)
        exists = getattr(runtime, "exists", None)
        return bool(exists and exists())
    except Exception:
        return False


def _should_skip(event_type: str) -> bool:
    scope = _DEDUP_SCOPES.get(event_type)
    if not scope:
        return False
    if not _has_streamlit_runtime():
        return False

    flag_key = f"_analytics_dedup_{event_type}"
    try:
        import streamlit as st

        if scope == "session":
            if st.session_state.get(flag_key):
                return True
            st.session_state[flag_key] = True
        elif scope == "prediction":
            pred_id = st.session_state.get("_analytics_pred_id", "")
            stored = st.session_state.get(flag_key, "")
            if stored == pred_id:
                return True
            st.session_state[flag_key] = pred_id
    except Exception:
        pass
    return False


def _bump_prediction_id() -> None:
    if not _has_streamlit_runtime():
        return
    import uuid

    try:
        import streamlit as st

        st.session_state["_analytics_pred_id"] = str(uuid.uuid4())[:8]
    except Exception:
        pass


def track(event_type: str, **data: Any) -> None:
    if _should_skip(event_type):
        return

    event = AnalEvent(
        event_type=event_type,
        session_id=_get_session_id(),
        user_email=_get_user_email(),
        data=data,
    )

    with _QUEUE_LOCK:
        _EVENT_QUEUE.append(event)
        should_flush = len(_EVENT_QUEUE) >= BATCH_SIZE

    if should_flush:
        _flush()

    global _THREAD_STARTED
    if not _THREAD_STARTED:
        _start_flush_thread()


def _flush() -> None:
    with _QUEUE_LOCK:
        if not _EVENT_QUEUE:
            return
        events = list(_EVENT_QUEUE)
        _EVENT_QUEUE.clear()

    if not events:
        return

    rows = [e.to_row() for e in events]
    new_df = pd.DataFrame(rows)
    target = _today_file()

    with _FLUSH_LOCK:
        try:
            if target.exists():
                try:
                    existing = pd.read_feather(target)
                except Exception:
                    _logger.warning("Corrupted analytics file detected, recreating: %s", target)
                    target.unlink(missing_ok=True)
                    existing = None
                if existing is not None:
                    combined = pd.concat([existing, new_df], ignore_index=True)
                else:
                    combined = new_df
                combined.to_feather(target)
            else:
                new_df.to_feather(target)
        except Exception:
            _logger.warning("Failed to flush analytics events", exc_info=True)
            with _QUEUE_LOCK:
                if len(_EVENT_QUEUE) + len(events) <= MAX_QUEUE_SIZE:
                    _EVENT_QUEUE.extend(events)
                else:
                    _logger.error(
                        "analytics queue 已满（%d），丢弃 %d 条事件",
                        MAX_QUEUE_SIZE,
                        len(events),
                    )


def _flush_loop() -> None:
    while not _SHUTDOWN_FLAG.wait(FLUSH_INTERVAL_SECONDS):
        try:
            _flush()
        except Exception:
            _logger.warning("Background flush failed", exc_info=True)


def _start_flush_thread() -> None:
    global _THREAD_STARTED, _FLUSH_THREAD
    if _THREAD_STARTED:
        return
    _THREAD_STARTED = True
    _FLUSH_THREAD = threading.Thread(target=_flush_loop, daemon=True, name="analytics-flush")
    _FLUSH_THREAD.start()


def _shutdown() -> None:
    _SHUTDOWN_FLAG.set()
    _flush()


atexit.register(_shutdown)
