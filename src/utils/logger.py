import logging
import logging.handlers
import os

import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx


class SessionIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        ctx = get_script_run_ctx()
        if ctx is None:
            record.session_id = "NO_SESSION"
            record.userid = "THREAD_MODE"
            return True

        try:
            record.session_id = st.session_state.get("session_id", "NO_SESSION")
        except Exception:
            record.session_id = "NO_SESSION"

        try:
            e2_nickname = st.session_state.get("e2_user_nickname")
            e2_email = st.session_state.get("e2_user_email")

            if e2_nickname and e2_email:
                record.userid = f"{e2_nickname} <{e2_email}>"
            elif e2_nickname:
                record.userid = e2_nickname
            elif e2_email:
                record.userid = e2_email
            else:
                record.userid = st.session_state.get("logged_in_user", "E2_USER_NOT_LOGGED_IN")
        except Exception:
            record.userid = "USER_INFO_ERROR"
        return True


class SessionDedupFilter(logging.Filter):
    _MAX_SESSIONS = 64
    _MAX_ENTRIES_PER_SESSION = 5000

    def __init__(self) -> None:
        super().__init__()
        self._seen: dict[str, set[int]] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        session_id: str = getattr(record, "session_id", "NO_SESSION")
        if session_id in ("NO_SESSION", "THREAD_MODE", "USER_INFO_ERROR", ""):
            return True

        msg = record.getMessage()
        key = hash((record.name, msg))

        seen = self._seen.get(session_id)
        if seen is None:
            self._seen[session_id] = {key}
            self._prune()
            return True

        if key in seen:
            return False

        if len(seen) < self._MAX_ENTRIES_PER_SESSION:
            seen.add(key)
        return True

    def _prune(self) -> None:
        if len(self._seen) > self._MAX_SESSIONS:
            excess = len(self._seen) - self._MAX_SESSIONS
            for sid in list(self._seen.keys())[:excess]:
                del self._seen[sid]


def setup_logger(page_name: str, sub_dir: str) -> logging.Logger:
    log_dir = os.path.join("logs", page_name, sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{sub_dir}.log")

    logger_name = f"{page_name}_{sub_dir}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(userid)s - %(session_id)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=100 * 1024 * 1024, backupCount=30, encoding="utf-8"
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.addFilter(SessionIDFilter())
    logger.addFilter(SessionDedupFilter())
    logger.propagate = False

    return logger


_LEAD_IN_LOGGER_NAMES = (
    "lead_in_tool_agent",
    "LeadInDispatcher",
    "lead_in_router",
    "AgentProtocol",
    "src.agent.tools.form_gateway",
    "src.agent.gateways.prediction_gateway",
    "src.agent.runtime.agent_factory",
    "src.agent.runtime.agent_runner",
    "intent_gate",
    "session_continuity",
)

_lead_in_logging_ready = False


def log_once(logger: logging.Logger, key: str, level: int, msg: str, *args, **kwargs) -> None:
    try:
        session_key = f"_log_once_{key}"
        if session_key not in st.session_state:
            st.session_state[session_key] = True
            logger.log(level, msg, *args, **kwargs)
    except Exception:
        logger.log(level, msg, *args, **kwargs)


def ensure_lead_in_console_logging(parent_name: str = "page3_prediction") -> None:
    global _lead_in_logging_ready
    if _lead_in_logging_ready:
        return
    parent = logging.getLogger(parent_name)
    if not parent.handlers:
        setup_logger("page3", "prediction")
        parent = logging.getLogger(parent_name)
    if not parent.handlers:
        return
    for name in _LEAD_IN_LOGGER_NAMES:
        child = logging.getLogger(name)
        if child.handlers:
            continue
        child.setLevel(logging.INFO)
        for handler in parent.handlers:
            child.addHandler(handler)
        for filt in parent.filters:
            child.addFilter(filt)
        child.propagate = False
    _lead_in_logging_ready = True
