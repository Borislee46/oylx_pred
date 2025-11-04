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
    logger.propagate = False

    return logger
