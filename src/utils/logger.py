import logging
import logging.handlers
import os
import time
from logging import FileHandler

import streamlit as st


class SizeAndTimeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self,
        filename: str,
        maxBytes: int = 0,
        backupCount: int = 0,
        when: str = "midnight",
        encoding: str | None = None,
    ) -> None:
        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding)
        self.when = when
        self.last_rollover_time = int(time.time())

        if when == "midnight":
            self.rollover_interval = 24 * 60 * 60
        elif when.startswith("H"):
            self.rollover_interval = 60 * 60
        else:
            self.rollover_interval = 24 * 60 * 60

    def shouldRollover(self, record: logging.LogRecord) -> int:
        if self.maxBytes > 0:
            try:
                msg = "%s\n" % self.format(record)
                if self.stream is None:
                    self.stream = self._open()
                self.stream.seek(0, 2)
                if self.stream.tell() + len(msg) >= self.maxBytes:
                    return 1
            except Exception:
                return 0

        current_time = int(time.time())
        if current_time - self.last_rollover_time >= self.rollover_interval:
            return 1

        return 0

    def doRollover(self) -> None:
        super().doRollover()
        self.last_rollover_time = int(time.time())


class SessionIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
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


def setup_logger(
    page_name: str,
    sub_dir: str,
    rotation_type: str = "size",
    max_bytes: int = 100 * 1024 * 1024,
    backup_count: int = 30,
    when: str = "midnight",
) -> logging.Logger:
    log_dir = os.path.join("logs", page_name, sub_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = os.path.join(log_dir, f"{sub_dir}.log")

    logger_name = f"{page_name}_{sub_dir}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        has_filter = any(isinstance(f, SessionIDFilter) for f in logger.filters)
        if not has_filter:
            logger.addFilter(SessionIDFilter())
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(userid)s - %(session_id)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler: FileHandler
    if rotation_type == "time":
        file_handler = logging.handlers.TimedRotatingFileHandler(
            log_file, when=when, backupCount=backup_count, encoding="utf-8"
        )
    elif rotation_type == "both":
        file_handler = SizeAndTimeRotatingFileHandler(
            log_file, maxBytes=max_bytes, when=when, backupCount=backup_count, encoding="utf-8"
        )
    else:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
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
