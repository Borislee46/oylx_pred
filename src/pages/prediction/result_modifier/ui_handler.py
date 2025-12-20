import hashlib
import time

import streamlit as st

from src.pages.prediction.config.ui_messages import RANKER_MESSAGES
from src.pages.prediction.flow.progress_reporter import ProgressReporter


def _has_streamlit_runtime() -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(exists and exists())


class LoadingMessageAnimator:
    def __init__(
        self,
        placeholder=None,
        min_interval: float = 1.2,
        *,
        progress_reporter: ProgressReporter | None = None,
    ):
        self.progress_reporter = progress_reporter
        self.placeholder = (
            placeholder
            if placeholder is not None
            else (st.empty() if _has_streamlit_runtime() and progress_reporter is None else None)
        )
        self.min_interval = min_interval
        self._cycle_count = 0
        self._current_message = ""
        self._last_update_time = 0.0

    def _render(self, message: str):
        if self.progress_reporter is not None:
            self.progress_reporter.emit(message, force=True)
        elif self.placeholder is not None:
            dots = [".", "..", "..."][self._cycle_count % 3]
            self.placeholder.markdown(
                f'<div style="color:#888;font-size:0.85em;margin-top:-15px;margin-bottom:0;line-height:1.2;">{message}{dots}</div>',
                unsafe_allow_html=True,
            )
        self._cycle_count += 1
        self._current_message = message
        self._last_update_time = time.time()

    def show(self, message: str, force: bool = False):
        now = time.time()
        if (
            not force
            and message == self._current_message
            and now - self._last_update_time < self.min_interval
        ):
            return
        self._render(message)

    def tick(self):
        if not self._current_message:
            return
        now = time.time()
        if now - self._last_update_time < self.min_interval:
            return
        self._render(self._current_message)

    def clear(self):
        if self.placeholder is not None:
            self.placeholder.empty()
        self._current_message = ""


class RankerUIHandler:
    BASIC_MESSAGES = RANKER_MESSAGES["basic"]
    WITH_BG_MAJOR_MESSAGES = RANKER_MESSAGES["cross_major"]
    WITH_FACULTY_MESSAGES = RANKER_MESSAGES["faculty"]
    RELAX_MODE_MESSAGES = RANKER_MESSAGES["relax"]
    TIGHTEN_MODE_MESSAGES = RANKER_MESSAGES["tighten"]
    FALLBACK_MESSAGES = RANKER_MESSAGES["fallback"]

    MIN_DISPLAY_INTERVAL = 1.2

    def __init__(
        self,
        background_major: str = "",
        background_faculty: str | None = None,
        mode: str = "relax",
        progress_reporter: ProgressReporter | None = None,
    ):
        self.background_major = background_major
        self.background_faculty = background_faculty
        self.mode = mode
        self.progress_reporter = progress_reporter
        self.placeholder = (
            None
            if progress_reporter is not None
            else (st.empty() if _has_streamlit_runtime() else None)
        )
        self._animator = LoadingMessageAnimator(
            self.placeholder,
            self.MIN_DISPLAY_INTERVAL,
            progress_reporter=progress_reporter,
        )
        self.is_active = False
        self._message_history: set = set()
        self._round_count = 0
        self._last_update_time = 0.0
        self._message_pools = self._build_message_pools()
        self._tone = "探索模式" if self.mode == "relax" else "精准模式"
        self._tone_seed = self._build_tone_seed()

    def _build_tone_seed(self) -> str:
        bg_major = (self.background_major or "").strip()
        faculty = (self.background_faculty or "").strip()
        mode = (self.mode or "").strip()
        return f"{mode}|{bg_major}|{faculty}"

    def _stable_index(self, key: str, n: int) -> int:
        if n <= 1:
            return 0
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % n

    def _build_message_pools(self) -> list[list[str]]:
        pools = [self.BASIC_MESSAGES]
        if self.background_major:
            pools.append(self.WITH_BG_MAJOR_MESSAGES)
        if self.background_faculty:
            pools.append(self.WITH_FACULTY_MESSAGES)
        if self.mode == "relax":
            pools.append(self.RELAX_MODE_MESSAGES)
        else:
            pools.append(self.TIGHTEN_MODE_MESSAGES)
        return pools

    def __enter__(self):
        bg_text = self.background_major.strip() if self.background_major else "未提供"
        faculty_text = self.background_faculty.strip() if self.background_faculty else "未提供"
        self._render(f"{self._tone}｜正在筛选专业推荐｜本科 {bg_text}｜领域 {faculty_text}")
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False
        self._animator.clear()
        return False

    def _render(self, message: str):
        self._current_message = message
        self._animator.show(message, force=True)

    def update_message(self, message: str):
        self._render(message)

    def update_loop(self):
        if self.is_active and hasattr(self, "_current_message"):
            self._animator.tick()

    def _pick_fresh_message(self, pool: list[str], **kwargs) -> str:
        available = [m for m in pool if m not in self._message_history]
        if not available:
            self._message_history.clear()
            available = pool
        majors = str(kwargs.get("majors", ""))
        seed = f"{self._tone_seed}|{self._round_count}|{majors}|{len(available)}"
        msg = available[self._stable_index(seed, len(available))]
        self._message_history.add(msg)
        return msg.format(**kwargs) if kwargs else msg

    def show_candidates(self, major_names: list[str]):
        now = time.time()
        if now - self._last_update_time < self.MIN_DISPLAY_INTERVAL:
            return

        self._round_count += 1
        self._last_update_time = now

        if major_names:
            text = major_names[
                self._stable_index(f"{self._tone_seed}|m|{self._round_count}", len(major_names))
            ]
            pool = self._message_pools[self._round_count % len(self._message_pools)]
            msg = self._pick_fresh_message(
                pool,
                majors=text,
                bg_major=self.background_major or "当前背景",
                faculty=self.background_faculty or "目标领域",
                tone=self._tone,
            )
            self._render(msg)
            if self.progress_reporter is not None:
                self.progress_reporter.advance_ratio(0.04, text=msg)
        else:
            msg = self._pick_fresh_message(self.FALLBACK_MESSAGES, tone=self._tone)
            self._render(msg)
            if self.progress_reporter is not None:
                self.progress_reporter.advance_ratio(0.02, text=msg)
