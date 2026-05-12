import random

import streamlit as st

from src.pages.prediction.config.ui_messages import RANKER_MESSAGES
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.result_modifier.utils import has_streamlit_runtime


class LoadingMessageAnimator:
    def __init__(
        self,
        placeholder=None,
        *,
        progress_reporter: ProgressReporter | None = None,
    ):
        self.progress_reporter = progress_reporter
        self.placeholder = (
            placeholder
            if placeholder is not None
            else (st.empty() if has_streamlit_runtime() and progress_reporter is None else None)
        )
        self._current_message = ""

    def show(self, message: str, force: bool = False):
        self._current_message = message
        self._render()

    def _render(self):
        if not self._current_message:
            return

        if self.progress_reporter is not None:
            self.progress_reporter.emit(self._current_message, force=True)
        elif self.placeholder is not None:
            self.placeholder.markdown(
                f'<div style="color:#888;font-size:0.85em;margin-top:-15px;margin-bottom:0;line-height:1.2;">{self._current_message}</div>',
                unsafe_allow_html=True,
            )

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

    MIN_DISPLAY_INTERVAL = 1.0

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
            else (st.empty() if has_streamlit_runtime() else None)
        )
        self._animator = LoadingMessageAnimator(
            self.placeholder,
            progress_reporter=progress_reporter,
        )
        self.is_active = False
        self._round_count = 0
        self._message_pools = self._build_message_pools()
        self._tone = "探索模式" if self.mode == "relax" else "精准模式"

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
        self._render(f"{self._tone}, 正在筛选专业推荐, 本科 {bg_text}, 所属学院 {faculty_text}")
        self.is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.is_active = False
        self._animator.clear()
        return False

    def _render(self, message: str):
        self._animator.show(message, force=True)

    def update_message(self, message: str):
        self._render(message)

    def _pick_message(self, pool: list[str], **kwargs) -> str:
        msg = random.choice(pool)
        return msg.format(**kwargs) if kwargs else msg

    def show_candidates(self, major_names: list[str]):
        self._round_count += 1

        if major_names:
            sample = major_names[0] if len(major_names) == 1 else f"{len(major_names)}个专业"
            pool = self._message_pools[self._round_count % len(self._message_pools)]
            msg = self._pick_message(
                pool,
                target_major=sample,
                background_major_ori=self.background_major or "当前背景",
                tone=self._tone,
                faculty=self.background_faculty or "所属学院",
            )
            self._render(msg)
        else:
            msg = self._pick_message(
                self.FALLBACK_MESSAGES,
                tone=self._tone,
                target_major="目标专业",
                background_major_ori=self.background_major or "当前背景",
            )
            self._render(msg)
