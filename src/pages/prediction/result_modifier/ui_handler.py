import random
import time

import streamlit as st


def _has_streamlit_runtime() -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(exists and exists())


class LoadingMessageAnimator:
    def __init__(self, placeholder=None, min_interval: float = 1.2):
        self.placeholder = (
            placeholder
            if placeholder is not None
            else (st.empty() if _has_streamlit_runtime() else None)
        )
        self.min_interval = min_interval
        self._cycle_count = 0
        self._current_message = ""
        self._last_update_time = 0.0

    def _render(self, message: str):
        if self.placeholder is None:
            return
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
        if self._current_message:
            self._render(self._current_message)

    def clear(self):
        if self.placeholder is not None:
            self.placeholder.empty()
        self._current_message = ""


class RankerUIHandler:
    BASIC_MESSAGES = [
        "分析 {majors} 是否为交叉学科",
        "发现潜在匹配：{majors}",
        "评估专业 {majors} 跨领域相关性",
        "分析专业适配度：{majors}",
        "匹配候选专业：{majors}",
        "{majors} 是否属于交叉学科",
        "分析 {majors} 与背景专业关联度",
        "评估 {majors} 跨学科潜力",
        "检验 {majors} 学科边界归属",
    ]

    WITH_BG_MAJOR_MESSAGES = [
        "分析 {majors} 与 {bg_major} 关联",
        "{bg_major} 背景申请 {majors}",
        "从 {bg_major} 到 {majors} 是否算作跨申",
        "{majors} 对 {bg_major} 背景接受度",
    ]

    WITH_FACULTY_MESSAGES = [
        "{faculty} 是否包含 {majors}",
        "从 {faculty} 背景出发，评估 {majors}",
        "在 {faculty} 背景范围内，{majors} 是否符合 {faculty} 偏好",
    ]

    RELAX_MODE_MESSAGES = [
        "发掘跨学科机会：{majors}",
        "扩展推荐边界：{majors}",
        "挖掘跨领域选项：{majors}",
    ]

    TIGHTEN_MODE_MESSAGES = [
        "精准筛选：{majors}",
        "严格评估专业 {majors} 的匹配度",
        "把更多合适的专业如 {majors} 纳入推荐范围",
    ]

    FALLBACK_MESSAGES = [
        "评估潜在推荐专业",
        "分析专业匹配程度",
        "智能筛选候选专业",
        "扩大探索范围",
    ]

    MIN_DISPLAY_INTERVAL = 1.2

    def __init__(
        self,
        background_major: str = "",
        background_faculty: str | None = None,
        mode: str = "relax",
    ):
        self.background_major = background_major
        self.background_faculty = background_faculty
        self.mode = mode
        self.placeholder = st.empty() if _has_streamlit_runtime() else None
        self._animator = LoadingMessageAnimator(self.placeholder, self.MIN_DISPLAY_INTERVAL)
        self.is_active = False
        self._message_history: set = set()
        self._round_count = 0
        self._last_update_time = 0.0
        self._message_pools = self._build_message_pools()

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
        self._render("智能筛选合适的专业推荐")
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
        msg = random.choice(available)
        self._message_history.add(msg)
        return msg.format(**kwargs) if kwargs else msg

    def show_candidates(self, major_names: list[str]):
        now = time.time()
        if now - self._last_update_time < self.MIN_DISPLAY_INTERVAL:
            return

        self._round_count += 1
        self._last_update_time = now

        if major_names:
            text = random.choice(major_names)

            pool = random.choice(self._message_pools)
            msg = self._pick_fresh_message(
                pool,
                majors=text,
                bg_major=self.background_major or "当前背景",
                faculty=self.background_faculty or "目标领域",
            )
            self._render(msg)
        else:
            msg = self._pick_fresh_message(self.FALLBACK_MESSAGES)
            self._render(msg)
