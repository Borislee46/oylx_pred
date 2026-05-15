"""
进度报告器 — 带节流 + 去重的进度回调封装。

为什么需要节流？
  预测 pipeline 内部多个子步骤都在高频输出进度文本。
  如果每一条都直接调用 progress_cb → Streamlit rerun → 页面闪烁 + 性能下降。
  ProgressReporter 限制最小间隔 0.5s，减少不必要的 rerun。

随机变体机制：
  当 text 是 list[str] 时，随机选择一条未重复的，避免连续两条相同消息。
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

ProgressCallback = Callable[[str], None]


class ProgressReporter:
    """带节流的进度回调封装。

    特性：
    - min_interval 节流：两次 emit 之间至少间隔 min_interval 秒
    - 随机变体：text 为 list 时随机选择（避免连续相同消息）
    - 去重：不连续输出相同文本
    - force=True 可跳过节流（用于关键节点如 phase 切换）
    """

    def __init__(
        self,
        progress_cb: ProgressCallback | None,
        *,
        min_interval: float = 0.5,
    ) -> None:
        self._progress_cb = progress_cb
        self._min_interval = float(min_interval)
        self._last_emit_at = 0.0  # 上次输出时间戳
        self._last_text = ""       # 上次输出文本（去重用）
        self.current_phase: str = ""  # 当前阶段标识

    def emit(
        self,
        text: str | list[str],
        *,
        force: bool = False,
        phase: str = "",
    ) -> None:
        """发送进度消息（受节流和去重约束）。

        Args:
            text: 进度文本，或文本列表（随机选择一条）
            force: True 跳过节流（用于阶段切换等关键节点）
            phase: 设置当前阶段标识（用于 UI 展示当前阶段名）
        """
        if phase:
            self.current_phase = phase

        if self._progress_cb is None:
            return

        now = time.time()

        # 随机变体选择（优先选与上次不同的）
        if isinstance(text, list):
            available_choices = [t for t in text if str(t).strip() != self._last_text]
            if not available_choices:
                available_choices = text
            t = str(random.choice(available_choices) if available_choices else "").strip()
        else:
            t = str(text or "").strip()

        # 节流（force=True 时跳过）
        if not force:
            if (now - self._last_emit_at) < self._min_interval:
                return

        self._progress_cb(t)
        self._last_emit_at = now
        self._last_text = t
