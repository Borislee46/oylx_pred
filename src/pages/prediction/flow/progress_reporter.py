from __future__ import annotations

import time
from collections.abc import Callable

ProgressCallback = Callable[[str], None]


class ProgressReporter:
    def __init__(
        self,
        progress_cb: ProgressCallback | None,
        *,
        min_interval: float = 0.5,
    ) -> None:
        self._progress_cb = progress_cb
        self._min_interval = float(min_interval)
        self._last_emit_at = 0.0
        self._last_text = ""

    def emit(
        self,
        text: str,
        *,
        force: bool = False,
    ) -> None:
        if self._progress_cb is None:
            return

        now = time.time()
        t = str(text or "").strip()

        if not force:
            if "[阶段]" not in t:
                if (now - self._last_emit_at) < self._min_interval:
                    return

        self._progress_cb(t)
        self._last_emit_at = now
        self._last_text = t
