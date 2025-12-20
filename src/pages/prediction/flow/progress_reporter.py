from __future__ import annotations

import time
from collections.abc import Callable

ProgressCallback = Callable[[float, str], None]


class ProgressReporter:
    def __init__(
        self,
        progress_cb: ProgressCallback | None,
        *,
        min_interval: float = 1.2,
    ) -> None:
        self._progress_cb = progress_cb
        self._min_interval = float(min_interval)
        self._stage_start = 0.0
        self._stage_end = 1.0
        self._p = 0.0
        self._last_emit_at = 0.0
        self._last_text = ""

    def set_stage(self, start: float, end: float, text: str | None = None) -> None:
        s = float(start)
        e = float(end)
        if e < s:
            s, e = e, s
        self._stage_start = max(0.0, min(1.0, s))
        self._stage_end = max(0.0, min(1.0, e))
        self._p = max(self._p, self._stage_start)
        self._p = min(self._p, self._stage_end)
        if text:
            self.emit(text, force=True)
        else:
            self._emit_if_possible(self._p, self._last_text, force=True)

    @property
    def progress(self) -> float:
        return self._p

    @property
    def last_text(self) -> str:
        return self._last_text

    @property
    def stage_end(self) -> float:
        return self._stage_end

    def force_progress(self, p: float, text: str = "") -> None:
        val = float(p)
        val = max(0.0, min(1.0, val))
        self._p = max(self._p, val)
        self._emit_if_possible(self._p, text or self._last_text, force=True)

    def emit(
        self,
        text: str,
        *,
        advance: float | None = None,
        force: bool = False,
    ) -> None:
        if advance is not None:
            self.advance(advance, force=False, text=text)
            return
        self._emit_if_possible(self._p, str(text), force=force)

    def advance(
        self,
        delta: float,
        *,
        force: bool = False,
        text: str | None = None,
    ) -> None:
        d = float(delta)
        if d < 0:
            d = 0.0
        new_p = self._p + d
        new_p = min(new_p, self._stage_end)
        self._p = max(self._p, new_p)
        self._emit_if_possible(
            self._p, str(text) if text is not None else self._last_text, force=force
        )

    def advance_ratio(
        self,
        ratio: float,
        *,
        force: bool = False,
        text: str | None = None,
    ) -> None:
        span = max(0.0, self._stage_end - self._stage_start)
        self.advance(span * float(ratio), force=force, text=text)

    def _emit_if_possible(self, p: float, text: str, *, force: bool) -> None:
        if self._progress_cb is None:
            return
        now = time.time()
        t = str(text or "").strip()
        if not force:
            if t == self._last_text and (now - self._last_emit_at) < self._min_interval:
                return
            if (now - self._last_emit_at) < self._min_interval and t:
                return
        try:
            self._progress_cb(float(p), t or self._last_text)
            self._last_emit_at = now
            if t:
                self._last_text = t
        except Exception:
            return
