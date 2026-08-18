from __future__ import annotations

import threading
import time

_session_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_global_fallback_lock = threading.Lock()
_lock_last_access: dict[str, float] = {}
_LOCK_TTL = 300


def get_session_lock(session_id: str | None) -> threading.Lock:
    if not session_id:
        return _global_fallback_lock

    with _locks_guard:
        now = time.monotonic()
        stale = [sid for sid, ts in _lock_last_access.items() if now - ts > _LOCK_TTL]
        for sid in stale:
            _session_locks.pop(sid, None)
            _lock_last_access.pop(sid, None)

        if session_id not in _session_locks:
            _session_locks[session_id] = threading.Lock()
        _lock_last_access[session_id] = now
        return _session_locks[session_id]
