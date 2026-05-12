"""Thread-safe persistent JSON cache shared across agent instances.

Used by BoundaryCaseAgent and BackgroundFacultyAgent to avoid
duplicating the same double-check locking + file I/O logic.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any


class PersistentCache:
    """Key-value store backed by a JSON file, safe for concurrent access."""

    def __init__(self, cache_file: str, cache_dir: str = "cache/agent_cache"):
        self._cache_file = cache_file
        self._cache_dir = cache_dir
        self._lock = threading.Lock()
        self._data: dict[str, Any] | None = None
        self._loaded = False

    @property
    def _file_path(self) -> str:
        return os.path.join(self._cache_dir, self._cache_file)

    def _ensure_loaded(self) -> dict[str, Any]:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._data = self._load_from_disk()
                    self._loaded = True
        return self._data or {}

    def _load_from_disk(self) -> dict[str, Any]:
        if not os.path.exists(self._file_path):
            return {}
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _save_to_disk(self, data: dict[str, Any]) -> None:
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            tmp_path = f"{self._file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError:
            pass

    def get(self, key: str) -> Any:
        return self._ensure_loaded().get(key)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            cache = self._ensure_loaded()
            cache[key] = value

    def flush(self) -> None:
        with self._lock:
            if self._data is not None:
                self._save_to_disk(self._data)
