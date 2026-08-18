import json
import logging
import os
import threading
from typing import Any

_log = logging.getLogger(__name__)


class PersistentCache:
    def __init__(self, cache_file: str, cache_dir: str = "cache/agent_cache"):
        self._cache_file = cache_file
        self._cache_dir = cache_dir
        self._lock = threading.RLock()
        self._data: dict[str, Any] | None = None
        self._loaded = False

    @property
    def _file_path(self) -> str:
        return os.path.join(self._cache_dir, self._cache_file)

    def get(self, key: str) -> Any:
        data = self._ensure_loaded()
        value = data.get(key)
        if value is not None:
            _log.debug("CACHE HIT  | %s | key=%s", self._cache_file, key)
        else:
            _log.debug("CACHE MISS | %s | key=%s", self._cache_file, key)
        return value

    def set(self, key: str, value: Any, *, persist: bool = True) -> None:
        with self._lock:
            _log.debug("CACHE SET  | %s | key=%s persist=%s", self._cache_file, key, persist)
            cache = self._ensure_loaded()
            cache[key] = value
            if persist:
                self._save_to_disk(self._data)

    def set_many(self, items: dict[str, Any]) -> None:
        with self._lock:
            cache = self._ensure_loaded()
            count = 0
            for key, value in items.items():
                cache[key] = value
                count += 1
            _log.debug(
                "CACHE SET_MANY | %s | %d keys written",
                self._cache_file,
                count,
            )
            self._save_to_disk(self._data)

    def flush(self) -> None:
        with self._lock:
            if self._data is not None:
                _log.debug("CACHE FLUSH | %s | %d entries", self._cache_file, len(self._data))
                self._save_to_disk(self._data)

    def _ensure_loaded(self) -> dict[str, Any]:
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._data = self._load_from_disk()
                    self._loaded = True
        return self._data if self._data is not None else {}

    def _load_from_disk(self) -> dict[str, Any]:
        if not os.path.exists(self._file_path):
            _log.debug("CACHE INIT | %s | file not found, starting empty", self._cache_file)
            return {}
        try:
            with open(self._file_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                _log.warning(
                    "CACHE CORRUPT | %s | expected dict, got %s — resetting",
                    self._cache_file,
                    type(data).__name__,
                )
                return {}
            _log.debug("CACHE LOAD | %s | %d entries", self._cache_file, len(data))
            return data
        except (OSError, json.JSONDecodeError) as exc:
            _log.warning(
                "CACHE CORRUPT | %s | read/parse failed: %s — resetting",
                self._cache_file,
                exc,
            )
            return {}

    def _save_to_disk(self, data: dict[str, Any]) -> None:
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            tmp_path = f"{self._file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self._file_path)
        except OSError as exc:
            _log.error("CACHE WRITE FAIL | %s | %s", self._cache_file, exc)
