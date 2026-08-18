from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from src.agent.persistent_cache import PersistentCache
from src.agent.runtime.model_factory import build_model_with_fallback


class BaseAgent:
    _cache: PersistentCache | None = None

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        timeout: int = 10,
        agent_name: str = "Agent",
        logger: logging.Logger | None = None,
    ) -> None:
        _ = config
        self._model = build_model_with_fallback(timeout=timeout)
        self.model = self._model.model_name if hasattr(self._model, "model_name") else "?"
        self.agent_name = agent_name
        self.logger = logger or logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _hash_cache_key(version: int, model: str, **fields: str) -> str:
        key_data: dict[str, str] = {"v": str(version), "model": model}
        for k, v in sorted(fields.items()):
            key_data[k] = str(v or "").strip()
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()
