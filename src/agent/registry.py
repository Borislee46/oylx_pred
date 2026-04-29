from __future__ import annotations

from collections.abc import Callable
from typing import Any


class AgentRegistry:
    """Agent factory registry. Each call to get() creates a fresh instance.

    Agents are NOT cached — every session thread gets its own instance,
    ensuring _memory_cache, _session, and _stream_buffer are isolated.
    """

    _factories: dict[str, Callable[[], Any]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Any]) -> None:
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls._factories:
            return cls._factories[name]()
        raise KeyError(f"Agent '{name}' 未注册。可用: {list(cls._factories.keys())}")

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._factories.keys())

    @classmethod
    def clear(cls) -> None:
        cls._factories.clear()
