from __future__ import annotations

from typing import Any, Callable


class AgentRegistry:
    _agents: dict[str, Any] = {}
    _factories: dict[str, Callable[[], Any]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[[], Any]) -> None:
        cls._factories[name] = factory

    @classmethod
    def get(cls, name: str) -> Any:
        if name in cls._agents:
            return cls._agents[name]
        if name in cls._factories:
            instance = cls._factories[name]()
            cls._agents[name] = instance
            return instance
        raise KeyError(f"Agent '{name}' 未注册。可用: {list(cls._factories.keys())}")

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._factories.keys())

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
        cls._factories.clear()
