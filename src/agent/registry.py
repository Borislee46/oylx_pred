from __future__ import annotations

from typing import Any


class AgentRegistry:
    _agents: dict[str, Any] = {}

    @classmethod
    def register(cls, name: str, agent: Any) -> None:
        cls._agents[name] = agent

    @classmethod
    def get(cls, name: str) -> Any:
        if name not in cls._agents:
            raise KeyError(f"Agent '{name}' 未注册。可用: {list(cls._agents.keys())}")
        return cls._agents[name]

    @classmethod
    def list(cls) -> list[str]:
        return list(cls._agents.keys())

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
