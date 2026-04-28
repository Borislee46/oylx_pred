from __future__ import annotations

<<<<<<< HEAD
from typing import Any, Callable
=======
from typing import Any
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc


class AgentRegistry:
    _agents: dict[str, Any] = {}
<<<<<<< HEAD
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
=======

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
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc

    @classmethod
    def clear(cls) -> None:
        cls._agents.clear()
<<<<<<< HEAD
        cls._factories.clear()
=======
>>>>>>> 8cd3b6eb5ec7ef4a084c3f4716d9429701e2f0fc
