from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol


@dataclass(frozen=True)
class SystemPrompt:
    content: str


@dataclass(frozen=True)
class UserPrompt:
    content: str


@dataclass(frozen=True)
class AssistantPrompt:
    content: str


@dataclass(frozen=True)
class PromptContext:
    system: Optional[SystemPrompt]
    few_shots: List[AssistantPrompt]
    metadata: Dict[str, Any]


class PromptBuilder(Protocol):
    def build_messages(
        self, user_query: str, extra_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]: ...


class DefaultPromptBuilder:
    def __init__(self, context: Optional[PromptContext] = None) -> None:
        self._context = context or PromptContext(system=None, few_shots=[], metadata={})

    def build_messages(
        self, user_query: str, extra_context: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        messages: List[Dict[str, Any]] = []
        if self._context.system is not None:
            messages.append({"role": "system", "content": self._context.system.content})
        for shot in self._context.few_shots:
            messages.append({"role": "assistant", "content": shot.content})
        messages.append({"role": "user", "content": user_query})
        return messages


__all__ = [
    "SystemPrompt",
    "UserPrompt",
    "AssistantPrompt",
    "PromptContext",
    "PromptBuilder",
    "DefaultPromptBuilder",
]
