from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, MutableMapping, Optional, Protocol


class Tool(Protocol):
    name: str

    def __call__(self, payload: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]: ...


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: Optional[Dict[str, Any]] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: MutableMapping[str, Tool] = {}
        self._specs: MutableMapping[str, ToolSpec] = {}

    def register(self, spec: ToolSpec, impl: Tool) -> None:
        if spec.name in self._tools:
            raise ValueError(f"工具已存在: {spec.name}")
        self._tools[spec.name] = impl
        self._specs[spec.name] = spec

    def get(self, name: str) -> Tool:
        impl = self._tools.get(name)
        if impl is None:
            raise KeyError(f"未找到工具: {name}")
        return impl

    def describe(self, name: str) -> ToolSpec:
        spec = self._specs.get(name)
        if spec is None:
            raise KeyError(f"未找到工具: {name}")
        return spec


tool_registry = ToolRegistry()


__all__ = [
    "Tool",
    "ToolSpec",
    "ToolRegistry",
    "tool_registry",
]
