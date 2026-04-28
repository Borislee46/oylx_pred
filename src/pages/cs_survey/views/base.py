from __future__ import annotations

from collections.abc import Callable

from ..schema import SurveyConfig

ViewFn = Callable[[SurveyConfig], None]

VIEW_REGISTRY: dict[str, ViewFn] = {}


def register_view(view_type: str, fn: ViewFn) -> None:
    VIEW_REGISTRY[view_type] = fn
