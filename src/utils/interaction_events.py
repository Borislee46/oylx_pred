"""Lightweight anonymized event logging for product feedback loops."""

from __future__ import annotations

import json
from typing import Any

from src.utils.logger import setup_logger

_event_logger = setup_logger("page3", "events")


def log_interaction_event(name: str, payload: dict[str, Any]) -> None:
    safe_payload = _json_safe(payload)
    _event_logger.info(
        "EVENT %s | %s",
        name,
        json.dumps(safe_payload, ensure_ascii=False, sort_keys=True),
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
