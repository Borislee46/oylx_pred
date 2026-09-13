from dataclasses import dataclass, field
from typing import Any


@dataclass
class DispatchResult:
    handled: bool = False
    path: str = ""
    feedback: str = ""
    applied_fields: dict[str, Any] = field(default_factory=dict)
    low_confidence_fields: dict[str, Any] | None = None
    should_expand_form: bool = False
    should_auto_predict: bool = False
    clarifying_questions: list[str] = field(default_factory=list)
    validation_issues: list[str] = field(default_factory=list)
    error: str | None = None
    trace_entries: list[dict[str, Any]] = field(default_factory=list)
