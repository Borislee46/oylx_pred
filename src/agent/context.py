import logging
import time
from typing import Any

from pydantic import BaseModel, Field, model_validator

from src.agent.schemas import ExtractedBackground

_log = logging.getLogger(__name__)


class StudentContext(BaseModel):
    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def _none_str_to_empty(cls, data: Any) -> Any:
        if isinstance(data, dict):
            for field_name, field_info in cls.model_fields.items():
                if field_info.annotation is str and field_name in data and data[field_name] is None:
                    data[field_name] = ""
        return data

    stage: str = "lead_in"
    created_at: float = Field(default_factory=time.time)

    raw_input: str = ""
    extracted_background: ExtractedBackground = Field(default_factory=ExtractedBackground)
    quick_assessment: str = ""
    suggested_questions: list[str] = Field(default_factory=list)
    conversation_turns: list[dict[str, Any]] = Field(default_factory=list)
    intent_gate_last: dict[str, Any] | None = None
    session_continuity: str = "continue"

    gpa: float = 0.0
    language_score: float = 0.0
    language_score_raw: float = 0.0
    language_type: str = ""
    gpa_raw: float = 0.0
    standardized_test_type: str = ""
    standardized_test_score: float = 0.0
    background_university: str = ""
    background_major: str = ""
    background_major_2: str = ""
    is_dual_degree: bool = False
    target_country: str = ""
    target_universities: list[str] = Field(default_factory=list)
    target_majors: list[str] = Field(default_factory=list)
    experience_details: dict[str, Any] = Field(default_factory=dict)

    prediction_results: dict[str, Any] = Field(default_factory=dict)
    ai_explanation: str = ""
    profile_type: str = ""
    matched_products: list[dict[str, Any]] = Field(default_factory=list)
    portfolio_combo: list[dict[str, Any]] = Field(default_factory=list)
    sales_snapshot: dict[str, Any] = Field(default_factory=dict)
    contract_tier: str = ""

    history: list[dict[str, Any]] = Field(default_factory=list)

    def record_tool_call(
        self,
        tool: str,
        *,
        capability: str = "",
        cost: str = "free",
        blocked: bool = False,
    ) -> None:
        self.history.append(
            {
                "event": "tool_call",
                "tool": tool,
                "capability": capability,
                "cost": cost,
                "blocked": blocked,
                "ts": time.time(),
            }
        )
        _log.debug(
            "记录工具审计: tool=%s capability=%s cost=%s blocked=%s",
            tool,
            capability,
            cost,
            blocked,
        )

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError as err:
            raise KeyError(key) from err

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.model_fields.keys()
