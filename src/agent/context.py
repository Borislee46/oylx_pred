from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.agent.schemas import ExtractedBackground


@dataclass
class StudentContext:
    """贯穿全链路的共享上下文，从前期的碎片信息逐步填充到后期的申请计划。"""

    stage: str = "lead_in"  # lead_in | match | application
    created_at: float = field(default_factory=time.time)

    # ---------- 前期：LeadInAgent 填充 ----------
    raw_input: str = ""

    extracted_background: ExtractedBackground = field(default_factory=dict)
    # { university, major, gpa, language_score, language_type, country,
    #   target_schools, target_majors, research, internship, award, paper }

    quick_assessment: str = ""
    suggested_questions: list[str] = field(default_factory=list)

    conversation_turns: list[dict[str, Any]] = field(default_factory=list)
    # [{role: "user"|"agent", content: str, ts: float}, ...]

    # ---------- 中期：MatchAnalyzer 填充 ----------
    match_analysis: dict[str, Any] = field(default_factory=dict)

    # ---------- 中期：MatchAdvisor 填充 ----------
    gpa: float = 0.0
    language_score: float = 0.0
    language_score_raw: float = 0.0
    language_type: str = ""
    gpa_raw: float = 0.0
    standardized_test_type: str = ""
    standardized_test_score: float = 0.0
    background_university: str = ""
    background_major: str = ""
    target_country: str = ""
    target_universities: list[str] = field(default_factory=list)
    target_majors: list[str] = field(default_factory=list)
    experience_details: dict[str, Any] = field(default_factory=dict)

    prediction_results: dict[str, Any] = field(default_factory=dict)
    ai_explanation: str = ""
    profile_type: str = ""
    matched_products: list[dict[str, Any]] = field(default_factory=list)

    # ---------- 后期：ApplicationAgent 填充 ----------
    application_plan: dict[str, Any] = field(default_factory=dict)

    # ---------- 审计追踪 ----------
    history: list[dict[str, Any]] = field(default_factory=list)

    def record(self, agent_name: str, summary: str, input_len: int = 0) -> None:
        self.history.append(
            {
                "agent": agent_name,
                "summary": summary,
                "input_len": input_len,
                "ts": time.time(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def _dataclass_to_dict(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return {k: _dataclass_to_dict(getattr(obj, k)) for k in obj.__dataclass_fields__}
    if isinstance(obj, list):
        return [_dataclass_to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _dataclass_to_dict(v) for k, v in obj.items()}
    return obj
