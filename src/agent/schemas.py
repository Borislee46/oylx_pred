"""TypedDict schemas for Agent system data contracts.

These make the otherwise bare ``dict[str, Any]`` payloads introspectable
and serve as documentation for what each agent produces / consumes.
"""

from __future__ import annotations

from typing import TypedDict


class ExtractedBackground(TypedDict, total=False):
    """Output schema for LeadInAgent.extracted_background.

    All keys are optional — the agent fills whatever it can infer from the
    consultant's free-text input. Missing keys mean the agent hasn't
    extracted that piece of information yet.
    """

    university: str
    major: str
    gpa: float
    language_score: float
    language_type: str
    country: str
    target_schools: list[str]
    target_majors: list[str]
    research: str
    internship: str
    award: str
    paper: str


class LeadInResult(TypedDict, total=False):
    """Return type for LeadInAgent.run()."""

    extracted_info: ExtractedBackground
    quick_assessment: str
    suggested_questions: list[str]
    _error: str


class PipelineStep(TypedDict):
    """A single step in AgentOrchestrator.run_pipeline()."""

    agent: str
    kwargs: dict[str, object]
