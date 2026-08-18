"""Signals Agent — AI Harness for admission prediction.

Simplified in 2026-07: evaluator, observability, validation, gateways, intent gate,
session continuity LLM, and 3b ReAct harness were deleted. ~5,800 lines removed.
"""

from functools import lru_cache

from .background_major_agent import BackgroundMajorAgent
from .background_school_level_agent import SchoolLevelAgent
from .context import StudentContext
from .explain_agent import ExplainAgent
from .explain_profiles import SYSTEM_PROMPT, classify_profile
from .schemas import ExtractedBackground
from .text_preprocessing_agent import TextPreprocessingAgent

# ── Agent factories (replaces AgentRegistry string-based lookup) ─────────
# Each factory is a plain function — no registry overhead.
# Non-singleton: fresh instance per call (thread-safe isolation).
# Singleton: @lru_cache for expensive pydantic-ai Agent construction.


def get_explain_agent() -> ExplainAgent:
    """Return a new ExplainAgent instance (thread-safe per-call isolation)."""
    return ExplainAgent()


def get_school_level_agent() -> SchoolLevelAgent:
    """Return a new SchoolLevelAgent instance."""
    return SchoolLevelAgent()


def get_background_major_agent() -> BackgroundMajorAgent:
    """Return a new BackgroundMajorAgent instance."""
    return BackgroundMajorAgent()


@lru_cache(maxsize=1)
def get_lead_in_router_agent():
    """Cached singleton for LeadInRouterAgent (pydantic-ai construction is expensive)."""
    from .lead_in.router_agent import LeadInRouterAgent

    return LeadInRouterAgent()


@lru_cache(maxsize=1)
def get_lead_in_tool_agent():
    """Cached singleton for LeadInToolAgent (pydantic-ai construction is expensive)."""
    from .lead_in.tool_agent import LeadInToolAgent

    return LeadInToolAgent()


__all__ = [
    "BackgroundMajorAgent",
    "SchoolLevelAgent",
    "ExplainAgent",
    "ExtractedBackground",
    "SYSTEM_PROMPT",
    "StudentContext",
    "TextPreprocessingAgent",
    "classify_profile",
    "get_background_major_agent",
    "get_explain_agent",
    "get_lead_in_router_agent",
    "get_lead_in_tool_agent",
    "get_school_level_agent",
]
