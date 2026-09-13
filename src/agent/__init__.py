from functools import lru_cache

from .background_faculty_agent import BackgroundFacultyAgent
from .background_major_agent import BackgroundMajorAgent
from .background_school_level_agent import SchoolLevelAgent
from .boundary_case_agent import BoundaryCaseAgent
from .context import StudentContext
from .explain_agent import ExplainAgent
from .explain_profiles import SYSTEM_PROMPT, classify_profile
from .schemas import ExtractedBackground
from .text_preprocessing_agent import TextPreprocessingAgent


@lru_cache(maxsize=1)
def get_explain_agent() -> ExplainAgent:
    return ExplainAgent()


@lru_cache(maxsize=1)
def get_school_level_agent() -> SchoolLevelAgent:
    return SchoolLevelAgent()


@lru_cache(maxsize=1)
def get_background_major_agent() -> BackgroundMajorAgent:
    return BackgroundMajorAgent()


@lru_cache(maxsize=1)
def get_lead_in_tool_agent():
    from .lead_in.tool_agent import LeadInToolAgent

    return LeadInToolAgent()


@lru_cache(maxsize=1)
def get_text_preprocessing_agent() -> TextPreprocessingAgent:
    return TextPreprocessingAgent()


@lru_cache(maxsize=1)
def get_background_faculty_agent() -> BackgroundFacultyAgent:
    return BackgroundFacultyAgent()


@lru_cache(maxsize=1)
def get_boundary_case_agent() -> BoundaryCaseAgent:
    return BoundaryCaseAgent()


__all__ = [
    "BackgroundFacultyAgent",
    "BackgroundMajorAgent",
    "BoundaryCaseAgent",
    "SchoolLevelAgent",
    "ExplainAgent",
    "ExtractedBackground",
    "SYSTEM_PROMPT",
    "StudentContext",
    "TextPreprocessingAgent",
    "classify_profile",
    "get_background_faculty_agent",
    "get_background_major_agent",
    "get_boundary_case_agent",
    "get_explain_agent",
    "get_lead_in_tool_agent",
    "get_school_level_agent",
    "get_text_preprocessing_agent",
]
