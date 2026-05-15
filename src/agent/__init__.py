from .application_agent import ApplicationAgent
from .background_faculty_agent import BackgroundFacultyAgent
from .base_agent import BaseAgent
from .blind_eval_agent import BlindEvalAgent
from .boundary_case_agent import BoundaryCaseAgent
from .context import StudentContext
from .explain_agent import ExplainAgent
from .explain_profiles import PROFILE_PROMPTS, classify_profile
from .form_bridge import apply_lead_in_to_form
from .lead_in_agent import LeadInAgent
from .orchestrator import AgentOrchestrator
from .registry import AgentRegistry
from .schemas import (
    ExplainResult,
    ExtractedBackground,
    LeadInResult,
)
from .text_preprocessing_agent import TextPreprocessingAgent


def _register_all_agents() -> None:
    AgentRegistry.register("lead_in", LeadInAgent)
    AgentRegistry.register("explain", ExplainAgent)
    AgentRegistry.register("blind_eval", BlindEvalAgent)
    AgentRegistry.register("boundary_case", BoundaryCaseAgent)
    AgentRegistry.register("text_preprocessing", TextPreprocessingAgent)
    AgentRegistry.register("background_faculty", BackgroundFacultyAgent)
    AgentRegistry.register("application", ApplicationAgent)


_register_all_agents()

__all__ = [
    "AgentOrchestrator",
    "AgentRegistry",
    "ApplicationAgent",
    "BackgroundFacultyAgent",
    "BaseAgent",
    "BlindEvalAgent",
    "BoundaryCaseAgent",
    "ExplainAgent",
    "ExplainResult",
    "ExtractedBackground",
    "LeadInAgent",
    "LeadInResult",
    "PROFILE_PROMPTS",
    "StudentContext",
    "TextPreprocessingAgent",
    "apply_lead_in_to_form",
    "classify_profile",
]
