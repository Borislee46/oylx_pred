from .application_agent import ApplicationAgent
from .background_faculty_agent import BackgroundFacultyAgent
from .base_agent import BaseAgent
from .boundary_case_agent import BoundaryCaseAgent
from .context import StudentContext
from .explain_agent import ExplainAgent
from .form_bridge import apply_lead_in_to_form
from .form_validation_agent import FormValidationAgent
from .lead_in_agent import LeadInAgent
from .orchestrator import AgentOrchestrator
from .registry import AgentRegistry
from .schemas import ExtractedBackground, LeadInResult, PipelineStep
from .text_preprocessing_agent import TextPreprocessingAgent


# ── Centralized agent registration (single source of truth) ──────────────
def _register_all_agents() -> None:
    """Register agent factories at module load time.

    Agents are instantiated lazily on first call to AgentRegistry.get(name),
    so import cost is O(1) regardless of agent count.
    """
    AgentRegistry.register("lead_in", LeadInAgent)
    AgentRegistry.register("explain", ExplainAgent)
    AgentRegistry.register("form_validation", FormValidationAgent)
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
    "BoundaryCaseAgent",
    "ExplainAgent",
    "ExtractedBackground",
    "FormValidationAgent",
    "LeadInAgent",
    "LeadInResult",
    "PipelineStep",
    "StudentContext",
    "TextPreprocessingAgent",
    "apply_lead_in_to_form",
]
