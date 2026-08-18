"""Lead-in subsystem — NLU → form extraction → dispatch.

Two paths:
  - Router:  LeadInRouterAgent (structured-output routing, Stage A)
  - Tools:   LeadInToolAgent (tool-calling ReAct loop, Stage B)

Dispatcher selects tools by default, falls back to router on failure.
"""

from src.agent.lead_in.decision import LeadInDecision
from src.agent.lead_in.dispatcher import LeadInDispatcher
from src.agent.lead_in.router_agent import LeadInRouterAgent
from src.agent.lead_in.state_machine import (
    LeadInPhase,
    LeadInTurnState,
    LeadInTurnStateMachine,
)
from src.agent.lead_in.tool_agent import LeadInToolAgent

__all__ = [
    "LeadInDecision",
    "LeadInDispatcher",
    "LeadInPhase",
    "LeadInRouterAgent",
    "LeadInToolAgent",
    "LeadInTurnState",
    "LeadInTurnStateMachine",
]
