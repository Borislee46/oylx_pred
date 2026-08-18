import logging

from pydantic_ai import Agent, PromptedOutput

from src.agent.lead_in.decision import LeadInDecision
from src.agent.lead_in.router_prompts import build_router_system_prompt
from src.agent.lead_in.session_lock import get_session_lock
from src.agent.runtime.model_factory import build_model_with_fallback

_log = logging.getLogger("lead_in_router")


class LeadInRouterAgent:
    def __init__(self, model=None) -> None:
        self._model = model or build_model_with_fallback()
        self._agent = Agent(
            self._model,
            output_type=PromptedOutput(LeadInDecision),
            instructions=build_router_system_prompt(),
        )
        self.agent_name = "LeadInRouterAgent"

    def run(
        self, user_input: str, history: str = "", *, session_id: str | None = None
    ) -> LeadInDecision:
        text = (user_input or "").strip()
        if not text:
            return LeadInDecision(
                intent="vague",
                next_action="ask_clarification",
                feedback="请提供学生信息，例如：本科院校、专业、GPA、语言成绩、目标院校。",
                clarifying_question="学生目前就读哪所大学？什么专业？GPA 和语言成绩大概多少？",
                missing_required=["院校", "专业", "GPA", "语言成绩", "目标院校"],
                confidence="low",
            )
        prompt = f"{history}\n\n## 当前输入\n{text}" if history else text
        lock = get_session_lock(session_id)
        with lock:
            result = self._agent.run_sync(prompt)
        decision = result.output
        _log.info(
            "ROUTER_DECISION | intent=%s action=%s conf=%s missing=%s predict=%s",
            decision.intent,
            decision.next_action,
            decision.confidence,
            decision.missing_required,
            decision.should_predict,
        )
        return decision
