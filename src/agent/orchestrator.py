from __future__ import annotations

import logging
from typing import Any

from src.agent.context import StudentContext
from src.agent.registry import AgentRegistry

_ORCHESTRATOR_LOGGER = logging.getLogger("AgentOrchestrator")


class AgentOrchestrator:
    @staticmethod
    def run(
        agent_name: str,
        context: StudentContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        try:
            agent = AgentRegistry.get(agent_name)
            result = agent.run(context, **kwargs)
        except KeyError:
            _ORCHESTRATOR_LOGGER.error(
                "Agent '%s' 未注册，可用: %s",
                agent_name,
                AgentRegistry.list(),
            )
            return {"_error": f"agent_not_found:{agent_name}"}
        except Exception:
            _ORCHESTRATOR_LOGGER.exception("Agent '%s' 执行异常", agent_name)
            return {"_error": f"agent_error:{agent_name}"}

        context.record(
            agent_name=agent_name,
            summary=_summarize(result),
            input_len=len(context.raw_input),
        )

        return result


def _summarize(result: dict[str, Any], max_len: int = 80) -> str:
    if not result:
        return "empty"
    keys = list(result.keys())
    summary = ", ".join(keys[:6])
    if len(summary) > max_len:
        summary = summary[: max_len - 3] + "..."
    return summary
