"""Lead-in subsystem — 单智能体 ReAct：回答问题 / 追问 / 查数据 / 写表单 / 触发预测。

只有一个执行体 `LeadInToolAgent`：由它自己决定动作，不再有 router / tools 分流。
分流会让"提问型输入"既拿不到回答、又白跑一轮工具循环。
"""

from src.agent.lead_in.dispatcher import LeadInDispatcher
from src.agent.lead_in.state_machine import (
    LeadInPhase,
    LeadInTurnState,
    LeadInTurnStateMachine,
)
from src.agent.lead_in.tool_agent import LeadInToolAgent

__all__ = [
    "LeadInDispatcher",
    "LeadInPhase",
    "LeadInToolAgent",
    "LeadInTurnState",
    "LeadInTurnStateMachine",
]
