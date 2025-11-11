import json
from typing import Any, Optional

import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.prompts import build_boundary_evaluation_prompt


class BoundaryCaseAgent(BaseAgent):
    def __init__(self, cases_df: pd.DataFrame, config: Optional[dict[str, Any]] = None):
        super().__init__(config=config, timeout=5, agent_name="边界CaseAgent")
        self.cases_df = cases_df

    def evaluate_boundary_cases(
        self, background_major: str, boundary_cases: list[dict[str, Any]], mode: str
    ) -> dict[str, Any]:
        fallback_result = {
            "decisions": [False] * len(boundary_cases) if boundary_cases else [],
            "needs_adjustment": False,
            "reason": "",
        }

        if not boundary_cases:
            self.logger.warning(f"[{self.agent_name}] 边界案例列表为空，跳过评估")
            return {"decisions": [], "needs_adjustment": False, "reason": "无待评估案例"}

        self.logger.info(
            f"[{self.agent_name}] 开始评估 - 模式: {mode}, 背景专业: {background_major}, "
            f"案例数量: {len(boundary_cases)}"
        )

        prompt = build_boundary_evaluation_prompt(background_major, boundary_cases, mode)

        self.logger.debug(
            f"[{self.agent_name}] 发送请求到 {self.api_url}, 模型: {self.model}, "
            f"prompt长度: {len(prompt)}"
        )

        content = self._call_api(prompt)
        if content is None:
            return fallback_result

        content = self._clean_json_content(content)

        try:
            result = json.loads(content)

            decisions = result.get("decisions", [])
            needs_adjustment = result.get("needs_adjustment", True)
            reason = result.get("reason", "")

            if len(decisions) != len(boundary_cases):
                self.logger.warning(
                    f"[{self.agent_name}] decisions 长度 ({len(decisions)}) 与 boundary_cases 长度 "
                    f"({len(boundary_cases)}) 不匹配"
                )
                decisions = [False] * len(boundary_cases)
                needs_adjustment = False

            result = {
                "decisions": decisions,
                "needs_adjustment": needs_adjustment,
                "reason": reason,
            }
            self.logger.info(
                f"[{self.agent_name}] 评估完成 - 需要调整: {needs_adjustment}, "
                f"决策数: {len(decisions)}, 通过数: {sum(decisions) if decisions else 0}"
            )
            return result

        except json.JSONDecodeError as e:
            self.logger.error(
                f"[{self.agent_name}] JSON解析失败 - 错误: {e}, 响应内容长度: {len(content)}"
            )
            self.logger.debug(f"[{self.agent_name}] 响应内容预览: {content[:200]}")
            return fallback_result

    def _clean_json_content(self, content: str) -> str:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        return content.strip()
