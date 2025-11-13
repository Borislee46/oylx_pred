from typing import Any, Dict, Optional

import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.prompts import build_consultation_prompt


class AIAgent(BaseAgent):
    def __init__(
        self,
        user_profile: Optional[Dict[str, Any]],
        prediction_results: Any,
        cases_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(config=config, timeout=5, agent_name="咨询Agent")
        self.cases_df = cases_df
        self.update_context(user_profile, prediction_results)

    def update_context(self, user_profile: Optional[Dict[str, Any]], prediction_results: Any):
        self.user_profile = user_profile if user_profile else {}
        self.prediction_results = prediction_results

    def run(self, user_query: str) -> Optional[str]:
        if not user_query or not user_query.strip():
            self.logger.warning(f"[{self.agent_name}] 用户查询为空")
            return None

        prompt = build_consultation_prompt(
            user_query=user_query,
            user_profile=self.user_profile,
            prediction_results=self.prediction_results,
        )

        content = self._call_api(prompt)
        if not content:
            self.logger.warning(f"[{self.agent_name}] API调用失败，返回空响应")
        return content
