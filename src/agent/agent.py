from typing import Any, Dict, Optional

import pandas as pd
import requests

from src.agent.prompts import build_consultation_prompt
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

consultation_agent_logger = setup_logger("page3", "prediction")


class AIAgent:
    def __init__(
        self,
        user_profile: Optional[Dict[str, Any]],
        prediction_results: Any,
        cases_df: pd.DataFrame,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.cases_df = cases_df

        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")

        if not self.api_url or not self.api_key:
            consultation_agent_logger.error("[咨询Agent] API URL 或 API Key 未在配置中找到。")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        self.update_context(user_profile, prediction_results)

    def update_context(self, user_profile: Optional[Dict[str, Any]], prediction_results: Any):
        self.user_profile = user_profile if user_profile else {}
        self.prediction_results = prediction_results

    def run(self, user_query: str) -> str:
        prompt = build_consultation_prompt(
            user_query=user_query,
            user_profile=self.user_profile,
            prediction_results=self.prediction_results,
        )

        data = {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=data, timeout=5)
            response.raise_for_status()

            response_json = response.json()

            if response_json.get("choices"):
                content = response_json["choices"][0].get("message", {}).get("content", "")
                return content
            else:
                error_info = response_json.get("error", {})
                error_message = error_info.get("message", "未知错误")
                consultation_agent_logger.error(f"[咨询Agent] API返回错误: {error_message}")
                return ""

        except requests.exceptions.Timeout:
            consultation_agent_logger.warning("[咨询Agent] 请求超时（5秒），使用fallback响应")
            return ""
        except requests.exceptions.RequestException as e:
            consultation_agent_logger.error(f"[咨询Agent] 网络请求失败: {e}")
            return ""
        except Exception as e:
            consultation_agent_logger.error(f"[咨询Agent] 未知错误: {e}", exc_info=True)
            return ""
