from abc import ABC
from typing import Any, Dict, Optional

import requests

from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger


class BaseAgent(ABC):
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 5,
        agent_name: str = "Agent",
    ):
        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")
        self.timeout = timeout
        self.agent_name = agent_name
        self.logger = setup_logger("page3", "prediction")

        if not self.api_url or not self.api_key:
            self.logger.error(f"[{self.agent_name}] API URL 或 API Key 未在配置中找到。")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _build_request_data(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
        }

    def _call_api(self, prompt: str) -> Optional[str]:
        if not self.api_url or not self.api_key:
            self.logger.warning(f"[{self.agent_name}] API 未配置，无法调用")
            return None

        data = self._build_request_data(prompt)

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()

            response_json = response.json()

            if response_json.get("choices"):
                content = response_json["choices"][0].get("message", {}).get("content", "")
                return content.strip() if content else None
            else:
                error_info = response_json.get("error", {})
                error_message = error_info.get("message", "未知错误")
                self.logger.error(f"[{self.agent_name}] API返回错误: {error_message}")
                return None

        except requests.exceptions.Timeout:
            self.logger.warning(
                f"[{self.agent_name}] 请求超时（{self.timeout}秒），使用fallback响应"
            )
            return None
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[{self.agent_name}] 网络请求失败: {e}")
            return None
        except Exception as e:
            self.logger.error(
                f"[{self.agent_name}] 未知错误 - 错误类型: {type(e).__name__}, 错误信息: {e}",
                exc_info=True,
            )
            return None
