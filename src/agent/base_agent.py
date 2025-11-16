import hashlib
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
        cache_ttl: Optional[int] = 3600,
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
        self.cache_ttl = cache_ttl
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

    def _generate_cache_key(self, prompt: str, cache_prefix: Optional[str] = None) -> str:
        key_data = {
            "model": self.model,
            "prompt": prompt,
        }
        key_str = f"{cache_prefix}:{key_data}" if cache_prefix else str(key_data)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def _call_api(
        self,
        prompt: str,
        cache_prefix: Optional[str] = None,
        use_cache: bool = True,
        custom_cache_key: Optional[str] = None,
    ) -> Optional[str]:
        if not self.api_url or not self.api_key:
            self.logger.warning(f"[{self.agent_name}] API 未配置，无法调用")
            return None

        data = self._build_request_data(prompt)

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=self.timeout
            )
            response.raise_for_status()

            try:
                response_json = response.json()
            except ValueError as e:
                self.logger.error(
                    f"[{self.agent_name}] JSON解析失败: {e}, 响应内容: {response.text[:200]}"
                )
                return None

            choices = response_json.get("choices")
            if choices and isinstance(choices, list) and len(choices) > 0:
                message = choices[0].get("message", {})
                if isinstance(message, dict):
                    content = message.get("content", "")
                    if content:
                        result = content.strip()
                        return result
                    else:
                        self.logger.warning(f"[{self.agent_name}] API返回的content为空")
                        return None
                else:
                    self.logger.error(
                        f"[{self.agent_name}] API返回的message格式不正确: {type(message)}"
                    )
                    return None
            else:
                error_info = response_json.get("error", {})
                if error_info:
                    error_message = (
                        error_info.get("message", "未知错误")
                        if isinstance(error_info, dict)
                        else str(error_info)
                    )
                    self.logger.error(f"[{self.agent_name}] API返回错误: {error_message}")
                else:
                    self.logger.error(f"[{self.agent_name}] API响应格式异常，未找到choices字段")
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
