from typing import Any, Dict

import requests

from src.agent.prompts import build_field_validation_prompt
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

text_preprocessing_logger = setup_logger("page3", "prediction")


class TextPreprocessingAgent:
    def __init__(self, config: Dict[str, Any] | None = None):
        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")

        if not self.api_url or not self.api_key:
            text_preprocessing_logger.error(
                "[文本预处理Agent] API URL 或 API Key 未在配置中找到。"
            )

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def validate_field(self, field_type: str, content: str) -> bool:
        if not content or not content.strip():
            text_preprocessing_logger.debug(
                f"[文本预处理Agent] 字段验证 - 字段类型: {field_type}, 内容为空，返回 False"
            )
            return False

        prompt = build_field_validation_prompt(field_type, content)

        data = {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
        }

        try:
            text_preprocessing_logger.debug(
                f"[文本预处理Agent] 发送字段验证请求 - 字段类型: {field_type}, "
                f"内容长度: {len(content)}"
            )
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=5
            )
            response.raise_for_status()

            response_json = response.json()

            if response_json.get("choices"):
                content_response = response_json["choices"][0].get("message", {}).get(
                    "content", ""
                )
                content_response = content_response.strip().lower()

                is_valid = content_response == "是" or content_response.startswith("是")

                text_preprocessing_logger.info(
                    f"[文本预处理Agent] 字段验证完成 - 字段类型: {field_type}, "
                    f"验证结果: {is_valid}"
                )
                return is_valid
            else:
                error_info = response_json.get("error", {})
                error_message = error_info.get("message", "未知错误")
                text_preprocessing_logger.error(
                    f"[文本预处理Agent] API返回错误: {error_message}"
                )
                return False

        except requests.exceptions.Timeout:
            text_preprocessing_logger.warning(
                f"[文本预处理Agent] 请求超时（5秒），使用fallback输出（返回False，使用原始输入）"
            )
            return False
        except requests.exceptions.RequestException as e:
            text_preprocessing_logger.error(
                f"[文本预处理Agent] 网络请求失败 - 错误: {e}"
            )
            return False
        except Exception as e:
            text_preprocessing_logger.error(
                f"[文本预处理Agent] 未知错误 - 错误类型: {type(e).__name__}, 错误信息: {e}",
                exc_info=True,
            )
            return False

