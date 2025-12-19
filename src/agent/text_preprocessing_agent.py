from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.text_preprocessing_prompts import build_field_validation_prompt


class TextPreprocessingAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="文本预处理Agent")

    def validate_field(self, field_type: str, content: str) -> bool:
        if not content or not content.strip():
            self.logger.debug(
                f"[{self.agent_name}] 字段验证 - 字段类型: {field_type}, 内容为空，返回 False"
            )
            return False

        prompt = build_field_validation_prompt(field_type, content)

        self.logger.debug(
            f"[{self.agent_name}] 发送字段验证请求 - 字段类型: {field_type}, "
            f"内容长度: {len(content)}"
        )

        content_response = self._call_api(prompt, cache_prefix="field_validation", use_cache=True)
        if content_response is None:
            return False

        s = str(content_response).strip().lower()
        if not s:
            return False

        is_yes = "是" in s or "yes" in s or "true" in s
        is_no = "否" in s or "no" in s or "false" in s

        if is_yes and not is_no:
            is_valid = True
        elif is_no and not is_yes:
            is_valid = False
        else:
            is_valid = s.startswith("是") or s.startswith("yes")

        self.logger.info(
            f"[{self.agent_name}] 字段验证完成 - 字段类型: {field_type}, 验证结果: {is_valid}"
        )
        return is_valid
