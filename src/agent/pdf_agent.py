import hashlib
import json
import time
from typing import Any, Dict, Optional

from src.agent.base_agent import BaseAgent
from src.pages.prediction.page_components.pdf_generation.utils import pdf_cache


class PDFAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, timeout=15, agent_name="PDFAgent")
        self.cache_ttl = 86400

        if not self.api_url or not self.api_key:
            self.logger.warning(
                f"[{self.agent_name}] API URL 或 API Key 未配置，将使用fallback建议"
            )

    def _truncate_if_needed(self, content: str, max_length: int = 250) -> str:
        content_without_tags = content.replace("<b>", "").replace("</b>", "").replace(
            "<br/>", "\n"
        )
        actual_length = len(content_without_tags)

        if actual_length <= max_length:
            return content

        lines = content.split("<br/>")
        truncated_lines = []
        current_length = 0
        header_present = False

        for line in lines:
            line_clean = line.replace("<b>", "").replace("</b>", "").strip()

            if "<b>分析师建议:</b>" in line:
                header_present = True
                truncated_lines.append(line)
                continue

            if not line_clean or line_clean.startswith("-"):
                line_length = len(line_clean)

                if current_length + line_length <= max_length:
                    truncated_lines.append(line)
                    current_length += line_length
                else:
                    remaining = max_length - current_length
                    if remaining > 30 and line_clean.startswith("-"):
                        truncated_line = line_clean[:remaining] + "..."
                        truncated_lines.append(truncated_line)
                    break

        if truncated_lines:
            result = "<br/>".join(truncated_lines)
            if not header_present and "<b>分析师建议:</b>" not in result:
                result = (
                    "<b>分析师建议:</b><br/>" + result
                    if result.strip()
                    else "<b>分析师建议:</b>"
                )
            return result

        return "<b>分析师建议:</b>" if not content else content[:max_length] + "..."

    def _generate_cache_key(
        self, user_data: Dict[str, Any], soft_skills: Dict[str, Any]
    ) -> str:
        key_data = {
            "user_data": {
                k: v
                for k, v in user_data.items()
                if k
                in (
                    "gpa_score",
                    "gpa_raw",
                    "gpa_scale",
                    "background_university",
                    "background_major",
                    "language_type",
                    "language_score",
                    "language_score_raw",
                )
            },
            "soft_skills": soft_skills,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def generate_analyst_notes(
        self, user_data: Dict[str, Any], soft_skills: Dict[str, Any]
    ) -> Optional[str]:
        self.logger.info(f"[{self.agent_name}] 开始生成分析师建议")

        if not self.api_url or not self.api_key:
            self.logger.warning(
                f"[{self.agent_name}] API URL 或 API Key 未配置，无法生成分析师建议"
            )
            return None

        cache_key = f"analyst_notes:{self._generate_cache_key(user_data, soft_skills)}"
        self.logger.debug(f"[{self.agent_name}] 缓存键: {cache_key[:50]}...")

        pdf_cache.clear_expired()
        cached_data = pdf_cache.get(cache_key)

        if cached_data and isinstance(cached_data, dict):
            cached_value = cached_data.get("value")
            if cached_value:
                self.logger.info(
                    f"[{self.agent_name}] 使用缓存的分析师建议，长度: {len(cached_value)}"
                )
                return cached_value

        self.logger.info(f"[{self.agent_name}] 缓存未命中，开始调用API生成分析师建议")

        from src.agent.prompts import build_analyst_notes_prompt

        prompt = build_analyst_notes_prompt(user_data, soft_skills)
        prompt_length = len(prompt)
        self.logger.info(
            f"[{self.agent_name}] 构建prompt完成，长度: {prompt_length}，模型: {self.model}，"
            f"API URL: {self.api_url}"
        )

        start_time = time.time()
        self.logger.info(
            f"[{self.agent_name}] 发送API请求，超时设置: {self.timeout}秒"
        )

        content = self._call_api(prompt)
        elapsed_time = time.time() - start_time

        if content:
            original_length = len(content)
            result = self._truncate_if_needed(content.strip())
            result_length = len(result)
            self.logger.info(
                f"[{self.agent_name}] 成功获取分析师建议，原始长度: {original_length}，"
                f"截断后长度: {result_length}，耗时: {elapsed_time:.2f}秒"
            )
            pdf_cache.set(cache_key, result, self.cache_ttl)
            self.logger.info(
                f"[{self.agent_name}] 已缓存分析师建议，TTL: {self.cache_ttl}秒"
            )
            return result
        else:
            self.logger.warning(
                f"[{self.agent_name}] 获取分析师建议失败，耗时: {elapsed_time:.2f}秒，"
                f"使用fallback建议"
            )
            return None

