import hashlib
import json
import time
from typing import Any, Dict, Optional

import requests

from src.pages.prediction.page_components.pdf_generation.utils import pdf_cache
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class PDFAIAgent:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")
        self.timeout = 15
        self.cache_ttl = 86400

        if not self.api_url or not self.api_key:
            logger.warning("PDFAIAgent: API URL 或 API Key 未配置，将使用fallback建议")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _truncate_if_needed(self, content: str, max_length: int = 250) -> str:
        content_without_tags = content.replace("<b>", "").replace("</b>", "").replace("<br/>", "\n")
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
                    "<b>分析师建议:</b><br/>" + result if result.strip() else "<b>分析师建议:</b>"
                )
            return result

        return "<b>分析师建议:</b>" if not content else content[:max_length] + "..."

    def _generate_cache_key(self, user_data: Dict[str, Any], soft_skills: Dict[str, Any]) -> str:
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
        logger.info("PDFAIAgent: 开始生成分析师建议")

        if not self.api_url or not self.api_key:
            logger.warning("PDFAIAgent: API URL 或 API Key 未配置，无法生成分析师建议")
            return None

        cache_key = f"analyst_notes:{self._generate_cache_key(user_data, soft_skills)}"
        logger.debug(f"PDFAIAgent: 缓存键: {cache_key[:50]}...")

        pdf_cache.clear_expired()
        cached_data = pdf_cache.get(cache_key)

        if cached_data and isinstance(cached_data, dict):
            cached_value = cached_data.get("value")
            if cached_value:
                logger.info(f"PDFAIAgent: 使用缓存的分析师建议，长度: {len(cached_value)}")
                return cached_value

        logger.info("PDFAIAgent: 缓存未命中，开始调用API生成分析师建议")

        from src.agent.pdf_prompts import build_analyst_notes_prompt

        prompt = build_analyst_notes_prompt(user_data, soft_skills)
        prompt_length = len(prompt)
        logger.info(
            f"PDFAIAgent: 构建prompt完成，长度: {prompt_length}，模型: {self.model}，API URL: {self.api_url}"
        )

        data = {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
        }

        start_time = time.time()
        try:
            logger.info(f"PDFAIAgent: 发送API请求，超时设置: {self.timeout}秒")
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=self.timeout
            )
            elapsed_time = time.time() - start_time
            logger.info(
                f"PDFAIAgent: API请求完成，状态码: {response.status_code}，耗时: {elapsed_time:.2f}秒"
            )

            response.raise_for_status()

            response_json = response.json()

            if response_json.get("choices"):
                content = response_json["choices"][0].get("message", {}).get("content", "")
                if content and content.strip():
                    original_length = len(content)
                    result = self._truncate_if_needed(content.strip())
                    result_length = len(result)
                    logger.info(
                        f"PDFAIAgent: 成功获取分析师建议，原始长度: {original_length}，"
                        f"截断后长度: {result_length}，耗时: {elapsed_time:.2f}秒"
                    )
                    pdf_cache.set(cache_key, result, self.cache_ttl)
                    logger.info(f"PDFAIAgent: 已缓存分析师建议，TTL: {self.cache_ttl}秒")
                    return result
                else:
                    logger.warning("PDFAIAgent: API返回的content为空")
            else:
                logger.warning("PDFAIAgent: API响应中未找到choices字段")

            error_info = response_json.get("error", {})
            error_message = error_info.get("message", "未知错误")
            logger.warning(
                f"PDFAIAgent: API返回错误: {error_message}，响应: {json.dumps(response_json, ensure_ascii=False)[:200]}"
            )
            return None

        except requests.exceptions.Timeout:
            elapsed_time = time.time() - start_time
            logger.warning(
                f"PDFAIAgent: 请求超时（{self.timeout}秒），实际耗时: {elapsed_time:.2f}秒，使用fallback建议"
            )
            return None
        except requests.exceptions.RequestException as e:
            elapsed_time = time.time() - start_time
            logger.warning(
                f"PDFAIAgent: API调用失败: {e}，耗时: {elapsed_time:.2f}秒，使用fallback建议"
            )
            return None
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.warning(
                f"PDFAIAgent: 生成分析师建议时发生错误: {e}，耗时: {elapsed_time:.2f}秒，使用fallback建议",
                exc_info=True,
            )
            return None
