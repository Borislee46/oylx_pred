import hashlib
import json
import os
import time
from collections import OrderedDict
from typing import Any

import requests

from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger


class BaseAgent:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        timeout: int = 10,
        agent_name: str = "Agent",
        cache_ttl: int | None = 3600,
    ):
        if config is None:
            app_config = load_app_config()
        else:
            app_config = config

        self.api_url = app_config.get("OPEN_AI_BASE_URL")
        self.api_key = app_config.get("OPEN_AI_API_KEY")
        self.model = app_config.get("OPEN_AI_MODEL")
        self.thinking_type = str(
            app_config.get("OPEN_AI_THINKING", "disabled") or "disabled"
        ).strip()
        self.max_tokens = app_config.get("OPEN_AI_MAX_TOKENS")
        self.timeout = timeout
        self.agent_name = agent_name
        self.cache_ttl = cache_ttl
        self.logger = setup_logger("page3", "prediction")
        self._session: requests.Session | None = None
        self._memory_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

        if not self.api_url or not self.api_key:
            self.logger.error(f"[{self.agent_name}] API URL 或 API Key 未在配置中找到。")

        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
        return self._session

    def close(self):
        if self._session is not None:
            self._session.close()
            self._session = None

    def _load_persistent_json(self, cache_dir: str, cache_file: str) -> dict[str, Any]:
        file_path = os.path.join(cache_dir, cache_file)
        if not os.path.exists(file_path):
            return {}
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f"[{self.agent_name}] 加载持久化缓存失败: {e}")
            return {}

    def _save_persistent_json(self, cache_dir: str, cache_file: str, data: dict[str, Any]) -> None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            file_path = os.path.join(cache_dir, cache_file)
            tmp_path = f"{file_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, file_path)
        except OSError as e:
            self.logger.error(f"[{self.agent_name}] 保存持久化缓存失败: {e}")

    def _build_request_data(
        self,
        prompt: str,
        thinking_type: str | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": thinking_type or self.thinking_type or "disabled"},
        }

        resolved_max_tokens: int | None = None
        if max_tokens is not None:
            try:
                resolved_max_tokens = int(max_tokens)
            except (TypeError, ValueError):
                resolved_max_tokens = None
        elif self.max_tokens is not None:
            try:
                resolved_max_tokens = int(self.max_tokens)
            except (TypeError, ValueError):
                resolved_max_tokens = None

        if resolved_max_tokens and resolved_max_tokens > 0:
            data["max_tokens"] = resolved_max_tokens

        return data

    def _generate_cache_key(
        self,
        prompt: str,
        cache_prefix: str | None = None,
        thinking_type: str | None = None,
        max_tokens: int | None = None,
    ) -> str:
        key_data = {
            "model": self.model,
            "prompt": prompt,
            "thinking": thinking_type or self.thinking_type or "disabled",
        }
        if max_tokens is not None:
            key_data["max_tokens"] = max_tokens
        elif self.max_tokens is not None:
            key_data["max_tokens"] = self.max_tokens
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        key_str = f"{cache_prefix}:{payload}" if cache_prefix else payload
        return hashlib.md5(key_str.encode("utf-8")).hexdigest()

    def _get_from_cache(self, key: str) -> str | None:
        if key not in self._memory_cache:
            return None

        cache_entry = self._memory_cache[key]
        if self.cache_ttl and (time.time() - cache_entry["timestamp"] > self.cache_ttl):
            del self._memory_cache[key]
            return None

        self._memory_cache.move_to_end(key)
        return cache_entry["value"]

    def _extract_text_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for it in content:
                if isinstance(it, str):
                    if it.strip():
                        parts.append(it)
                    continue
                if isinstance(it, dict):
                    t = it.get("text")
                    if isinstance(t, str) and t.strip():
                        parts.append(t)
                    continue
            return "\n".join(parts).strip()
        if isinstance(content, dict):
            t = content.get("text")
            if isinstance(t, str):
                return t
        return ""

    def _save_to_cache(self, key: str, value: str) -> None:
        if key in self._memory_cache:
            del self._memory_cache[key]
        self._memory_cache[key] = {"value": value, "timestamp": time.time()}
        self._memory_cache.move_to_end(key)
        while len(self._memory_cache) > 1000:
            self._memory_cache.popitem(last=False)

    def _estimate_tokens(self, text: str) -> float:
        if not text:
            return 0.0
        token_count = 0.0
        for char in text:
            if ord(char) < 128:
                token_count += 0.3
            else:
                token_count += 0.6
        return round(token_count, 2)

    def _clean_json_content(self, content: str) -> str:
        if not content:
            return ""

        content = content.strip()

        if content.startswith("```"):
            newline_idx = content.find("\n")
            if newline_idx != -1:
                content = content[newline_idx + 1 :]
            else:
                content = content[3:]

        if content.endswith("```"):
            content = content[:-3]

        content = content.strip()

        start_idx = content.find("{")
        end_idx = content.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            content = content[start_idx : end_idx + 1]

        return content.strip()

    def _fix_json_lightweight(self, content: str) -> str | None:
        """Fix common LLM JSON formatting without an API call.

        Handles: missing commas between fields, trailing commas, extra text.
        Returns fixed JSON string or None if unfixable.
        """
        import re

        if not content or not content.strip():
            return None

        s = content.strip()

        # 1. Missing comma after string value, before next key (any whitespace including newlines)
        # Lookahead ensures the second quote starts a key (letter/underscore), avoiding
        # false matches on empty string values like "key": ""
        s = re.sub(r'("\s*)(?="\s*[a-zA-Z_])', r"\1,", s)

        # 2. Missing comma after } or ] before next key
        s = re.sub(r'([}\]]\s*)(")', r"\1,\2", s)

        # 3. Missing comma after number/true/false/null before next key
        s = re.sub(r'(\d|true|false|null)(\s+)(")', r"\1, \3", s)

        # 4. Trailing comma before } or ]
        s = re.sub(r",(\s*[}\]])", r"\1", s)

        # 5. Double commas
        s = re.sub(r",\s*,", ",", s)

        if s == content.strip():
            return None  # no changes made
        return s

    def _parse_json_response(
        self,
        raw: str | None,
        schema_hint: str,
        cache_prefix: str = "json_repair",
        max_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        """Three-tier JSON parsing: direct → lightweight regex → API repair."""
        if not raw:
            self.logger.warning(f"[{self.agent_name}] PARSE: empty raw response")
            return None

        content = self._clean_json_content(raw)

        # Tier 1: direct parse
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            self.logger.warning(f"[{self.agent_name}] PARSE: JSON decode failed | error={e}")

        # Tier 2: lightweight regex fix (no API call)
        fixed = self._fix_json_lightweight(content)
        if fixed:
            try:
                result = json.loads(fixed)
                self.logger.info(f"[{self.agent_name}] PARSE: lightweight fix succeeded")
                return result
            except json.JSONDecodeError:
                pass

        # Tier 3: API repair
        self.logger.info(f"[{self.agent_name}] PARSE: lightweight fix failed, trying API repair")
        repaired = self._repair_json_once(content, schema_hint, cache_prefix, max_tokens=max_tokens)
        if not repaired:
            self.logger.warning(f"[{self.agent_name}] PARSE: API repair returned empty")
            return None
        try:
            return json.loads(repaired)
        except json.JSONDecodeError as e2:
            self.logger.warning(f"[{self.agent_name}] PARSE: API repair still invalid | error={e2}")
            return None

    def _repair_json_once(
        self,
        raw_text: str,
        schema_hint: str,
        cache_prefix: str = "json_repair",
        thinking_type: str | None = None,
        max_tokens: int | None = None,
    ) -> str | None:
        raw_text = str(raw_text or "").strip()
        if not raw_text:
            return None

        prompt = (
            "你是一个 JSON 修复器。你的任务是把输入内容修复为“严格合法”的 JSON。\n"
            "要求：\n"
            "- 只输出 JSON，本次输出不得包含任何解释、注释、代码块标记、前后缀文字。\n"
            "- 如果输入包含多余文字，请删除，只保留 JSON。\n"
            "- 字段必须严格符合 schema_hint，不要新增字段。\n\n"
            f"schema_hint:\n{schema_hint}\n\n"
            "input:\n"
            f"{raw_text}\n"
        )
        fixed = self._call_api(
            prompt,
            cache_prefix=cache_prefix,
            use_cache=True,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
        )
        if not fixed:
            return None
        fixed = self._clean_json_content(fixed)
        return fixed if fixed else None

    def _call_api_streaming(
        self,
        prompt: str,
        thinking_type: str | None = None,
        max_tokens: int | None = None,
        max_retries: int = 1,
    ):
        """Stream API response, yielding text chunks. Use as generator for st.write_stream.

        Yields str chunks. The caller should collect them and parse the full response.
        Retries once on timeout / connection errors.
        """
        if not self.api_url or not self.api_key:
            self.logger.warning(f"[{self.agent_name}] 流式API未配置")
            return

        thinking = thinking_type or self.thinking_type or "disabled"
        input_tokens = self._estimate_tokens(prompt)
        self.logger.info(
            f"[{self.agent_name}] STREAM START | prompt={len(prompt)}chars ~{input_tokens}tk | "
            f"max_tokens={max_tokens}"
        )

        data = self._build_request_data(prompt, thinking_type=thinking_type, max_tokens=max_tokens)
        data["stream"] = True

        t_start = time.perf_counter()
        for attempt in range(max_retries + 1):
            t_attempt = time.perf_counter()
            try:
                response = self.session.post(
                    self.api_url,
                    headers=self.headers,
                    json=data,
                    timeout=self.timeout,
                    stream=True,
                )
                response.raise_for_status()

                chunk_count = 0
                for line in response.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    if not payload:
                        continue
                    try:
                        chunk = json.loads(payload)
                        choices = chunk.get("choices")
                        if choices and isinstance(choices, list):
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                chunk_count += 1
                                yield content
                    except json.JSONDecodeError:
                        continue

                elapsed_ms = (time.perf_counter() - t_start) * 1000
                self.logger.info(
                    f"[{self.agent_name}] STREAM OK | total={elapsed_ms:.0f}ms | "
                    f"chunks={chunk_count}"
                )
                return  # success, exit retry loop

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{self.agent_name}] STREAM RETRY {attempt + 1}/{max_retries} | "
                        f"elapsed={elapsed_ms:.0f}ms | type={type(e).__name__} | error={e}"
                    )
                    time.sleep(1)
                    continue
                total_ms = (time.perf_counter() - t_start) * 1000
                self.logger.error(
                    f"[{self.agent_name}] STREAM ERROR | "
                    f"total={total_ms:.0f}ms | type={type(e).__name__} | error={e}"
                )
                return

            except Exception as e:
                total_ms = (time.perf_counter() - t_start) * 1000
                self.logger.error(
                    f"[{self.agent_name}] STREAM ERROR | "
                    f"total={total_ms:.0f}ms | type={type(e).__name__} | error={e}"
                )
                return

    def _call_api(
        self,
        prompt: str,
        cache_prefix: str | None = None,
        use_cache: bool = True,
        custom_cache_key: str | None = None,
        thinking_type: str | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
    ) -> str | None:
        if not self.api_url or not self.api_key:
            self.logger.warning(f"[{self.agent_name}] API 未配置，无法调用")
            return None

        cache_key = custom_cache_key or self._generate_cache_key(
            prompt,
            cache_prefix=cache_prefix,
            thinking_type=thinking_type,
            max_tokens=max_tokens,
        )

        input_tokens = self._estimate_tokens(prompt)
        prompt_chars = len(prompt)

        if use_cache:
            cached_result = self._get_from_cache(cache_key)
            if cached_result is not None:
                output_tokens = self._estimate_tokens(cached_result)
                self.logger.info(
                    f"[{self.agent_name}] CACHE HIT | key={cache_key[:12]}... | "
                    f"prompt={prompt_chars}chars ~{input_tokens}tk | "
                    f"output ~{output_tokens}tk"
                )
                return cached_result

        thinking = thinking_type or self.thinking_type or "disabled"
        self.logger.info(
            f"[{self.agent_name}] REQ START | prompt={prompt_chars}chars ~{input_tokens}tk | "
            f"max_tokens={max_tokens} | thinking={thinking} | timeout={self.timeout}s | "
            f"url={self.api_url[:60]}..."
        )

        t_start = time.perf_counter()
        data = self._build_request_data(prompt, thinking_type=thinking_type, max_tokens=max_tokens)

        last_error: str | None = None
        for attempt in range(max_retries + 1):
            t_attempt = time.perf_counter()
            try:
                self.logger.info(f"[{self.agent_name}] → attempt {attempt + 1}/{max_retries + 1}")
                response = self.session.post(
                    self.api_url, headers=self.headers, json=data, timeout=self.timeout
                )
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000

                if not response.ok:
                    self.logger.warning(
                        f"[{self.agent_name}] HTTP {response.status_code} | "
                        f"elapsed={elapsed_ms:.0f}ms | body={response.text[:300]}"
                    )
                response.raise_for_status()

                try:
                    response_json = response.json()
                except ValueError as e:
                    self.logger.error(
                        f"[{self.agent_name}] JSON解析失败 | elapsed={elapsed_ms:.0f}ms | "
                        f"error={e} | body={response.text[:300]}"
                    )
                    return None

                choices = response_json.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        raw_content = message.get("content", "")
                        result = self._extract_text_content(raw_content).strip()
                        if result:
                            total_elapsed_ms = (time.perf_counter() - t_start) * 1000
                            output_tokens = self._estimate_tokens(result)
                            output_chars = len(result)
                            real_usage = response_json.get("usage", {})
                            usage_str = ""
                            if real_usage:
                                usage_str = (
                                    f" | usage: prompt={real_usage.get('prompt_tokens', '?')} "
                                    f"completion={real_usage.get('completion_tokens', '?')} "
                                    f"total={real_usage.get('total_tokens', '?')}"
                                )

                            self.logger.info(
                                f"[{self.agent_name}] REQ OK | "
                                f"total={total_elapsed_ms:.0f}ms | "
                                f"attempts={attempt + 1} | "
                                f"prompt={prompt_chars}chars ~{input_tokens}tk | "
                                f"output={output_chars}chars ~{output_tokens}tk"
                                f"{usage_str}"
                            )

                            if use_cache:
                                self._save_to_cache(cache_key, result)
                            return result
                        else:
                            self.logger.warning(
                                f"[{self.agent_name}] API返回content为空 | "
                                f"elapsed={elapsed_ms:.0f}ms"
                            )
                            return None
                    else:
                        self.logger.error(
                            f"[{self.agent_name}] message格式异常 | type={type(message).__name__}"
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
                        self.logger.error(
                            f"[{self.agent_name}] API错误 | "
                            f"code={error_info.get('code', '?') if isinstance(error_info, dict) else '?'} | "
                            f"message={error_message}"
                        )
                    else:
                        self.logger.error(
                            f"[{self.agent_name}] 响应无choices | "
                            f"keys={list(response_json.keys())} | "
                            f"body={str(response_json)[:300]}"
                        )
                    return None

            except requests.exceptions.Timeout:
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{self.agent_name}] TIMEOUT retry={attempt + 1}/{max_retries} | "
                        f"elapsed={elapsed_ms:.0f}ms (limit={self.timeout}s) | "
                        f"sleeping 1s before retry..."
                    )
                    time.sleep(1)
                    continue
                self.logger.warning(
                    f"[{self.agent_name}] TIMEOUT exhausted | "
                    f"total_attempts={max_retries + 1} | timeout={self.timeout}s | "
                    f"last_elapsed={elapsed_ms:.0f}ms"
                )
                return None
            except requests.exceptions.ConnectionError as e:
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000
                last_error = f"ConnectionError: {e}"
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{self.agent_name}] CONNECTION ERROR retry={attempt + 1}/{max_retries} | "
                        f"elapsed={elapsed_ms:.0f}ms | error={e}"
                    )
                    time.sleep(1)
                    continue
                self.logger.error(f"[{self.agent_name}] CONNECTION ERROR exhausted | error={e}")
                return None
            except requests.exceptions.RequestException as e:
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000
                last_error = f"RequestException: {e}"
                if attempt < max_retries:
                    self.logger.warning(
                        f"[{self.agent_name}] HTTP ERROR retry={attempt + 1}/{max_retries} | "
                        f"elapsed={elapsed_ms:.0f}ms | type={type(e).__name__} | error={e}"
                    )
                    time.sleep(1)
                    continue
                self.logger.error(
                    f"[{self.agent_name}] HTTP ERROR exhausted | "
                    f"type={type(e).__name__} | error={e}"
                )
                return None
            except Exception as e:
                elapsed_ms = (time.perf_counter() - t_attempt) * 1000
                self.logger.error(
                    f"[{self.agent_name}] UNKNOWN ERROR | "
                    f"elapsed={elapsed_ms:.0f}ms | type={type(e).__name__} | error={e}",
                    exc_info=True,
                )
                return None

        total_elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.logger.error(
            f"[{self.agent_name}] REQ FAILED | "
            f"total={total_elapsed_ms:.0f}ms | last_error={last_error}"
        )
        return None
