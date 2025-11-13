import hashlib
import json
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests

from src.agent.base_agent import BaseAgent
from src.agent.single_pred_shap_prompt import build_shap_explanation_prompt
from src.pages.prediction.page_components.pdf_generation.utils import pdf_cache


class SinglePredShapAgent(BaseAgent):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, timeout=30, agent_name="SHAP解释Agent", cache_ttl=86400)

    def _normalize_raw_feature_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, np.integer, np.floating)):
            return round(float(value), 6)
        elif isinstance(value, str):
            return value.strip()
        elif value is None:
            return ""
        else:
            str_val = str(value).strip()
            if str_val.lower() in ("nan", "none", ""):
                return ""
            return str_val

    def _generate_shap_cache_key(
        self,
        target_university: str,
        target_major: str,
        background_major: str,
        feature_names: List[str],
        shap_values: np.ndarray,
        raw_feature_values: np.ndarray,
        expected_value: float,
    ) -> str:
        shap_values_normalized = (
            np.round(shap_values, decimals=6).tolist()
            if isinstance(shap_values, np.ndarray)
            else [round(float(v), 6) for v in shap_values]
        )
        expected_value_normalized = round(float(expected_value), 6)

        raw_feature_values_normalized = []
        if isinstance(raw_feature_values, np.ndarray):
            for val in raw_feature_values:
                raw_feature_values_normalized.append(self._normalize_raw_feature_value(val))
        else:
            raw_feature_values_normalized = [
                self._normalize_raw_feature_value(val) for val in raw_feature_values
            ]

        key_data = {
            "target_university": target_university,
            "target_major": target_major,
            "background_major": background_major,
            "feature_names": feature_names,
            "shap_values": shap_values_normalized,
            "raw_feature_values": raw_feature_values_normalized,
            "expected_value": expected_value_normalized,
        }
        key_str = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        cache_key_hash = hashlib.md5(key_str.encode("utf-8")).hexdigest()

        shap_range_str = "N/A"
        if shap_values_normalized:
            shap_range_str = (
                f"[{min(shap_values_normalized):.6f}, {max(shap_values_normalized):.6f}]"
            )

        self.logger.debug(
            f"[{self.agent_name}] 缓存键生成 - "
            f"特征数: {len(feature_names)}, "
            f"SHAP值范围: {shap_range_str}, "
            f"期望值: {expected_value_normalized:.6f}, "
            f"缓存键哈希: {cache_key_hash[:16]}..."
        )

        return cache_key_hash

    def _build_stream_request_data(self, prompt: str) -> Dict[str, Any]:
        return {
            "model": self.model,
            "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
            "thinking": {"type": "disabled"},
            "stream": True,
        }

    def _call_api_stream(self, prompt: str) -> Iterator[str]:
        if not self.api_url or not self.api_key:
            self.logger.warning(f"[{self.agent_name}] API 未配置，无法调用")
            return

        data = self._build_stream_request_data(prompt)

        try:
            response = requests.post(
                self.api_url, headers=self.headers, json=data, timeout=self.timeout, stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if not line:
                    continue

                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break

                    try:
                        chunk_data = json.loads(data_str)
                        choices = chunk_data.get("choices", [])
                        if choices and len(choices) > 0:
                            delta = choices[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue

        except requests.exceptions.Timeout:
            self.logger.warning(f"[{self.agent_name}] 请求超时（{self.timeout}秒）")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"[{self.agent_name}] 网络请求失败: {e}")
        except Exception as e:
            self.logger.error(
                f"[{self.agent_name}] 未知错误 - 错误类型: {type(e).__name__}, 错误信息: {e}",
                exc_info=True,
            )

    def _validate_shap_inputs(
        self,
        target_university: str,
        target_major: str,
        feature_names: List[str],
        shap_values: np.ndarray,
        raw_feature_values: np.ndarray,
    ) -> bool:
        if not target_university or not target_major:
            self.logger.warning(f"[{self.agent_name}] 目标院校或专业为空")
            return False

        if len(feature_names) != len(shap_values) or len(feature_names) != len(raw_feature_values):
            self.logger.warning(
                f"[{self.agent_name}] 特征数量不匹配: "
                f"feature_names={len(feature_names)}, "
                f"shap_values={len(shap_values)}, "
                f"raw_feature_values={len(raw_feature_values)}"
            )
            return False
        return True

    def _build_shap_prompt_and_cache_key(
        self,
        target_university: str,
        target_major: str,
        background_major: str,
        feature_names: List[str],
        shap_values: np.ndarray,
        raw_feature_values: np.ndarray,
        expected_value: float,
    ) -> Tuple[str, str]:
        prompt = build_shap_explanation_prompt(
            target_university=target_university,
            target_major=target_major,
            background_major=background_major,
            feature_names=feature_names,
            shap_values=shap_values,
            raw_feature_values=raw_feature_values,
        )
        cache_key = f"shap_explanation:{self._generate_shap_cache_key(target_university, target_major, background_major, feature_names, shap_values, raw_feature_values, expected_value)}"
        self.logger.debug(f"[{self.agent_name}] 缓存键: {cache_key[:50]}...")
        self.logger.debug(f"[{self.agent_name}] Prompt 预览（前200字符）: {prompt[:200]}...")
        return prompt, cache_key

    def explain_shap_values(
        self,
        target_university: str,
        target_major: str,
        background_major: str,
        feature_names: List[str],
        shap_values: np.ndarray,
        raw_feature_values: np.ndarray,
        expected_value: float,
    ) -> Optional[str]:
        if not self._validate_shap_inputs(
            target_university, target_major, feature_names, shap_values, raw_feature_values
        ):
            return None

        prompt, cache_key = self._build_shap_prompt_and_cache_key(
            target_university,
            target_major,
            background_major,
            feature_names,
            shap_values,
            raw_feature_values,
            expected_value,
        )

        start_time = time.time()
        explanation = self._call_api(prompt, custom_cache_key=cache_key)
        elapsed_time = time.time() - start_time

        if explanation:
            self.logger.info(f"[{self.agent_name}] SHAP解释生成成功，耗时: {elapsed_time:.2f}秒")
            return explanation.strip()
        else:
            self.logger.warning(
                f"[{self.agent_name}] API调用失败，返回空解释，耗时: {elapsed_time:.2f}秒"
            )
            return None

    def explain_shap_values_stream(
        self,
        target_university: str,
        target_major: str,
        background_major: str,
        feature_names: List[str],
        shap_values: np.ndarray,
        raw_feature_values: np.ndarray,
        expected_value: float,
    ) -> Iterator[str]:
        if not self._validate_shap_inputs(
            target_university, target_major, feature_names, shap_values, raw_feature_values
        ):
            return

        prompt, cache_key = self._build_shap_prompt_and_cache_key(
            target_university,
            target_major,
            background_major,
            feature_names,
            shap_values,
            raw_feature_values,
            expected_value,
        )

        try:
            pdf_cache.clear_expired()
            cached_data = pdf_cache.get(cache_key)

            if cached_data and isinstance(cached_data, dict):
                cached_value = cached_data.get("value")
                if cached_value:
                    self.logger.info(
                        f"[{self.agent_name}] 使用缓存的SHAP解释（流式返回），长度: {len(cached_value)}"
                    )
                    for char in cached_value:
                        yield char
                    return
        except Exception as e:
            self.logger.warning(f"[{self.agent_name}] 缓存操作失败: {e}，继续调用API")

        self.logger.info(
            f"[{self.agent_name}] 缓存未命中，开始流式生成SHAP解释 - "
            f"目标: {target_university} {target_major}，缓存键: {cache_key[:30]}..."
        )

        collected_chunks: List[str] = []
        for chunk in self._call_api_stream(prompt):
            collected_chunks.append(chunk)
            yield chunk

        if collected_chunks:
            result = "".join(collected_chunks).strip()
            if result:
                try:
                    ttl = self.cache_ttl if self.cache_ttl is not None else 3600
                    pdf_cache.set(cache_key, result, ttl)
                    self.logger.info(f"[{self.agent_name}] 已缓存流式生成的SHAP解释，TTL: {ttl}秒")
                except Exception as e:
                    self.logger.warning(f"[{self.agent_name}] 缓存写入失败: {e}，但不影响返回结果")
