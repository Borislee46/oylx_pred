import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.boundary_case_prompts import build_boundary_evaluation_prompt

CACHE_DIR = "cache/agent_cache"
CACHE_FILE = "boundary_case_decisions.json"


class BoundaryCaseAgent(BaseAgent):
    def __init__(self, cases_df: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        super().__init__(config=config, timeout=10, agent_name="边界CaseAgent")
        self.cases_df = cases_df

    def _get_persistent_cache_key(
        self, background_major: str, boundary_cases: List[Dict[str, Any]], mode: str
    ) -> str:
        key_data = {
            "background_major": background_major,
            "boundary_cases": boundary_cases,
            "mode": mode,
            "model": self.model,
        }
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _load_persistent_cache(self) -> Dict[str, Any]:
        if not os.path.exists(CACHE_DIR):
            return {}

        file_path = os.path.join(CACHE_DIR, CACHE_FILE)
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            self.logger.warning(f"[{self.agent_name}] 加载持久化缓存失败: {e}")
            return {}

    def _save_persistent_cache(self, key: str, value: Any) -> None:
        try:
            if not os.path.exists(CACHE_DIR):
                os.makedirs(CACHE_DIR, exist_ok=True)

            cache_data = self._load_persistent_cache()
            cache_data[key] = value

            file_path = os.path.join(CACHE_DIR, CACHE_FILE)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.error(f"[{self.agent_name}] 保存持久化缓存失败: {e}")

    def evaluate_boundary_cases(
        self, background_major: str, boundary_cases: List[Dict[str, Any]], mode: str
    ) -> Dict[str, Any]:
        fallback_result = {
            "decisions": [False] * len(boundary_cases) if boundary_cases else [],
            "needs_adjustment": False,
        }

        if not boundary_cases:
            self.logger.warning(f"[{self.agent_name}] 边界案例列表为空，跳过评估")
            return {"decisions": [], "needs_adjustment": False}

        cache_key = self._get_persistent_cache_key(background_major, boundary_cases, mode)
        cached_data = self._load_persistent_cache()
        if cache_key in cached_data:
            result = cached_data[cache_key]
            decisions = result.get("decisions", [])
            self.logger.info(
                f"[{self.agent_name}] 命中持久化缓存 - 模式: {mode}, "
                f"决策数: {len(decisions)}, 通过数: {sum(decisions) if decisions else 0}"
            )
            return result

        self.logger.info(
            f"[{self.agent_name}] 开始评估 - 模式: {mode}, 背景专业: {background_major}, "
            f"案例数量: {len(boundary_cases)}"
        )

        prompt = build_boundary_evaluation_prompt(background_major, boundary_cases, mode)

        self.logger.debug(
            f"[{self.agent_name}] 发送请求到 {self.api_url}, 模型: {self.model}, "
            f"prompt长度: {len(prompt)}"
        )

        content = self._call_api(prompt)
        if content is None:
            return fallback_result

        content = self._clean_json_content(content)

        try:
            result = json.loads(content)

            decisions = result.get("decisions", [])
            needs_adjustment = result.get("needs_adjustment", True)

            if len(decisions) != len(boundary_cases):
                self.logger.warning(
                    f"[{self.agent_name}] decisions 长度 ({len(decisions)}) 与 boundary_cases 长度 "
                    f"({len(boundary_cases)}) 不匹配"
                )
                decisions = [False] * len(boundary_cases)
                needs_adjustment = False

            result = {
                "decisions": decisions,
                "needs_adjustment": needs_adjustment,
            }

            self._save_persistent_cache(cache_key, result)

            self.logger.info(
                f"[{self.agent_name}] 评估完成 - 需要调整: {needs_adjustment}, "
                f"决策数: {len(decisions)}, 通过数: {sum(decisions) if decisions else 0}"
            )
            return result

        except json.JSONDecodeError as e:
            self.logger.error(
                f"[{self.agent_name}] JSON解析失败 - 错误: {e}, 响应内容长度: {len(content)}"
            )
            self.logger.debug(f"[{self.agent_name}] 响应内容预览: {content[:200]}")
            return fallback_result
