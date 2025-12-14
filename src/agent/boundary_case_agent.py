import hashlib
import json
import os
from typing import Any

import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.boundary_case_prompts import build_boundary_evaluation_prompt

CACHE_DIR = "cache/agent_cache"
CACHE_FILE = "boundary_case_decisions.json"
PROMPT_VERSION = 2


class BoundaryCaseAgent(BaseAgent):
    def __init__(self, cases_df: pd.DataFrame, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="边界CaseAgent")
        self.cases_df = cases_df
        self._persistent_cache = self._load_persistent_cache()

    def _case_cache_key(self, background_major: str, case: dict[str, Any], mode: str) -> str:
        university = str(case.get("university", "")).strip()
        major = str(case.get("major", "")).strip()
        key_data = {
            "v": PROMPT_VERSION,
            "background_major": background_major,
            "university": university,
            "major": major,
            "mode": mode,
            "model": self.model,
        }
        payload = json.dumps(key_data, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(payload.encode("utf-8")).hexdigest()

    def _load_persistent_cache(self) -> dict[str, Any]:
        file_path = os.path.join(CACHE_DIR, CACHE_FILE)
        if not os.path.exists(file_path):
            return {}

        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError) as e:
            self.logger.warning(f"[{self.agent_name}] 加载持久化缓存失败: {e}")
            return {}

    def _flush_persistent_cache(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        file_path = os.path.join(CACHE_DIR, CACHE_FILE)
        tmp_path = f"{file_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._persistent_cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, file_path)

    def evaluate_boundary_cases(
        self,
        background_major: str,
        boundary_cases: list[dict[str, Any]],
        mode: str,
        use_persistent_cache: bool = True,
    ) -> dict[str, Any]:
        fallback_result = {
            "decisions": [False] * len(boundary_cases) if boundary_cases else [],
            "needs_adjustment": False,
        }

        if not boundary_cases:
            self.logger.warning(f"[{self.agent_name}] 边界案例列表为空，跳过评估")
            return {"decisions": [], "needs_adjustment": False}

        decisions = [False] * len(boundary_cases)
        pending_cases: list[dict[str, Any]] = []
        pending_indices: list[int] = []
        new_cache_entries = 0

        for i, case in enumerate(boundary_cases):
            if not isinstance(case, dict):
                continue
            university = str(case.get("university", "")).strip()
            major = str(case.get("major", "")).strip()
            if not university or not major:
                continue

            if not use_persistent_cache:
                pending_cases.append(case)
                pending_indices.append(i)
                continue

            cache_key = self._case_cache_key(background_major, case, mode)
            cached = self._persistent_cache.get(cache_key)
            if isinstance(cached, bool):
                decisions[i] = cached
                continue
            if isinstance(cached, dict) and isinstance(cached.get("decision"), bool):
                decisions[i] = bool(cached["decision"])
                continue

            pending_cases.append(case)
            pending_indices.append(i)

        if pending_cases:
            self.logger.info(
                f"[{self.agent_name}] 开始评估 - 模式: {mode}, 背景专业: {background_major}, "
                f"待评估: {len(pending_cases)}, 缓存命中: {len(boundary_cases) - len(pending_cases)}"
            )
            prompt = build_boundary_evaluation_prompt(background_major, pending_cases, mode)
            self.logger.debug(
                f"[{self.agent_name}] 发送请求到 {self.api_url}, 模型: {self.model}, prompt长度: {len(prompt)}"
            )

            content = self._call_api(prompt)
            if content is None:
                return {"decisions": decisions, "needs_adjustment": any(decisions)}

            content = self._clean_json_content(content)
            try:
                result = json.loads(content)
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"[{self.agent_name}] JSON解析失败 - 错误: {e}, 响应内容长度: {len(content)}"
                )
                self.logger.debug(f"[{self.agent_name}] 响应内容预览: {content[:200]}")
                return {"decisions": decisions, "needs_adjustment": any(decisions)}

            agent_decisions = result.get("decisions", [])
            if len(agent_decisions) != len(pending_cases):
                self.logger.warning(
                    f"[{self.agent_name}] decisions 长度 ({len(agent_decisions)}) 与 boundary_cases 长度 "
                    f"({len(pending_cases)}) 不匹配"
                )
                agent_decisions = [False] * len(pending_cases)

            for j, idx in enumerate(pending_indices):
                d = bool(agent_decisions[j])
                decisions[idx] = d
                if use_persistent_cache:
                    cache_key = self._case_cache_key(background_major, boundary_cases[idx], mode)
                    self._persistent_cache[cache_key] = {"decision": d}
                    new_cache_entries += 1

            if use_persistent_cache and new_cache_entries:
                try:
                    self._flush_persistent_cache()
                except OSError as e:
                    self.logger.error(f"[{self.agent_name}] 保存持久化缓存失败: {e}")

        needs_adjustment = any(decisions)
        self.logger.info(
            f"[{self.agent_name}] 评估完成 - 需要调整: {needs_adjustment}, "
            f"决策数: {len(decisions)}, 通过数: {sum(decisions) if decisions else 0}"
        )
        return {"decisions": decisions, "needs_adjustment": needs_adjustment}
