import hashlib
import json
from typing import Any

import pandas as pd

from src.agent.base_agent import BaseAgent
from src.agent.boundary_case_prompts import build_boundary_evaluation_prompt
from src.agent.utils import parse_bool

CACHE_DIR = "cache/agent_cache"
CACHE_FILE = "boundary_case_decisions.json"
PROMPT_VERSION = 2


class BoundaryCaseAgent(BaseAgent):
    def __init__(self, cases_df: pd.DataFrame, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=10, agent_name="边界CaseAgent")
        self.cases_df = cases_df
        self._persistent_cache = self._load_persistent_json(CACHE_DIR, CACHE_FILE)

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

    def _flush_persistent_cache(self) -> None:
        self._save_persistent_json(CACHE_DIR, CACHE_FILE, self._persistent_cache)

    def evaluate_boundary_cases(
        self,
        background_major: str,
        boundary_cases: list[dict[str, Any]],
        mode: str,
        use_persistent_cache: bool = True,
        chunk_size: int = 40,
    ) -> dict[str, Any]:
        if not boundary_cases:
            self.logger.warning(f"[{self.agent_name}] 边界案例列表为空，跳过评估")
            return {"decisions": [], "needs_adjustment": False, "evaluated": []}

        final_decisions = [False] * len(boundary_cases)
        evaluated = [False] * len(boundary_cases)

        pending_indices = []
        for i, case in enumerate(boundary_cases):
            if not isinstance(case, dict):
                continue

            if use_persistent_cache:
                cache_key = self._case_cache_key(background_major, case, mode)
                cached = self._persistent_cache.get(cache_key)
                if isinstance(cached, bool):
                    final_decisions[i] = cached
                    evaluated[i] = True
                    continue
                if isinstance(cached, dict) and isinstance(cached.get("decision"), bool):
                    final_decisions[i] = bool(cached["decision"])
                    evaluated[i] = True
                    continue

            pending_indices.append(i)

        if not pending_indices:
            return {
                "decisions": final_decisions,
                "needs_adjustment": any(final_decisions),
                "evaluated": evaluated,
            }

        self.logger.info(
            f"[{self.agent_name}] 开始评估 - 模式: {mode}, 背景专业: {background_major}, "
            f"总数: {len(boundary_cases)}, 待评估: {len(pending_indices)}, 缓存命中: {len(boundary_cases) - len(pending_indices)}"
        )

        new_cache_entries = 0
        for start_idx in range(0, len(pending_indices), chunk_size):
            current_batch_indices = pending_indices[start_idx : start_idx + chunk_size]
            current_batch_cases = [boundary_cases[idx] for idx in current_batch_indices]

            self.logger.debug(
                f"[{self.agent_name}] 正在处理分块: {start_idx // chunk_size + 1}, 大小: {len(current_batch_cases)}"
            )

            prompt = build_boundary_evaluation_prompt(background_major, current_batch_cases, mode)
            raw = self._call_api(prompt, max_tokens=256)
            if raw is None:
                continue

            result = self._parse_json_response(
                raw,
                schema_hint='{"decisions": [bool, ...], "needs_adjustment": bool}',
                cache_prefix="boundary_json_repair",
                max_tokens=256,
            )
            if result is None or not isinstance(result, dict):
                continue

            agent_decisions = result.get("decisions", [])
            if len(agent_decisions) != len(current_batch_cases):
                repaired = self._repair_json_once(
                    json.dumps(result, ensure_ascii=False),
                    schema_hint=f'{{"decisions": [bool, ... (len={len(current_batch_cases)})], "needs_adjustment": bool}}',
                    cache_prefix="boundary_json_repair_len",
                    max_tokens=256,
                )
                if repaired:
                    try:
                        fixed = json.loads(repaired)
                        if isinstance(fixed, dict) and isinstance(fixed.get("decisions"), list):
                            agent_decisions = fixed.get("decisions", [])
                    except json.JSONDecodeError:
                        pass

            for j, idx in enumerate(current_batch_indices):
                evaluated[idx] = True
                d = False
                if j < len(agent_decisions):
                    d = parse_bool(agent_decisions[j])

                final_decisions[idx] = d
                if use_persistent_cache:
                    cache_key = self._case_cache_key(background_major, boundary_cases[idx], mode)
                    self._persistent_cache[cache_key] = d
                    new_cache_entries += 1

        if use_persistent_cache and new_cache_entries > 0:
            try:
                self._flush_persistent_cache()
            except OSError as e:
                self.logger.error(f"[{self.agent_name}] 保存持久化缓存失败: {e}")

        needs_adjustment = any(final_decisions)
        self.logger.info(
            f"[{self.agent_name}] 评估完成 - 需要调整: {needs_adjustment}, 通过数: {sum(final_decisions)}"
        )
        return {
            "decisions": final_decisions,
            "needs_adjustment": needs_adjustment,
            "evaluated": evaluated,
        }

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        """Orchestrator-compatible entry point.

        Expects kwargs:
            background_major: str
            boundary_cases: list[dict]
            mode: str ("similarity" | "cross_major")
            use_persistent_cache: bool = True
            chunk_size: int = 40
        """
        from src.agent.context import StudentContext

        bg_major = kwargs.get("background_major", "")
        if not bg_major and isinstance(context, StudentContext):
            bg_major = context.background_major or context.extracted_background.get("major", "")

        return self.evaluate_boundary_cases(
            background_major=str(bg_major),
            boundary_cases=kwargs.get("boundary_cases", []),
            mode=str(kwargs.get("mode", "similarity")),
            use_persistent_cache=bool(kwargs.get("use_persistent_cache", True)),
            chunk_size=int(kwargs.get("chunk_size", 40)),
        )
