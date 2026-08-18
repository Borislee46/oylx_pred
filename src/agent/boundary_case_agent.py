import logging
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.base_agent import BaseAgent
from src.agent.persistent_cache import PersistentCache
from src.agent.prompts.boundary_case import build_boundary_evaluation_prompt


class BoundaryCaseDecisions(BaseModel):
    decisions: list[bool] = Field(default_factory=list)
    needs_adjustment: bool = False


_BOUNDARY_SYSTEM_PROMPT = """你是一位留学申请专业匹配度评估专家。
你的任务是判断目标专业与学生的本科背景专业是否相似/相关。

评估标准：
- 相似判定：学科领域相同、核心课程重叠度高，或在该本科背景下跨专业申请属于常规路径
- 根据给定的模式（放宽/收紧）做出二元判断
- 对每个待评专业返回 true 或 false
- needs_adjustment：当至少有一个待评专业需要调整时为 true

只返回 JSON，不要添加解释。"""


def _build_boundary_case_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(BoundaryCaseDecisions),
        instructions=_BOUNDARY_SYSTEM_PROMPT,
        retries=2,
    )


from src.agent.utils import parse_bool

_log = logging.getLogger(__name__)

PROMPT_VERSION = 3


class BoundaryCaseAgent(BaseAgent):
    _cache = PersistentCache("boundary_case_decisions.json")

    def __init__(
        self,
        cases_df: pd.DataFrame | None = None,
        config: dict[str, Any] | None = None,
    ):
        super().__init__(config, timeout=10, agent_name="边界CaseAgent", logger=_log)
        self._agent = _build_boundary_case_agent(self._model)
        self.cases_df = cases_df

    def _case_cache_key(self, background_major: str, case: dict[str, Any], mode: str) -> str:
        return self._hash_cache_key(
            PROMPT_VERSION,
            self.model,
            background_major=str(background_major or "").strip(),
            university=str(case.get("university", "")).strip(),
            major=str(case.get("major", "")).strip(),
            mode=str(mode or "").strip(),
        )

    def evaluate_boundary_cases(
        self,
        background_major: str,
        boundary_cases: list[dict[str, Any]],
        mode: str,
        use_persistent_cache: bool = True,
        chunk_size: int = 40,
    ) -> dict[str, Any]:
        if not boundary_cases:
            _log.info("边界案例列表为空，跳过评估")
            return {
                "decisions": [],
                "needs_adjustment": False,
                "evaluated": [],
                "api_errors": 0,
            }

        if mode not in ("relax", "tighten"):
            mode = "tighten"

        final_decisions = [False] * len(boundary_cases)
        evaluated = [False] * len(boundary_cases)

        cache_hits = 0
        pending_indices = []
        for i, case in enumerate(boundary_cases):
            if not isinstance(case, dict):
                continue

            if use_persistent_cache:
                cache_key = self._case_cache_key(background_major, case, mode)
                cached = BoundaryCaseAgent._cache.get(cache_key)
                if isinstance(cached, bool):
                    final_decisions[i] = cached
                    evaluated[i] = True
                    cache_hits += 1
                    continue
                if isinstance(cached, dict) and isinstance(cached.get("decision"), bool):
                    final_decisions[i] = bool(cached["decision"])
                    evaluated[i] = True
                    cache_hits += 1
                    continue

            pending_indices.append(i)

        if not pending_indices:
            _log.info(
                "边界案例全缓存命中 | mode=%s bg_major=%s total=%d hits=%d",
                mode,
                background_major,
                len(boundary_cases),
                cache_hits,
            )
            return {
                "decisions": final_decisions,
                "needs_adjustment": any(final_decisions),
                "evaluated": evaluated,
                "api_errors": 0,
            }

        total_chunks = (len(pending_indices) + chunk_size - 1) // chunk_size
        _log.info(
            "边界案例评估开始 | mode=%s bg_major=%s total=%d pending=%d cache_hits=%d chunks=%d",
            mode,
            background_major,
            len(boundary_cases),
            len(pending_indices),
            cache_hits,
            total_chunks,
        )

        new_cache_entries = 0
        api_errors = 0
        for start_idx in range(0, len(pending_indices), chunk_size):
            chunk_num = start_idx // chunk_size + 1
            batch_indices = pending_indices[start_idx : start_idx + chunk_size]
            batch_cases = [boundary_cases[idx] for idx in batch_indices]

            _log.debug(
                "分块处理开始 | chunk=%d/%d size=%d",
                chunk_num,
                total_chunks,
                len(batch_cases),
            )

            prompt = build_boundary_evaluation_prompt(
                background_major,
                batch_cases,
                mode,
            )
            _log.debug("分块API调用 | chunk=%d prompt_len=%d", chunk_num, len(prompt))

            try:
                result = self._agent.run_sync(prompt)
                output = result.output
                if not isinstance(output, BoundaryCaseDecisions):
                    api_errors += 1
                    _log.warning("分块返回非预期类型 | chunk=%d", chunk_num)
                    continue
                agent_decisions = output.decisions
            except Exception:
                api_errors += 1
                _log.warning("分块API调用失败 | chunk=%d", chunk_num, exc_info=True)
                continue

            expected = len(batch_cases)
            actual = len(agent_decisions)
            if actual != expected:
                _log.warning(
                    "决策数量不匹配 | chunk=%d expected=%d actual=%d",
                    chunk_num,
                    expected,
                    actual,
                )
                if actual < expected:
                    agent_decisions = list(agent_decisions) + [False] * (expected - actual)
                else:
                    agent_decisions = list(agent_decisions)[:expected]

            chunk_tighten = 0
            chunk_relax = 0
            chunk_cache_updates: dict[str, bool] = {}
            for j, idx in enumerate(batch_indices):
                evaluated[idx] = True
                d = parse_bool(agent_decisions[j]) if j < len(agent_decisions) else False
                final_decisions[idx] = d
                if d:
                    chunk_tighten += 1
                else:
                    chunk_relax += 1

                if use_persistent_cache:
                    cache_key = self._case_cache_key(
                        background_major,
                        boundary_cases[idx],
                        mode,
                    )
                    chunk_cache_updates[cache_key] = d

            if chunk_cache_updates:
                BoundaryCaseAgent._cache.set_many(chunk_cache_updates)
                new_cache_entries += len(chunk_cache_updates)

            _log.info(
                "分块处理完成 | chunk=%d/%d tighten=%d relax=%d",
                chunk_num,
                total_chunks,
                chunk_tighten,
                chunk_relax,
            )

        tighten_count = sum(final_decisions)
        relax_count = len(final_decisions) - tighten_count
        needs_adjustment = tighten_count > 0
        _log.info(
            "边界案例评估完成 | needs_adjustment=%s tighten=%d relax=%d api_errors=%d new_cache=%d",
            needs_adjustment,
            tighten_count,
            relax_count,
            api_errors,
            new_cache_entries,
        )
        return {
            "decisions": final_decisions,
            "needs_adjustment": needs_adjustment,
            "evaluated": evaluated,
            "api_errors": api_errors,
        }

    def run(self, context: Any = None, **kwargs: Any) -> dict[str, Any]:
        from src.agent.context import StudentContext

        bg_major = kwargs.get("background_major", "")
        if not bg_major and isinstance(context, StudentContext):
            bg_major = context.background_major or context.extracted_background.get("major", "")

        mode = str(kwargs.get("mode", "tighten"))
        if mode not in ("relax", "tighten"):
            mode = "tighten"

        return self.evaluate_boundary_cases(
            background_major=str(bg_major),
            boundary_cases=kwargs.get("boundary_cases", []),
            mode=mode,
            use_persistent_cache=bool(kwargs.get("use_persistent_cache", True)),
            chunk_size=int(kwargs.get("chunk_size", 40)),
        )
