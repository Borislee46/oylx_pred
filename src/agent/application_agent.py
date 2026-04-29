import time
from typing import Any

from src.agent.application_prompts import (
    APPLICATION_SYSTEM_PROMPT,
    build_application_prompt,
)
from src.agent.base_agent import BaseAgent
from src.agent.context import StudentContext


class ApplicationAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=18, agent_name="申请策略Agent")
        self._system_prompt = APPLICATION_SYSTEM_PROMPT

    def run(self, context: StudentContext) -> dict[str, Any]:
        t_start = time.perf_counter()
        results = context.prediction_results or {}
        unified = results.get("unified_results") or []

        if not unified:
            self.logger.info(f"[{self.agent_name}] RUN SKIP (no results)")
            return {"strategy_overview": "暂无预测结果，无法生成申请策略。", "_error": "no_results"}

        self.logger.info(
            f"[{self.agent_name}] RUN START | results={len(unified)} | gpa={context.gpa}"
        )

        bg = context.extracted_background or {}
        prompt = build_application_prompt(
            gpa=context.gpa or 0,
            language_type=context.language_type or bg.get("language_type", ""),
            language_score=context.language_score or bg.get("language_score", 0),
            background_university=context.background_university or bg.get("university", ""),
            background_major=context.background_major or bg.get("major", ""),
            target_country=context.target_country or bg.get("country", ""),
            experience_details=context.experience_details,
            unified_results=unified,
        )
        full_prompt = f"{self._system_prompt}\n\n{prompt}"
        raw = self._call_api(full_prompt, cache_prefix="application", max_tokens=800)

        result = self._parse_response(raw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if result is None:
            self.logger.warning(f"[{self.agent_name}] RUN FAILED | total={elapsed_ms:.0f}ms")
            return {"strategy_overview": "申请策略暂不可用", "_error": "api_failed"}

        context.application_plan = result
        self.logger.info(
            f"[{self.agent_name}] RUN OK | total={elapsed_ms:.0f}ms | "
            f"overview_len={len(result.get('strategy_overview', ''))}chars | "
            f"timeline_items={len(result.get('timeline', []))} | "
            f"checklist_items={len(result.get('material_checklist', []))}"
        )
        return result

    def _parse_response(self, raw: str | None) -> dict[str, Any] | None:
        return self._parse_json_response(
            raw,
            schema_hint=(
                '{"strategy_overview": "...", '
                '"tier_recommendations": {"reach": [...], "match": [...], "safety": [...]}, '
                '"timeline": ["...", ...], '
                '"material_checklist": [{"item":"","importance":"高|中|低","note":""}], '
                '"risk_assessment": [{"risk":"","mitigation":""}]}'
            ),
            cache_prefix="application_json_repair",
            max_tokens=800,
        )
