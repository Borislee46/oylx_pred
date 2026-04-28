import time
from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.context import StudentContext
from src.agent.lead_in_prompts import LEAD_IN_SYSTEM_PROMPT, build_lead_in_prompt
from src.agent.schemas import ExtractedBackground, LeadInResult


class LeadInAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=15, agent_name="前期LeadInAgent")
        self._system_prompt = LEAD_IN_SYSTEM_PROMPT

    def run(
        self,
        context: StudentContext,
        user_input: str | None = None,
    ) -> LeadInResult:
        t_start = time.perf_counter()
        text = (user_input or context.raw_input or "").strip()
        input_chars = len(text)
        turn_count = len(context.conversation_turns)
        has_existing = bool(context.extracted_background and any(
            v for v in context.extracted_background.values() if v
        ))
        self.logger.info(
            f"[{self.agent_name}] RUN START | input={input_chars}chars | "
            f"turn={turn_count} | has_existing_bg={has_existing}"
        )

        if not text:
            self.logger.info(f"[{self.agent_name}] RUN SKIP (empty input)")
            return {
                "extracted_info": {},
                "quick_assessment": "请提供学生信息，例如：学校、专业、GPA、目标国家/院校。",
                "suggested_questions": [
                    "学生目前就读于哪所大学？什么专业？",
                    "GPA大概在什么范围？是否有语言成绩？",
                    "学生想去哪个国家/地区？有目标的学校和专业吗？",
                ],
            }

        context.conversation_turns.append(
            {"role": "user", "content": text, "ts": time.time()}
        )

        prompt = build_lead_in_prompt(
            text, context.extracted_background, context.conversation_turns
        )
        full_prompt = f"{self._system_prompt}\n\n{prompt}"
        raw = self._call_api(full_prompt, cache_prefix="lead_in", max_tokens=300)

        result = self._parse_response(raw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if result is None:
            self.logger.warning(
                f"[{self.agent_name}] RUN FAILED | total={elapsed_ms:.0f}ms"
            )
            return {
                "extracted_info": {},
                "quick_assessment": "分析暂不可用，请稍后重试。",
                "suggested_questions": [],
                "_error": "api_failed",
            }

        context.raw_input = text
        _merge_extracted_background(context, result.get("extracted_info", {}))
        context.quick_assessment = result.get("quick_assessment", "")
        context.suggested_questions = result.get("suggested_questions", [])

        context.conversation_turns.append(
            {
                "role": "agent",
                "summary": context.quick_assessment[:120] if context.quick_assessment else "",
                "fields_filled": sum(
                    1 for v in context.extracted_background.values() if v
                ),
                "ts": time.time(),
            }
        )

        fields_filled = sum(
            1 for v in context.extracted_background.values() if v
        )
        self.logger.info(
            f"[{self.agent_name}] RUN OK | total={elapsed_ms:.0f}ms | "
            f"turn={turn_count}→{turn_count + 1} | "
            f"fields_extracted={fields_filled} | "
            f"assessment_len={len(context.quick_assessment)}chars | "
            f"questions={len(context.suggested_questions)}"
        )
        return result

    def _parse_response(self, raw: str | None) -> LeadInResult | None:
        return self._parse_json_response(
            raw,
            schema_hint=(
                '{"extracted_info": {...}, "quick_assessment": "...", '
                '"suggested_questions": ["...", "..."]}'
            ),
            cache_prefix="lead_in_json_repair",
            max_tokens=300,
        )


def _merge_extracted_background(
    context: StudentContext, new_info: ExtractedBackground
) -> None:
    """Merge new extracted info into existing background.

    Scalars overwrite on first non-null value only (preserving earlier
    extractions that are more likely to be correct). Lists are merged.
    Numeric fields are normalized to avoid float precision issues.
    """
    existing = context.extracted_background or {}
    mergeable_lists = {"target_schools", "target_majors"}
    numeric_fields = {"gpa": 2, "language_score": 1}

    for k, v in new_info.items():
        if v is None or v == "" or v == []:
            continue
        if k in existing and existing[k] and k not in mergeable_lists:
            continue  # preserve existing scalar values
        if k in mergeable_lists and k in existing and isinstance(existing[k], list):
            seen = set(existing[k])
            for item in (v if isinstance(v, list) else [v]):
                if str(item) not in seen:
                    existing[k].append(str(item))
                    seen.add(str(item))
        elif k in numeric_fields and isinstance(v, (int, float)):
            existing[k] = round(float(v), numeric_fields[k])
        else:
            existing[k] = v

    context.extracted_background = existing
