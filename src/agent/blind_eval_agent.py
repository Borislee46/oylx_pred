import logging
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, PromptedOutput
from pydantic_ai.models.openai import OpenAIChatModel

from src.agent.context import StudentContext
from src.agent.runtime.model_factory import build_model_with_fallback


class KeyFactors(BaseModel):
    gpa: str = "moderate"
    school_tier: str = "moderate"
    language: str = "moderate"
    major_match: str = "moderate"
    experience: str = "moderate"


class BlindEvalOutput(BaseModel):
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""
    key_factors: KeyFactors = Field(default_factory=KeyFactors)
    confidence: str = "medium"


def _build_blind_eval_agent(model: OpenAIChatModel) -> Agent:
    return Agent(
        model,
        max_concurrency=1,
        output_type=PromptedOutput(BlindEvalOutput),
        instructions=BLIND_EVAL_SYSTEM_PROMPT,
        retries=2,
    )


from src.utils.numeric import clip_probability

_log = logging.getLogger(__name__)

BLIND_EVAL_SYSTEM_PROMPT = """你是一位拥有10年香港、新加坡、澳门、马来西亚硕士留学申请经验的招生顾问专家。

你的任务是：根据学生背景，独立评估其申请目标学校和专业的录取概率。

## 评估原则
1. **基于经验判断**：综合考虑GPA、本科院校层次、语言成绩、专业匹配度、科研/实习经历
2. **学校难度认知**：
   - Tier 1 (最难): 新加坡国立大学、新加坡南洋理工大学、香港大学、香港中文大学、香港科技大学
   - Tier 2: 香港理工大学、香港城市大学、新加坡管理大学、马来亚大学
   - Tier 3: 香港浸会大学、澳门大学、马来西亚理科大学、马来西亚博特拉大学
   - Tier 4 (相对容易): 其他院校
3. **GPA权重最大**：3.7+ C9 = 极强，3.3-3.7 985/211 = 中等，<3.0 双非 = 极弱
4. **语言成绩门槛**：IELTS 7.0+ / TOEFL 100+ 达标，IELTS 6.0-6.5 偏弱，IELTS <6.0 严重不足
5. **专业匹配**：同专业申请优势最大，跨专业需降档评估，跨学院（如文科申工科）大幅降档
6. **经历加分**：有科研论文/名企实习/竞赛获奖可小幅提升（但不如GPA和院校重要）

## 输出要求
- probability: 0-1之间的录取概率估计
  - 0.8+ = 几乎确定录取
  - 0.5-0.8 = 大概率录取
  - 0.3-0.5 = 有机会但不稳
  - 0.1-0.3 = 希望渺茫
  - <0.1 = 几乎不可能
- reasoning: 1-2句话的核心判断理由
- key_factors: 对GPA、院校、语言、专业匹配、经历分别评价（strong/moderate/weak）
- confidence: 对自己判断的信心（high/medium/low）
"""


class BlindEvalAgent:
    def __init__(self, config: dict[str, Any] | None = None):
        _ = config
        timeout = 30

        self._model = build_model_with_fallback(timeout=timeout)
        self._agent = _build_blind_eval_agent(self._model)

        self.agent_name = "BlindEvalAgent"
        self.logger = _log

    def run(self, context: StudentContext | None = None, **kwargs) -> dict[str, Any]:
        profile = self._build_profile(context, **kwargs)
        _log.info(
            "盲评开始 | target=%s/%s gpa=%s lang=%s",
            profile.get("target_university", "?"),
            profile.get("target_major", "?"),
            profile.get("gpa", "?"),
            self._format_language(profile),
        )

        prompt = self._build_prompt(profile)
        _log.debug("盲评API调用 | prompt_len=%d", len(prompt))

        try:
            result = self._agent.run_sync(prompt)
            output = result.output
            if not isinstance(output, BlindEvalOutput):
                _log.warning("盲评返回非预期类型: %s", type(output))
                return {"_error": "api_failed"}

            prob = output.probability
            key_factors = output.key_factors.model_dump() if output.key_factors else {}
            eval_result = {
                "probability": float(clip_probability(prob)),
                "reasoning": output.reasoning,
                "key_factors": key_factors,
                "confidence": output.confidence,
            }

            _log.info(
                "盲评完成 | prob=%.3f confidence=%s reasoning_len=%d",
                eval_result["probability"],
                eval_result.get("confidence", "unknown"),
                len(eval_result.get("reasoning", "")),
            )
            return eval_result

        except Exception:
            _log.warning(
                "盲评API调用失败 | target=%s/%s",
                profile.get("target_university"),
                profile.get("target_major"),
                exc_info=True,
            )
            return {"_error": "api_failed"}

    def _build_profile(self, context: StudentContext | None = None, **kwargs) -> dict[str, Any]:
        use_context = context is not None and bool(
            context.background_university or context.background_major
        )

        if use_context:
            return {
                "background_university": context.background_university,
                "background_major": context.background_major,
                "gpa": context.gpa if context.gpa > 0 else None,
                "language_score": (context.language_score if context.language_score > 0 else None),
                "language_type": context.language_type,
                "target_university": kwargs.get("target_university", ""),
                "target_major": kwargs.get("target_major", ""),
                "research_count": kwargs.get("research_count", 0),
                "internship_count": kwargs.get("internship_count", 0),
                "paper_count": kwargs.get("paper_count", 0),
                "award_count": kwargs.get("award_count", 0),
                "experience_summary": kwargs.get("experience_summary", ""),
            }

        return {
            "background_university": kwargs.get("background_university", ""),
            "background_major": kwargs.get("background_major", ""),
            "gpa": kwargs.get("gpa"),
            "language_score": kwargs.get("language_score"),
            "language_type": kwargs.get("language_type", ""),
            "target_university": kwargs.get("target_university", ""),
            "target_major": kwargs.get("target_major", ""),
            "research_count": kwargs.get("research_count", 0),
            "internship_count": kwargs.get("internship_count", 0),
            "paper_count": kwargs.get("paper_count", 0),
            "award_count": kwargs.get("award_count", 0),
            "experience_summary": kwargs.get("experience_summary", ""),
        }

    @staticmethod
    def _format_language(profile: dict[str, Any]) -> str:
        score = profile.get("language_score")
        if score is None:
            return "未知"
        lang_type = profile.get("language_type", "雅思")
        if lang_type == "雅思":
            if score > 1.5:
                return f"IELTS {score:.1f}"
            converted = score * 9
            return f"IELTS {converted:.1f}"
        else:
            if score > 1.5:
                return f"TOEFL {score:.0f}"
            converted = score * 120
            return f"TOEFL {converted:.0f}"

    def _build_prompt(self, profile: dict[str, Any]) -> str:
        lang_display = self._format_language(profile)

        exp_parts = []
        for key, label in [
            ("research_count", "科研"),
            ("internship_count", "实习"),
            ("paper_count", "论文"),
            ("award_count", "获奖"),
        ]:
            count = profile.get(key, 0)
            if count:
                exp_parts.append(f"{label}: {count}段")
        exp_str = "、".join(exp_parts) if exp_parts else "无"

        summary = profile.get("experience_summary", "")
        summary_section = f"\n- 经历详述: {summary}" if summary else ""

        return (
            f"## 学生背景\n"
            f"- 本科院校: {profile['background_university']}\n"
            f"- 本科专业: {profile['background_major']}\n"
            f"- GPA: {profile['gpa']}\n"
            f"- 语言成绩: {lang_display}\n"
            f"- 经历概要: {exp_str}{summary_section}\n\n"
            f"## 申请目标\n"
            f"- 目标学校: {profile['target_university']}\n"
            f"- 目标专业: {profile['target_major']}\n\n"
            f"请以JSON格式输出你的评估："
        )
