import time
from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.context import StudentContext

EXPLAIN_SYSTEM_PROMPT = """\
你是留学选校分析专家。根据预测结果和学生背景，生成专业易懂的中文解读。

结果中可能包含调整标记（_adjustment_trace），说明概率被调整的原因：
- "Cross Major Penalty"：跨专业惩罚（相似度不足）
- "Faculty Out of Scope Penalty"：跨学部惩罚（学部跨度大）
- "Professional Major Penalty"：专业项目缺实习背景
- "Language Penalty"：语言成绩未达要求
- "Text Boost"：经历文本质量提升

tier说明：高概率(≥60%)=保底，中等(30-60%)=适中，低(<30%)=冲刺。

解读要点：
1. overview：综合评估学生背景水平和整体申请竞争力（80-120字）
2. recommendations：选3-5个最优推荐，每个说明推荐理由
3. strengths：学生的竞争优势（2-4条）
4. concerns：需要关注的风险点（2-4条），如涉及跨专业/跨学部/语言等调整必须提及
5. summary：40-60字总结建议

严格输出JSON，无其他内容：
{"overview":"...","recommendations":[{"school":"","major":"","probability":0,"tier":"冲刺|适中|保底","reason":""}],"strengths":[""],"concerns":[""],"summary":"..."}
"""


def _fmt_prob(p: float) -> str:
    return f"{p:.0%}" if p else "0%"


def _fmt_trace(trace: list | None) -> str:
    if not trace:
        return ""
    parts = []
    for t in trace:
        if isinstance(t, str):
            parts.append(t)
            continue
        if not isinstance(t, dict):
            continue
        name = t.get("name", "")
        desc = t.get("description", "")
        ftype = t.get("factor_type", "")
        label = "↓" if ftype == "penalty" else "↑"
        if desc:
            parts.append(f"{label}{name}({desc})")
        else:
            parts.append(f"{label}{name}")
    return " ".join(parts)


def _build_explain_prompt(ctx: StudentContext) -> str:
    bg = ctx.extracted_background or {}
    lines = [
        "## 学生背景",
        f"- 院校：{ctx.background_university or bg.get('university', '未知')}",
        f"- 专业：{ctx.background_major or bg.get('major', '未知')}",
        f"- GPA：{ctx.gpa or bg.get('gpa', '未知')}",
        f"- 语言：{ctx.language_type or bg.get('language_type', '')} "
        f"{ctx.language_score or bg.get('language_score', '')}",
    ]

    exp = ctx.experience_details
    if exp:
        parts = []
        for k in ("research", "internship", "award", "paper"):
            v = exp.get(k)
            if v:
                parts.append(str(v))
        if parts:
            lines.append(f"- 经历：{'; '.join(parts)}")

    results = ctx.prediction_results or {}
    unified = results.get("unified_results") or []
    sim = results.get("similarity_results") or []
    cross = results.get("cross_major_results") or []

    lines.append(f"\n## 预测结果（共 {len(unified)} 条推荐）")

    def _fmt_results(label: str, items: list) -> None:
        if not items:
            return
        lines.append(f"\n### {label}")
        for i, r in enumerate(items[:6]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = r.get("probability", 0)
            sim_score = r.get("similarity", 0)
            trace_str = _fmt_trace(r.get("_adjustment_trace"))
            lang_adj = " [语言惩罚]" if r.get("language_penalty_applied") else ""
            line = f"{i + 1}. {uni} {maj}：{_fmt_prob(prob)} | 相似度{sim_score:.2f}"
            if trace_str:
                line += f" | 调整: {trace_str}"
            if lang_adj:
                line += lang_adj
            lines.append(line)

    _fmt_results("相似专业", sim)
    _fmt_results("跨专业", cross)

    if unified:
        lines.append(f"\n### 综合排序（前6）")
        for i, r in enumerate(unified[:6]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = r.get("probability", 0)
            trace_str = _fmt_trace(r.get("_adjustment_trace"))
            line = f"{i + 1}. {uni} {maj}：{_fmt_prob(prob)}"
            if trace_str:
                line += f" | {trace_str}"
            lines.append(line)

    return "\n".join(lines)


class ExplainAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=15, agent_name="解释Agent")
        self._system_prompt = EXPLAIN_SYSTEM_PROMPT
        self._stream_buffer = ""

    def stream(self, context: StudentContext):
        """Stream explanation text chunks via st.write_stream.

        Usage:
            agent = AgentRegistry.get("explain")
            raw = st.write_stream(agent.stream(ctx))
            result = agent._parse_response(raw)
        """
        t_start = time.perf_counter()
        pred = context.prediction_results or {}
        self.logger.info(
            f"[{self.agent_name}] STREAM START | "
            f"results: sim={len(pred.get('similarity_results') or [])} "
            f"cross={len(pred.get('cross_major_results') or [])}"
        )

        prompt = _build_explain_prompt(context)
        full_prompt = f"{self._system_prompt}\n\n{prompt}"
        self._stream_buffer = ""

        for chunk in self._call_api_streaming(
            full_prompt, max_tokens=600
        ):
            if chunk:
                self._stream_buffer += chunk
                yield chunk

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self.logger.info(
            f"[{self.agent_name}] STREAM DONE | total={elapsed_ms:.0f}ms | "
            f"output={len(self._stream_buffer)}chars"
        )

    def parse_stream_result(self) -> dict[str, Any] | None:
        return self._parse_response(self._stream_buffer)

    def run(self, context: StudentContext) -> dict[str, Any]:
        t_start = time.perf_counter()
        pred = context.prediction_results or {}
        n_sim = len(pred.get("similarity_results") or [])
        n_cross = len(pred.get("cross_major_results") or [])
        n_unified = len(pred.get("unified_results") or [])
        self.logger.info(
            f"[{self.agent_name}] RUN START | "
            f"results: sim={n_sim} cross={n_cross} unified={n_unified} | "
            f"gpa={context.gpa} lang={context.language_type} {context.language_score}"
        )

        prompt = _build_explain_prompt(context)
        full_prompt = f"{self._system_prompt}\n\n{prompt}"
        raw = self._call_api(full_prompt, cache_prefix="explain", max_tokens=600)

        result = self._parse_response(raw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if result is None:
            self.logger.warning(
                f"[{self.agent_name}] RUN FAILED | total={elapsed_ms:.0f}ms"
            )
            return {"overview": "解释暂不可用", "_error": "api_failed"}

        context.ai_explanation = result.get("overview", "")
        self.logger.info(
            f"[{self.agent_name}] RUN OK | total={elapsed_ms:.0f}ms | "
            f"overview_len={len(result.get('overview', ''))}chars | "
            f"strengths={len(result.get('strengths', []))} | "
            f"concerns={len(result.get('concerns', []))} | "
            f"recommendations={len(result.get('recommendations', []))}"
        )
        return result

    def _parse_response(self, raw: str | None) -> dict[str, Any] | None:
        return self._parse_json_response(
            raw,
            schema_hint=(
                '{"overview": "...", "recommendations": [...], '
                '"strengths": [...], "concerns": [...], "summary": "..."}'
            ),
            cache_prefix="explain_json_repair",
            max_tokens=600,
        )
