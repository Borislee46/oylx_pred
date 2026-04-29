import time
from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.context import StudentContext

EXPLAIN_SYSTEM_PROMPT = """\
你是一位资深留学顾问，正为客户一对一解读选校预测结果。语气专业但有温度，像和客户面对面交谈，不要像机器报告。

结果中可能出现调整标记（_adjustment_trace），含义如下：
- Cross Major Penalty：跨专业申请，专业匹配度不足
- Faculty Out of Scope Penalty：跨学部申请，学科跨度较大
- Professional Major Penalty：职业导向项目缺少对应实习
- Language Penalty：语言成绩低于项目常规要求
- Text Boost：经历描述质量较好，正向调整

写作要点：
1. overview：先肯定客户背景亮点，再客观指出短板。80-120字，口语化但不随意。
2. strengths：客户真正的竞争优势（2-3条），每条一句话。
3. concerns：需要正视的风险点（2-3条），给出具体原因而非泛泛而谈。
4. summary：40-60字，给出明确的下一步建议。如果客户有明显短板，可以建议对应的提升方案（如科研/实习/语言）。

严格输出JSON：
{"overview":"...","strengths":[""],"concerns":[""],"summary":"..."}
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

    # Format language score properly
    lang_type = ctx.language_type or bg.get("language_type", "")
    raw_lang = ctx.language_score_raw or bg.get("language_score")
    if raw_lang and raw_lang > 0:
        if lang_type in ("托福", "TOEFL"):
            lang_str = f"{lang_type} {raw_lang:.0f}"
        else:
            lang_str = f"雅思 {raw_lang:.1f}"
    else:
        lang_str = "未知"

    lines = [
        "## 学生背景",
        f"院校：{ctx.background_university or bg.get('university', '未知')}",
        f"专业：{ctx.background_major or bg.get('major', '未知')}",
        f"GPA：{ctx.gpa or bg.get('gpa', '未知')}",
        f"语言：{lang_str}",
    ]

    exp = ctx.experience_details or bg
    exp_parts = []
    for k, label in [
        ("research", "科研"),
        ("internship", "实习"),
        ("award", "获奖"),
        ("paper", "论文"),
    ]:
        v = exp.get(k) if exp else None
        if v:
            exp_parts.append(f"{label}：{v}")
    if exp_parts:
        lines.append(f"经历：{'；'.join(exp_parts)}")
    else:
        lines.append("经历：暂无")

    results = ctx.prediction_results or {}
    sim = results.get("similarity_results") or []
    cross = results.get("cross_major_results") or []
    unified = results.get("unified_results") or []

    total = len(unified) or len(sim) + len(cross)
    lines.append(f"\n## 预测结果（共 {total} 条推荐）")

    def _add(label: str, items: list) -> None:
        if not items:
            return
        lines.append(f"\n{label}：")
        for i, r in enumerate(items[:5]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = r.get("probability", 0)
            traces = []
            trace = r.get("_adjustment_trace")
            if isinstance(trace, list):
                for t in trace:
                    name = t.get("name", "") if isinstance(t, dict) else str(t)
                    if "penalty" in name.lower():
                        traces.append(name)
            lang_pen = " [语言]" if r.get("language_penalty_applied") else ""
            trace_note = f"({' '.join(traces)})" if traces else ""
            lines.append(f"  {i + 1}. {uni} {maj}  {_fmt_prob(prob)}{lang_pen}{trace_note}")

    _add("相似专业推荐", sim)
    _add("跨专业推荐", cross)

    if unified:
        lines.append("\n综合排名前5：")
        for i, r in enumerate(unified[:5]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = r.get("probability", 0)
            lines.append(f"  {i + 1}. {uni} {maj}  {_fmt_prob(prob)}")

    return "\n".join(lines)


class ExplainAgent(BaseAgent):
    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config=config, timeout=15, agent_name="解释Agent")
        self._system_prompt = EXPLAIN_SYSTEM_PROMPT
        self._stream_buffer = ""

    # ── shared prompt construction ──────────────────────────────────────
    def _prepare(self, context: StudentContext) -> str:
        pred = context.prediction_results or {}
        self.logger.info(
            f"[{self.agent_name}] PREPARE | "
            f"sim={len(pred.get('similarity_results') or [])} "
            f"cross={len(pred.get('cross_major_results') or [])} "
            f"unified={len(pred.get('unified_results') or [])} | "
            f"gpa={context.gpa} lang={context.language_type} {context.language_score}"
        )
        prompt = _build_explain_prompt(context)
        return f"{self._system_prompt}\n\n{prompt}"

    # ── streaming path ──────────────────────────────────────────────────
    def stream(self, context: StudentContext):
        t_start = time.perf_counter()
        full_prompt = self._prepare(context)
        self._stream_buffer = ""

        for chunk in self._call_api_streaming(full_prompt, max_tokens=600):
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

    # ── sync path ───────────────────────────────────────────────────────
    def run(self, context: StudentContext) -> dict[str, Any]:
        t_start = time.perf_counter()
        full_prompt = self._prepare(context)
        raw = self._call_api(full_prompt, cache_prefix="explain", max_tokens=600)

        result = self._parse_response(raw)
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        if result is None:
            self.logger.warning(f"[{self.agent_name}] RUN FAILED | total={elapsed_ms:.0f}ms")
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
