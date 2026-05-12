import time
from typing import Any

from src.agent.base_agent import BaseAgent
from src.agent.context import StudentContext
from src.agent.explain_profiles import PROFILE_PROMPTS, classify_profile


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
        magnitude = t.get("value")
        if desc:
            parts.append(f"{label}{name}({desc})")
        elif isinstance(magnitude, (int, float)):
            parts.append(f"{label}{name}({magnitude:+.2f})")
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

    # Format GPA line — show raw GPA + standardized test bonus if applicable
    raw_gpa = ctx.gpa_raw or bg.get("gpa_raw") or 0
    exam_type = ctx.standardized_test_type or bg.get("standardized_test_type", "")
    exam_score = ctx.standardized_test_score or bg.get("standardized_test_score", 0)
    gpa_display = ctx.gpa or bg.get("gpa", 0)

    if (
        exam_type
        and exam_score
        and float(exam_score) > 0
        and raw_gpa > 0
        and gpa_display > raw_gpa + 0.001
    ):
        gpa_line = (
            f"GPA：{raw_gpa:.2f}（原始） + {exam_type} {exam_score:.0f} 加成 → "
            f"模型计算用 {gpa_display:.2f}"
        )
    elif ctx.gpa or bg.get("gpa"):
        gpa_line = f"GPA：{gpa_display:.2f}"
    else:
        gpa_line = "GPA：未知"

    lines = [
        "## 学生背景",
        f"院校：{ctx.background_university or bg.get('university', '未知')}",
        f"专业：{ctx.background_major or bg.get('major', '未知')}",
        gpa_line,
        f"语言：{lang_str}",
    ]
    if exam_type and exam_score and float(exam_score) > 0:
        lines.append(f"标化考试：{exam_type} {exam_score:.0f}")

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
            steps = r.get("_adjustment_steps") or []
            for s in steps:
                if not isinstance(s, dict):
                    continue
                if s.get("type") != "penalty":
                    continue
                name = s.get("name", "")
                desc = s.get("description", "")
                traces.append(f"{name}({desc})" if desc else name)
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
        super().__init__(config=config, timeout=20, agent_name="解释Agent")
        self._stream_buffer = ""

    # ── shared prompt construction ──────────────────────────────────────
    def _prepare(self, context: StudentContext) -> str:
        pred = context.prediction_results or {}
        profile = context.profile_type or classify_profile(pred)
        system_prompt = PROFILE_PROMPTS.get(profile, PROFILE_PROMPTS["medium_mixed"])
        self.logger.info(
            f"[{self.agent_name}] PREPARE | profile={profile} | "
            f"sim={len(pred.get('similarity_results') or [])} "
            f"cross={len(pred.get('cross_major_results') or [])} "
            f"unified={len(pred.get('unified_results') or [])} | "
            f"gpa={context.gpa} lang={context.language_type} {context.language_score}"
        )
        data_prompt = _build_explain_prompt(context)
        products = context.matched_products
        if products:
            prod_lines = ["\n### 已推荐服务产品"]
            for p in products:
                prod_lines.append(
                    f"- {p.get('name', '')}（{p.get('variant', '')} · {p.get('price', '')}）"
                )
            data_prompt += "\n".join(prod_lines)
        return f"{system_prompt}\n\n{data_prompt}"

    # ── streaming path ──────────────────────────────────────────────────
    def stream(self, context: StudentContext):
        t_start = time.perf_counter()
        full_prompt = self._prepare(context)
        self._stream_buffer = ""

        for chunk in self._call_api_streaming(full_prompt, max_tokens=700):
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
        raw = self._call_api(full_prompt, cache_prefix="explain", max_tokens=700)

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
                '{"overview": "...", "strengths": [...], "concerns": [...], "summary": "...", '
                '"school_notes": [{"university": "...", "major": "...", "note": "..."}], '
                '"products": [{"name": "...", "reason": "..."}]}'
            ),
            cache_prefix="explain_json_repair",
            max_tokens=700,
        )
