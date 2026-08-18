from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from pydantic import BaseModel, Field
from pydantic_ai import Agent, AgentRunResultEvent, PromptedOutput
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
from pydantic_ai.models import Model

from src.agent.context import StudentContext
from src.agent.explain_profiles import SYSTEM_PROMPT, sales_mode_suffix
from src.agent.runtime.model_factory import (
    build_fallback_models,
    build_model_with_fallback,
)
from src.utils.numeric import clip_probability_coerce

_log = logging.getLogger(__name__)


class SchoolNote(BaseModel):
    university: str
    major: str
    note: str


class ProductSuggestion(BaseModel):
    name: str
    reason: str


class ExplainOutput(BaseModel):
    overview: str
    strengths: list[str] = Field(default_factory=list)
    concerns: list[str] = Field(default_factory=list)
    summary: str = ""
    school_notes: list[SchoolNote] = Field(default_factory=list)
    products: list[ProductSuggestion] = Field(default_factory=list)


def _build_explain_agent(model: Model) -> Agent:
    try:
        from src.agent.harness import load_harness_config

        mc = load_harness_config().max_concurrency
    except Exception:
        mc = 1
    return Agent(
        model,
        max_concurrency=mc,
        output_type=PromptedOutput(ExplainOutput),
        instructions=(
            "你是一位资深留学顾问，擅长分析香港、新加坡、澳门、马来西亚硕士申请。"
            "根据提供的学生背景和预测结果，生成结构化的申请解读。"
            "包含：总体概述(overview)、优势(strengths)、关注点(concerns)、"
            "总结(summary)、院校备注(school_notes)、推荐产品(products)。"
        ),
        retries=2,
    )


def _fmt_prob(p: float) -> str:
    return f"{p:.0%}" if p else "0%"


def _get_contract_prompt_context(contract_tier: str | None = None) -> str:
    try:
        if not contract_tier:
            return ""
        from src.utils.contract_config import get_contract_context

        return get_contract_context(contract_tier, format_type="full")
    except Exception:
        _log.warning("读取合同上下文失败", exc_info=True)
        return ""


def _build_explain_prompt(ctx: StudentContext) -> str:
    bg = ctx.extracted_background or {}

    lang_type = ctx.language_type or bg.get("language_type", "")
    raw_lang = ctx.language_score_raw or bg.get("language_score")
    if raw_lang and raw_lang > 0:
        lang_str = (
            f"{lang_type} {raw_lang:.0f}"
            if lang_type in ("托福", "TOEFL")
            else f"雅思 {raw_lang:.1f}"
        )
    else:
        lang_str = "未知"

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
        *(
            [f"第二专业（{'双学位' if ctx.is_dual_degree else '辅修'}）：{ctx.background_major_2}"]
            if ctx.background_major_2
            else []
        ),
        gpa_line,
        f"语言：{lang_str}",
    ]
    if exam_type and exam_score and float(exam_score) > 0:
        lines.append(f"标化考试：{exam_type} {exam_score:.0f}")

    exp = ctx.experience_details or bg
    exp_parts = []
    for k, label, alt_k in [
        ("research_details", "科研", "research"),
        ("internship_details", "实习", "internship"),
        ("award_details", "获奖", "award"),
        ("paper_details", "论文", "paper"),
    ]:
        v = exp.get(k) if exp else None
        if not v:
            v = exp.get(alt_k) if exp else None
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

    quality_signals: dict[str, object] = {}
    for r in unified or sim or cross or []:
        qs = (r.get("_adjustment_trace") or {}).get("quality_signals")
        if qs:
            quality_signals = qs
            break
    if quality_signals and isinstance(quality_signals, dict):
        field_cn = {
            "research_details": "科研经历",
            "award_details": "获奖经历",
            "internship_details": "实习经历",
            "paper_details": "论文发表",
        }
        llm_verified = quality_signals.get("llm_verified", {})
        raw_tags = quality_signals.get("raw_tags", {})
        lines.append("\n## 经历含金量标签（仅定性参考，未进入概率计算）")
        if isinstance(llm_verified, dict) and llm_verified:
            for field, lv in llm_verified.items():
                if not isinstance(lv, dict):
                    continue
                label = field_cn.get(field, field)
                verified = lv.get("verified_tags") or []
                missed = lv.get("missed_signals") or []
                ql = lv.get("quality_level", "")
                concern = lv.get("concern")
                all_tags = list(verified) + list(missed)
                if all_tags:
                    lines.append(f"- {label}（{ql}）：{', '.join(all_tags)}")
                if concern:
                    lines.append(f"  ⚠ 注意：{concern}")
        elif isinstance(raw_tags, dict) and raw_tags:
            for field, tags in raw_tags.items():
                if tags:
                    label = field_cn.get(field, field)
                    lines.append(f"- {label}：{', '.join(tags)}")
        lines.append("\n注意：含金量标签经 AI 校验，仅供参考。")

    total = len(unified) or len(sim) + len(cross)
    lines.append(f"\n## 预测结果（共 {total} 条推荐）")

    def _add(label: str, items: list) -> None:
        if not items:
            return
        lines.append(f"\n{label}：")
        for i, r in enumerate(items[:5]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = clip_probability_coerce(r.get("probability"))
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
            biz_flags = r.get("_business_flags") or []
            flag_tags = ""
            if biz_flags:
                flag_parts = []
                for bf in biz_flags:
                    sev = bf.get("severity", "")
                    title = bf.get("title", "")
                    icon = _SEVERITY_ICON.get(sev, "")
                    if icon:
                        flag_parts.append(f"[{icon} {title}]")
                if flag_parts:
                    flag_tags = " " + " ".join(flag_parts)
            lines.append(
                f"  {i + 1}. {uni} {maj}  {_fmt_prob(prob)}{lang_pen}{trace_note}{flag_tags}"
            )

    _add("相似专业推荐", sim)
    _add("跨专业推荐", cross)

    if unified:
        lines.append("\n综合排名前5：")
        for i, r in enumerate(unified[:5]):
            uni = r.get("university", "")
            maj = r.get("major", "")
            prob = clip_probability_coerce(r.get("probability"))
            biz_flags = r.get("_business_flags") or []
            flag_tags = ""
            if biz_flags:
                flag_parts = []
                for bf in biz_flags:
                    sev = bf.get("severity", "")
                    title = bf.get("title", "")
                    icon = _SEVERITY_ICON.get(sev, "")
                    if icon:
                        flag_parts.append(f"[{icon} {title}]")
                if flag_parts:
                    flag_tags = " " + " ".join(flag_parts)
            lines.append(f"  {i + 1}. {uni} {maj}  {_fmt_prob(prob)}{flag_tags}")

    portfolio = ctx.portfolio_combo or []
    if portfolio:
        from src.agent.schemas import compute_tiers

        lines.append(f"\n## 选校优化组合（共 {len(portfolio)} 所）")
        tiers = compute_tiers([clip_probability_coerce(p.get("probability")) for p in portfolio])
        for i, (s, tier) in enumerate(zip(portfolio, tiers, strict=False)):
            uni = s.get("university", "")
            maj = s.get("major", "")
            prob = clip_probability_coerce(s.get("probability"))
            sim = s.get("similarity", 0)
            lines.append(
                f"  {i + 1}. [{tier}] {uni} {maj}  {_fmt_prob(prob)}"
                f"  专业匹配度 {float(sim) * 100:.0f}%"
            )
        lines.append("\n请重点解释上述「选校优化组合」中的学校。")

    return "\n".join(lines)


_SEVERITY_ICON: dict[str, str] = {"blocker": "X", "warning": "!", "positive": "OK", "info": "i"}


def _extract_model_response(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    if body:
        return str(body)[:500]
    return (getattr(exc, "message", None) or str(exc))[:500]


class ExplainAgent:
    def __init__(self, config: dict[str, Any] | None = None):
        _ = config
        timeout = 20

        self._model = build_model_with_fallback(timeout=timeout)
        self._agent = _build_explain_agent(self._model)

        fb_models = build_fallback_models(timeout=timeout)
        self._refusal_fallback_agents: list[Agent] = [_build_explain_agent(m) for m in fb_models]

        self._stream_buffer = ""
        self._stream_prompt = ""

        self.agent_name = "解释Agent"
        self.logger = _log

    def _prepare(self, context: StudentContext) -> str:
        system_prompt = SYSTEM_PROMPT + sales_mode_suffix()
        contract_ctx = _get_contract_prompt_context(context.contract_tier)
        data_prompt = _build_explain_prompt(context)
        if context.sales_snapshot:
            ss = context.sales_snapshot
            sel = ss.get("blocks_selection") or []
            data_prompt += (
                f"\n### 销售方案快照\n"
                f"- 展示概率: {ss.get('display_pct')}%\n"
                f"- 基准: {ss.get('base_pct')}%\n"
                f"- 已选: {', '.join(sel)}\n"
            )
        products = context.matched_products
        if products:
            prod_lines = ["\n### 已推荐服务产品"]
            for p in products:
                prod_lines.append(
                    f"- {p.get('name', '')}（{p.get('variant', '')} · {p.get('price', '')}）"
                )
            data_prompt += "\n".join(prod_lines)
        full = f"{system_prompt}\n\n{contract_ctx}\n\n{data_prompt}"
        return full

    def stream(self, context: StudentContext):
        t_start = time.perf_counter()
        full_prompt = self._prepare(context)
        self._stream_buffer = ""
        self._stream_prompt = full_prompt
        self._stream_context = context
        prompt_len = len(full_prompt)

        _log.info("解释流式生成开始: prompt_len=%d", prompt_len)

        try:
            yield from self._iter_async(self._async_stream(full_prompt))
        except Exception:
            _log.warning("流式生成失败", exc_info=True)
            yield ""

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        output_len = len(self._stream_buffer)
        _log.info(
            "解释流式完成 | total=%dms prompt=%dchars output=%dchars",
            int(elapsed_ms),
            prompt_len,
            output_len,
        )
        if output_len == 0:
            _log.warning("流式输出为空")

    @staticmethod
    def _iter_async(agen):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError(
                "_iter_async 只能在无运行中 event loop 的同步线程调用（Streamlit 脚本线程）"
            )
        loop = asyncio.new_event_loop()
        try:
            while True:
                try:
                    chunk = loop.run_until_complete(agen.__anext__())
                    yield chunk
                except StopAsyncIteration:
                    break
        finally:
            try:
                loop.run_until_complete(agen.aclose())
            except Exception:
                pass
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            loop.close()

    @staticmethod
    async def _iter_stream_events(agent: Agent, prompt: str):
        async with agent.run_stream_events(prompt) as stream:
            async for event in stream:
                yield event

    async def _async_stream(self, prompt: str):
        try:
            async for event in self._iter_stream_events(self._agent, prompt):
                if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                    chunk = event.delta.content_delta
                    if chunk:
                        self._stream_buffer += chunk
                        yield chunk
                elif isinstance(event, AgentRunResultEvent):
                    self._stream_parsed = event.result.output
        except UnexpectedModelBehavior:
            _log.warning(
                "流式主模型拒答 | prompt_head=%s",
                prompt[:500],
            )
            if not self._refusal_fallback_agents:
                _log.warning("流式拒答，无 refusal fallback agent 可用")
                raise
            for i, fb_agent in enumerate(self._refusal_fallback_agents):
                _log.info("流式拒答，尝试 refusal fallback #%d", i + 1)
                try:
                    async for event in self._iter_stream_events(fb_agent, prompt):
                        if isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            chunk = event.delta.content_delta
                            if chunk:
                                self._stream_buffer += chunk
                                yield chunk
                        elif isinstance(event, AgentRunResultEvent):
                            self._stream_parsed = event.result.output
                    return
                except UnexpectedModelBehavior as fb_exc:
                    _log.warning(
                        "流式 Fallback #%d 拒答 | response=%s",
                        i + 1,
                        _extract_model_response(fb_exc),
                    )
                    continue
                except Exception:
                    _log.warning("流式 refusal fallback #%d 失败", i + 1, exc_info=True)
                    continue
            _log.warning(
                "所有流式 refusal fallback 均失败 (%d agents)", len(self._refusal_fallback_agents)
            )
            raise

    def _refusal_fallback_sync(self, prompt: str) -> tuple[ExplainOutput, int] | None:
        if not self._refusal_fallback_agents:
            _log.warning("无 refusal fallback agent 可用")
            return None
        for i, fb_agent in enumerate(self._refusal_fallback_agents):
            _log.info("模型拒答，尝试 refusal fallback #%d", i + 1)
            try:
                fb_result = fb_agent.run_sync(prompt)
                fb_output = fb_result.output
                if isinstance(fb_output, ExplainOutput):
                    return fb_output, i + 1
            except UnexpectedModelBehavior as fb_exc:
                _log.warning(
                    "Fallback #%d 拒答 | response=%s",
                    i + 1,
                    _extract_model_response(fb_exc),
                )
            except Exception:
                _log.warning("Refusal fallback #%d 失败", i + 1, exc_info=True)
        _log.warning(
            "所有 refusal fallback 均失败 (%d agents)",
            len(self._refusal_fallback_agents),
        )
        return None

    def parse_stream_result(self) -> dict[str, Any] | None:
        if hasattr(self, "_stream_parsed") and isinstance(self._stream_parsed, ExplainOutput):
            output = self._stream_parsed
            result = output.model_dump()
        elif self._stream_buffer.strip():
            try:
                import json
                import re

                text = self._stream_buffer.strip()
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
                parsed = json.loads(text)
                result = {
                    "overview": parsed.get("overview", ""),
                    "strengths": parsed.get("strengths", []),
                    "concerns": parsed.get("concerns", []),
                    "summary": parsed.get("summary", ""),
                    "school_notes": parsed.get("school_notes", []),
                    "products": parsed.get("products", []),
                }
            except (json.JSONDecodeError, KeyError):
                _log.warning("parse_stream_result: buffer 解析失败")
                return None
        else:
            return None

        ctx = getattr(self, "_stream_context", None)
        if ctx is not None:
            result = self._postprocess(result, ctx)
            ctx.ai_explanation = result.get("overview", "")

        result["_trace"] = {
            "agent_name": self.agent_name,
            "prompt_length": len(self._stream_prompt),
            "response_length": len(self._stream_buffer),
        }
        return result

    def run(self, context: StudentContext) -> dict[str, Any]:
        t_start = time.perf_counter()
        full_prompt = self._prepare(context)
        prompt_len = len(full_prompt)

        _log.info("解释同步生成开始: prompt_len=%d", prompt_len)

        try:
            result = self._agent.run_sync(full_prompt)
            output = result.output
            if not isinstance(output, ExplainOutput):
                _log.warning("解释返回非预期类型: %s", type(output))
                return _fallback_result("解释暂不可用", full_prompt, prompt_len, 0)

            result_dict = output.model_dump()
            elapsed_ms = (time.perf_counter() - t_start) * 1000

            context.ai_explanation = result_dict.get("overview", "")
            result_dict = self._postprocess(result_dict, context)

            result_dict["_trace"] = {
                "agent_name": self.agent_name,
                "elapsed_ms": round(elapsed_ms, 1),
                "prompt_length": prompt_len,
                "response_length": len(str(result.output)),
            }

            _log.info(
                "解释同步完成 | total=%dms overview_len=%d strengths=%d concerns=%d",
                int(elapsed_ms),
                len(result_dict.get("overview", "")),
                len(result_dict.get("strengths", [])),
                len(result_dict.get("concerns", [])),
            )
            return result_dict

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - t_start) * 1000
            _log.warning(
                "解释同步失败 | total=%dms prompt_len=%d",
                int(elapsed_ms),
                prompt_len,
                exc_info=True,
            )

            if isinstance(exc, UnexpectedModelBehavior):
                refusal_text = _extract_model_response(exc)
                _log.warning(
                    "模型拒答诊断 | response=%s prompt_head=%s prompt_tail=%s",
                    refusal_text,
                    full_prompt[:500],
                    full_prompt[-500:],
                )
                fb = self._refusal_fallback_sync(full_prompt)
                if fb is not None:
                    fb_output, fb_index = fb
                    result_dict = fb_output.model_dump()
                    fb_elapsed = (time.perf_counter() - t_start) * 1000
                    context.ai_explanation = result_dict.get("overview", "")
                    result_dict = self._postprocess(result_dict, context)
                    result_dict["_trace"] = {
                        "agent_name": self.agent_name,
                        "elapsed_ms": round(fb_elapsed, 1),
                        "prompt_length": prompt_len,
                        "response_length": len(str(fb_output)),
                        "refusal_fallback": True,
                        "refusal_fallback_index": fb_index,
                    }
                    _log.info(
                        "Refusal fallback #%d 成功 | total=%dms",
                        fb_index,
                        int(fb_elapsed),
                    )
                    return result_dict

            return _generate_data_fallback(context, full_prompt, prompt_len, elapsed_ms)

    def _postprocess(self, result: dict[str, Any], context: StudentContext) -> dict[str, Any]:
        changes = []

        school_notes = result.get("school_notes") or []
        unified = (context.prediction_results or {}).get("unified_results") or []
        if school_notes and unified:
            uni_names = list({r.get("university", "") for r in unified if r.get("university")})
            from rapidfuzz import fuzz, process

            for note in school_notes:
                if not isinstance(note, dict):
                    continue
                uni = note.get("university", "")
                if uni and uni in uni_names:
                    continue
                if uni and uni_names:
                    match = process.extractOne(
                        uni, uni_names, scorer=fuzz.partial_ratio, score_cutoff=70
                    )
                    if match:
                        old_name = uni
                        note["university"] = match[0]
                        changes.append(f"院校名对齐: '{old_name}' → '{match[0]}'")

        products = result.get("products") or []
        if products:
            from src.pages.prediction.result_display.product_registry import (
                registry_by_name,
            )

            if context.sales_snapshot:
                selectable = context.sales_snapshot.get("_selectable_names")
                valid_names = set(selectable) if selectable else set(registry_by_name().keys())
            else:
                valid_names = set(registry_by_name().keys())
            filtered = [
                p for p in products if isinstance(p, dict) and p.get("name", "") in valid_names
            ]
            removed = len(products) - len(filtered)
            if removed > 0:
                changes.append(f"产品白名单过滤: 移除 {removed} 个无效产品")
            result["products"] = filtered

        if changes:
            _log.info("后处理变更: %s", "; ".join(changes))
        return result


def _generate_data_fallback(
    context: StudentContext,
    full_prompt: str,
    prompt_len: int,
    elapsed_ms: float,
) -> dict[str, Any]:
    results = context.prediction_results or {}
    unified = results.get("unified_results") or []
    sim = results.get("similarity_results") or []
    cross = results.get("cross_major_results") or []
    portfolio = context.portfolio_combo or []
    bg = context.extracted_background or {}

    all_results: list[dict] = unified or sim + cross
    total = len(all_results)
    if total == 0:
        return _fallback_result("", full_prompt, prompt_len, elapsed_ms)

    top = max(all_results, key=lambda r: float(r.get("probability", 0)))
    top_uni = top.get("university", "")
    top_major = top.get("major", "")
    top_prob = float(top.get("probability", 0))

    high = sum(1 for r in all_results if float(r.get("probability", 0)) >= 0.5)
    mid = sum(1 for r in all_results if 0.2 <= float(r.get("probability", 0)) < 0.5)
    low = sum(1 for r in all_results if float(r.get("probability", 0)) < 0.2)

    parts: list[str] = [f"共为您匹配 {total} 条推荐。"]
    if top_uni and top_major:
        parts.append(f"其中 **{top_uni} {top_major}** 录取概率最高，约 {top_prob:.0%}。")
    if high > 0:
        parts.append(f"高录取概率（≥50%）共 {high} 条。")
    if mid > 0:
        parts.append(f"中等录取概率（20%-50%）共 {mid} 条。")
    if low > 0:
        parts.append(f"低录取概率（<20%）共 {low} 条。")
    overview = " ".join(parts)

    strengths: list[str] = []
    raw_gpa = context.gpa_raw or float(bg.get("gpa_raw", 0) or 0)
    if raw_gpa > 3.5:
        strengths.append(f"GPA {raw_gpa:.2f} 具有竞争力，为申请加分")
    elif raw_gpa >= 3.0:
        strengths.append(f"GPA {raw_gpa:.2f} 处于中等水平，满足多数项目要求")
    lang_raw = context.language_score_raw or float(bg.get("language_score", 0) or 0)
    lang_type = context.language_type or bg.get("language_type", "")
    if lang_raw >= 7.0:
        strengths.append(f"语言成绩 {lang_type} {lang_raw:.1f} 达到较高水平")
    elif lang_raw >= 6.0:
        strengths.append(f"语言成绩 {lang_type} {lang_raw:.1f} 满足基本要求")
    exp = context.experience_details or bg
    if isinstance(exp, dict):
        if exp.get("research_details"):
            strengths.append("有科研经历，可增强学术背景")
        if exp.get("internship_details"):
            strengths.append("有实习经历，体现实践能力")
        if exp.get("award_details"):
            strengths.append("有获奖经历，体现综合素质")
    if not strengths:
        strengths.append("请完善个人背景信息以获得更精准的分析")

    concerns: list[str] = []
    if 0 < raw_gpa < 3.0:
        concerns.append(f"GPA {raw_gpa:.2f} 偏低，建议重点申请录取概率 ≥50% 的项目")
    if 0 < lang_raw < 6.0:
        concerns.append(f"语言成绩偏低（{lang_type} {lang_raw:.1f}），部分项目可能要求补充语言成绩")
    cross_penalties = sum(
        1 for r in sim if (r.get("_adjustment_trace") or {}).get("cross_major_penalty")
    )
    if cross_penalties > 0:
        concerns.append("部分推荐受跨专业惩罚影响，建议关注专业匹配度较高的项目")
    if not concerns:
        concerns.append("当前背景无明显短板，建议均衡选校")

    school_notes: list[dict[str, str]] = []
    display_results = portfolio if portfolio else all_results[:5]
    for r in display_results[:5]:
        uni = str(r.get("university", ""))
        maj = str(r.get("major", ""))
        prob = float(r.get("probability", 0))
        flags = r.get("_business_flags") or []
        flag_notes = [
            f.get("title", "")
            for f in flags
            if isinstance(f, dict) and f.get("severity") in ("blocker", "warning")
        ]
        note_parts = [f"录取概率约 {prob:.0%}"]
        if flag_notes:
            note_parts.append(f"注意: {', '.join(flag_notes)}")
        school_notes.append({"university": uni, "major": maj, "note": "；".join(note_parts)})

    summary = (
        f"以上分析基于历史申请数据自动生成，非 AI 解读。"
        f"共评估 {total} 个志愿，最高录取概率 {top_prob:.0%}。"
        f"建议结合个人偏好与项目特点进一步筛选。"
    )

    return {
        "overview": overview,
        "strengths": strengths,
        "concerns": concerns,
        "summary": summary,
        "school_notes": school_notes,
        "products": [],
        "_data_fallback": True,
        "_trace": {
            "agent_name": "解释Agent",
            "elapsed_ms": round(elapsed_ms, 1),
            "prompt_length": prompt_len,
            "data_fallback": True,
        },
    }


def _fallback_result(
    overview: str, full_prompt: str, prompt_len: int, elapsed_ms: float
) -> dict[str, Any]:
    return {
        "overview": overview,
        "strengths": [],
        "concerns": [],
        "summary": "",
        "school_notes": [],
        "products": [],
        "_error": "ai_failed",
        "_raw_snippet": full_prompt[:300],
        "_trace": {
            "agent_name": "解释Agent",
            "elapsed_ms": round(elapsed_ms, 1),
            "prompt_length": prompt_len,
            "error": True,
        },
    }
