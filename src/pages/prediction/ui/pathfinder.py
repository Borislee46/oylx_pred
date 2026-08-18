from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import time
from datetime import datetime
from typing import Any

import streamlit as st

from src.agent import get_explain_agent
from src.agent.context import StudentContext
from src.agent.explain_profiles import classify_profile
from src.pages.prediction.result_display.ai_report import (
    SchoolFeatureStats,
    render_ai_section,
    render_ai_section_streaming,
    render_static_frame,
)
from src.pages.prediction.ui.explain_cache import (
    _PDF_PORTFOLIO_FP_KEY,
    build_explain_cache_key,
    compact_results,
    load_explain_cache,
    persist_explain_cache,
    portfolio_combo_fingerprint,
)
from src.pages.prediction.ui.timeline import mark_timeline_report_engaged
from src.utils.analytics import track as _track
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

pathfinder_logger = setup_logger("page3", "prediction")

_PERCENTILE_CACHE_KEY = "_hk_percentile_cache"


@st.cache_resource(show_spinner=False)
def _get_school_stats() -> SchoolFeatureStats | None:
    try:
        from src.pages.prediction.page_data_loader import machine_learning_model

        mlm = machine_learning_model.resource_loader()
        return SchoolFeatureStats(mlm.cases_df)
    except Exception:
        pathfinder_logger.exception("学校特征统计加载失败")
        return None


def _build_percentile_cache_key(student_feat: dict[str, Any], unified: list) -> str:
    key_data = {
        "feat": {k: student_feat.get(k) for k in sorted(student_feat)},
        "unified": compact_results(unified[:5]),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()


def _school_display_results(portfolio_combo: list | None, unified: list) -> list:
    return portfolio_combo if portfolio_combo else unified


def _get_percentile_map(student_feat: dict[str, Any], unified: list) -> dict[str, dict[str, Any]]:
    if not unified:
        return {}
    pct_key = _build_percentile_cache_key(student_feat, unified)
    cache = st.session_state.setdefault(_PERCENTILE_CACHE_KEY, {})
    if pct_key in cache:
        return cache[pct_key]
    stats = _get_school_stats()
    if not stats:
        return {}
    result = stats.get_school_feature_summary(student_feat, unified)
    cache[pct_key] = result
    return result


def _extract_json_array(buffer: str, field: str) -> list | None:
    key_m = re.search(rf'"{field}"\s*:\s*', buffer)
    if not key_m:
        return None
    start = key_m.end()
    if start >= len(buffer) or buffer[start] != "[":
        return None

    depth = 0
    in_string = False
    escape = False
    for i, ch in enumerate(buffer[start:], start):
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in ("[", "{"):
            depth += 1
        elif ch in ("]", "}"):
            depth -= 1
        if depth == 0:
            try:
                return json.loads(buffer[start : i + 1])
            except json.JSONDecodeError:
                return None
    return None


def _try_extract_partial(buffer: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ["overview", "summary"]:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', buffer)
        if m:
            result[field] = re.sub(r"\\(.)", r"\1", m.group(1))
    for field in ["strengths", "concerns", "school_notes", "products"]:
        parsed = _extract_json_array(buffer, field)
        if parsed is not None:
            result[field] = parsed
    return result


def _stream_explain_content(
    ctx: StudentContext,
    agent,
    percentile_data: dict | None = None,
    unified_results: list | None = None,
) -> tuple[dict[str, Any] | None, object]:
    stream_placeholder = st.empty()
    with stream_placeholder:
        st.markdown(":shimmer[Pathfinder 正在解读...]")
    buffer = ""
    last_update = 0.0
    _last_buf_len = 0
    merged: dict[str, Any] = {}
    seen_fields: set[str] = set()
    stream_error = False

    try:
        for chunk in agent.stream(ctx):
            buffer += chunk or ""
            now = time.monotonic()
            new_chars = len(buffer) - _last_buf_len
            if now - last_update < 0.025 and new_chars < 6:
                continue
            partial = _try_extract_partial(buffer)
            for key, val in partial.items():
                if val:
                    merged[key] = val
            new_seen = set(merged.keys()) - seen_fields
            seen_fields |= new_seen
            with stream_placeholder.container():
                render_ai_section_streaming(
                    merged,
                    seen_fields=frozenset(new_seen),
                    percentile_data=percentile_data,
                    unified_results=unified_results,
                )
            last_update = now
            _last_buf_len = len(buffer)
        result = agent.parse_stream_result()
    except Exception:
        pathfinder_logger.warning("流式解释失败，尝试同步降级", exc_info=True)
        stream_error = True
        result = None

    if result is None and (stream_error or not buffer.strip()):
        pathfinder_logger.info("降级到同步 API 调用")
        try:
            result = agent.run(ctx)
        except Exception:
            pathfinder_logger.exception("同步 API 降级也失败")
            result = None

    return result, stream_placeholder


def _ensure_pathfinder_pdf(
    portfolio_combo: list,
    ai_explanation: dict,
    input_data: dict,
) -> None:
    combo_fp = portfolio_combo_fingerprint(portfolio_combo)
    if (
        combo_fp
        and st.session_state.get(_PDF_PORTFOLIO_FP_KEY) == combo_fp
        and st.session_state.get("_hk_pdf_bytes")
    ):
        return
    _generate_and_cache_pdf(portfolio_combo, ai_explanation, input_data)
    if combo_fp:
        st.session_state[_PDF_PORTFOLIO_FP_KEY] = combo_fp


def _generate_and_cache_pdf(
    portfolio_combo: list | None,
    ai_explanation: dict | None,
    input_data: dict,
) -> None:
    if not portfolio_combo or not ai_explanation:
        return
    try:
        from src.report.pdf.generators.pdf_report_generator import (
            PDFReportGenerator,
        )

        nick = (
            st.session_state.get("user_nickname")
            or st.session_state.get("e2_user_nickname")
            or "用户"
        )
        date_str = datetime.now().strftime("%Y%m%d")
        prefix = "Signals择校报告_家长版"

        pdf_bytes = PDFReportGenerator().generate_report_from_pathfinder_v2(
            portfolio_combo=portfolio_combo,
            ai_explanation=ai_explanation,
            input_data=input_data,
            user_nickname=nick,
        )
        st.session_state["_hk_pdf_bytes"] = pdf_bytes
        st.session_state["_hk_pdf_name"] = f"{prefix}_{nick}_{date_str}.pdf"
        st.session_state["_hk_pdf_error"] = False
    except Exception:
        pathfinder_logger.exception("内嵌PDF生成失败")
        st.session_state["_hk_pdf_error"] = True


def _render_mini_pdf_download(label: str = "下载报告") -> None:
    pdf_bytes = st.session_state.get("_hk_pdf_bytes")
    pdf_name = st.session_state.get("_hk_pdf_name", "report.pdf")
    if pdf_bytes:
        st.download_button(
            label=label,
            data=pdf_bytes,
            file_name=pdf_name,
            mime="application/pdf",
            type="secondary",
            icon=":material/download:",
            key="hk_pdf_mini_dl",
        )


def render_ai_explanation(
    prediction_results,
    input_data: dict[str, Any],
    portfolio_combo: list | None = None,
    profile: str = "",
    sales_snapshot: Any | None = None,
) -> None:
    sim = prediction_results.similarity_results or []
    cross = prediction_results.cross_major_results or []
    unified = prediction_results.unified_results or []
    if not any([sim, cross]):
        return

    background_university = input_data.get("background_university", "")
    background_major = input_data.get("background_major", "")
    background_major_2 = input_data.get("background_major_2", "")
    is_dual_degree = input_data.get("is_dual_degree", False)
    gpa = float(input_data.get("gpa", 0) or 0)
    gpa_raw = float(input_data.get("gpa_raw", 0) or 0)
    exam_type = str(input_data.get("exam_type") or "")
    exam_score = float(input_data.get("exam_score", 0) or 0)
    language_score = float(input_data.get("language_score", 0) or 0)
    language_score_raw = float(input_data.get("language_score_raw", 0) or 0)
    language_type = str(input_data.get("language_type", ""))
    experience_details = input_data.get("experience_details")

    profile = profile or classify_profile(
        {"similarity_results": sim, "cross_major_results": cross, "unified_results": unified}
    )

    gpa_for_percentile = gpa_raw if gpa_raw > 0 else gpa
    student_feat = {
        "gpa": gpa_for_percentile,
        "language_score": language_score,
        "research_count": int(input_data.get("research_count", 0) or 0),
        "internship_count": int(input_data.get("internship_count", 0) or 0),
        "award_count": int(input_data.get("award_count", 0) or 0),
        "paper_count": int(input_data.get("paper_count", 0) or 0),
    }
    display_results = _school_display_results(portfolio_combo, unified)
    percentile_map = _get_percentile_map(student_feat, display_results) if display_results else {}

    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = load_explain_cache()

    blocks_selection = list(sales_snapshot.blocks_selection) if sales_snapshot else None
    display_pct = sales_snapshot.display_pct if sales_snapshot else None
    from src.pages.prediction.result_display.contract_state import CONTRACT_TIER_KEY

    contract_tier = str(SessionManager().get(CONTRACT_TIER_KEY) or "")

    cache_key = build_explain_cache_key(
        sim=sim,
        cross=cross,
        unified=unified,
        background_major=background_major,
        background_major_2=background_major_2,
        is_dual_degree=is_dual_degree,
        gpa=gpa,
        gpa_raw=gpa_raw,
        exam_type=exam_type,
        exam_score=exam_score,
        language_score=language_score,
        language_type=language_type,
        experience_details=experience_details,
        profile=profile,
        portfolio_combo=portfolio_combo,
        blocks_selection=blocks_selection,
        display_pct=display_pct,
        contract_tier=contract_tier,
    )

    use_blocks_selection = sales_snapshot is not None
    snapshot_products: list[dict] | None = None
    if use_blocks_selection and sales_snapshot:
        from src.pages.prediction.result_display.sales_recommendation import (
            blocks_selection_to_product_dicts,
        )

        snapshot_products = blocks_selection_to_product_dicts(
            list(sales_snapshot.blocks_selection), input_data
        )

    def _render_pathfinder_frame() -> list[dict]:
        return render_static_frame(
            input_data,
            sim,
            cross,
            [],
            products=snapshot_products,
            include_product_grid=True,
        )

    cached = st.session_state["explain_cache"].get(cache_key)
    if cached:
        mark_timeline_report_engaged(SessionManager())
        products = _render_pathfinder_frame()
        st.session_state["_hk_ar_products"] = products
        render_ai_section(
            cached,
            unified_results=display_results,
            percentile_data=percentile_map,
        )
        if portfolio_combo:
            _ensure_pathfinder_pdf(portfolio_combo, cached, input_data)
        _render_mini_pdf_download()
        return

    if "explain_generating" not in st.session_state:
        st.session_state["explain_generating"] = False

    is_generating = st.session_state["explain_generating"]
    c_left, _ = st.columns([2, 5])
    with c_left:
        label = "✦ AI 解读中..." if is_generating else "✦ Pathfinder AI 解读"
        clicked = st.button(label, key="explain_btn", width="stretch", disabled=is_generating)

    if not is_generating and not clicked:
        return

    if clicked:
        mark_timeline_report_engaged(SessionManager())
        st.session_state["explain_generating"] = True
        st.rerun(scope="fragment")

    stream_placeholder = st.empty()
    st.markdown(
        '<div id="hk-pin-ai"></div>'
        "<script>"
        "setTimeout(function(){"
        'var el=document.getElementById("hk-pin-ai");'
        'if(el)el.scrollIntoView({behavior:"smooth",block:"start"});'
        "},50);"
        "</script>",
        unsafe_allow_html=True,
    )
    result = None
    try:
        matched_products = _render_pathfinder_frame()

        ctx = StudentContext(
            stage="match",
            background_university=background_university,
            background_major=background_major,
            background_major_2=background_major_2,
            is_dual_degree=is_dual_degree,
            gpa=gpa,
            gpa_raw=gpa_raw,
            standardized_test_type=exam_type,
            standardized_test_score=exam_score,
            language_score=language_score,
            language_score_raw=language_score_raw,
            language_type=language_type or "",
            experience_details=experience_details or {},
            prediction_results={
                "similarity_results": sim,
                "cross_major_results": cross,
                "unified_results": unified,
            },
            contract_tier=contract_tier,
        )
        ctx.profile_type = profile
        ctx.matched_products = matched_products
        ctx.portfolio_combo = portfolio_combo or []
        if sales_snapshot:
            from src.pages.prediction.result_display.product_registry import (
                selectable_product_names,
            )

            ctx.sales_snapshot = dataclasses.asdict(sales_snapshot)
            ctx.sales_snapshot["_selectable_names"] = selectable_product_names(input_data)

        pathfinder_logger.info("使用ExplainAgent路径 | profile=%s", profile)
        _track("explain_requested", profile_type=profile)
        agent = get_explain_agent()
        result, stream_placeholder = _stream_explain_content(
            ctx,
            agent,
            percentile_data=percentile_map,
            unified_results=display_results,
        )

        if result and result.get("overview") and not result.get("_error"):
            mark_timeline_report_engaged(SessionManager())
            result["_ts"] = time.time()
            st.session_state["explain_cache"][cache_key] = result
            persist_explain_cache(st.session_state["explain_cache"])
            with stream_placeholder.container():
                render_ai_section(
                    result,
                    unified_results=display_results,
                    percentile_data=percentile_map,
                )
                if portfolio_combo:
                    _ensure_pathfinder_pdf(portfolio_combo, result, input_data)
                _render_mini_pdf_download()
        else:
            stream_placeholder.empty()
            st.caption("解读暂不可用，稍后重试。")
    except Exception:
        pathfinder_logger.exception("AI解读生成失败")
        stream_placeholder.empty()
        st.caption("解读暂不可用，稍后重试。")
    finally:
        st.session_state["explain_generating"] = False

    if result and result.get("overview"):
        st.rerun(scope="fragment")
