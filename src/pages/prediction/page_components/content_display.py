import hashlib
import json
import os
import time
from typing import Any

import streamlit as st

from src.agent.context import StudentContext
from src.agent.explain_profiles import ProfileType, classify_profile
from src.agent.registry import AgentRegistry
from src.pages.prediction.ai_report import (
    render_ai_section,
    render_static_frame,
)
from src.pages.prediction.ai_report_sections import (
    render_ai_section_streaming,
    render_school_cards,
)
from src.pages.prediction.ai_school_stats import SchoolFeatureStats
from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS, DEFAULT_UI_KEYS
from src.pages.prediction.page_components.result_section import display_results_section
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

content_display_logger = setup_logger("page3", "prediction")

# ─── persistent explain cache ──────────────────────────────────────────
_EXPLAIN_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".explain_cache")
_EXPLAIN_CACHE_MAX = 50
_EXPLAIN_CACHE_TTL = 30 * 60  # 30 minutes


def _explain_cache_path() -> str:
    return os.path.join(_EXPLAIN_CACHE_DIR, "explain.json")


def _load_explain_cache() -> dict[str, Any]:
    try:
        path = _explain_cache_path()
        if not os.path.exists(path):
            return {}
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        now = time.time()
        return {k: v for k, v in data.items() if now - v.get("_ts", 0) < _EXPLAIN_CACHE_TTL}
    except Exception:
        return {}


def _persist_explain_cache(cache: dict[str, Any]) -> None:
    try:
        os.makedirs(_EXPLAIN_CACHE_DIR, exist_ok=True)
        keys = sorted(cache.keys(), key=lambda k: cache[k].get("_ts", 0))
        while len(keys) > _EXPLAIN_CACHE_MAX:
            oldest = keys.pop(0)
            del cache[oldest]
        path = _explain_cache_path()
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


@st.cache_resource(show_spinner=False)
def _get_school_stats() -> SchoolFeatureStats | None:
    try:
        from src.pages.prediction.page_data_loader import machine_learning_model

        mlm = machine_learning_model.resource_loader()
        return SchoolFeatureStats(mlm.cases_df)
    except Exception:
        return None


def _render_unified_school_cards(
    result: dict[str, Any] | None,
    unified: list[dict[str, Any]],
    percentile_map: dict[str, dict[str, Any]],
) -> None:
    """Render unified per-school cards with AI note + probability + stats."""
    if not result or not unified:
        return
    school_notes = result.get("school_notes")
    if not school_notes:
        return
    cards_html = render_school_cards(
        school_notes,
        unified_results=unified,
        percentile_data=percentile_map,
        label="院校分析",
    )
    if cards_html:
        st.html(cards_html)


# ─── partial JSON extraction during streaming ────────────────────────────────


def _extract_json_array(buffer: str, field: str) -> list | None:
    """Extract a JSON array value for a key using bracket counting.

    Uses character-by-character scan with string/escape awareness so nested
    objects and arrays are handled correctly, unlike a lazy ``.*?`` regex.
    """
    import re

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
    return None  # not yet closed


def _try_extract_partial(buffer: str) -> dict[str, Any]:
    """Extract all available fields from a partial streaming JSON buffer.

    String fields use a fast regex; array/object fields use bracket counting.
    """
    import re

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


# ─── explain cache ───────────────────────────────────────────────────────────


def _compact_results(items: list[dict]) -> list[tuple[str, str, float]]:
    return [
        (
            str(item.get("university", "")),
            str(item.get("major", "")),
            round(float(item.get("probability", 0) or 0), 4),
        )
        for item in items
        if isinstance(item, dict)
    ]


def _compact_experience(experience_details: dict | None) -> dict[str, int]:
    return {str(k): len(str(v or "")) for k, v in (experience_details or {}).items() if v}


def _build_explain_cache_key(
    *,
    sim: list,
    cross: list,
    unified: list,
    background_major: str,
    gpa: float,
    gpa_raw: float = 0.0,
    exam_type: str = "",
    exam_score: float = 0.0,
    language_score: float = 0.0,
    language_type: str = "",
    experience_details: dict | None = None,
    profile: ProfileType | None = None,
) -> str:
    if profile is None:
        profile = classify_profile(
            {"similarity_results": sim, "cross_major_results": cross, "unified_results": unified}
        )
    key_data = {
        "v": 4,
        "profile": profile,
        "sim": _compact_results(sim[:5]),
        "cross": _compact_results(cross[:3]),
        "unified": _compact_results(unified[:5]),
        "background_major": background_major,
        "gpa": gpa,
        "gpa_raw": gpa_raw,
        "exam": [exam_type, exam_score] if exam_type and exam_score > 0 else [],
        "language": [language_type, language_score],
        "experience": _compact_experience(experience_details),
    }
    return hashlib.md5(json.dumps(key_data, sort_keys=True, default=str).encode()).hexdigest()


# ─── streaming core ──────────────────────────────────────────────────────────


def _stream_explain_content(
    ctx: StudentContext,
    agent,
    percentile_data: dict | None = None,
    unified_results: list | None = None,
) -> tuple[dict[str, Any] | None, object]:
    """Stream the AI explanation with progressive rendering.

    Returns (parsed_result, stream_placeholder).  If streaming produces no
    tokens we fall back to a synchronous API call automatically.
    """
    stream_placeholder = st.empty()
    with stream_placeholder.container():
        render_ai_section_streaming("")
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
        result = agent.parse_stream_result() or agent._parse_response(buffer)
    except Exception:
        content_display_logger.warning("流式解释失败，尝试同步降级", exc_info=True)
        stream_error = True
        result = None

    # Fallback: streaming produced nothing — try synchronous path
    if result is None and (stream_error or not buffer.strip()):
        content_display_logger.info("降级到同步 API 调用")
        try:
            result = agent.run(ctx)
        except Exception:
            content_display_logger.exception("同步 API 降级也失败")
            result = None

    return result, stream_placeholder


# ─── main explain orchestrator ───────────────────────────────────────────────


def _render_ai_explanation(
    prediction_results,
    input_data: dict[str, Any],
) -> None:
    """Orchestrate the full AI explanation flow: button-gate → static-frame
    → streaming LLM generation → caching → unified school cards."""
    sim = prediction_results.similarity_results or []
    cross = prediction_results.cross_major_results or []
    unified = prediction_results.unified_results or []
    if not any([sim, cross]):
        return

    background_university = input_data.get("background_university", "")
    background_major = input_data.get("background_major", "")
    gpa = float(input_data.get("gpa", 0) or 0)
    gpa_raw = float(input_data.get("gpa_raw", 0) or 0)
    exam_type = str(input_data.get("exam_type") or "")
    exam_score = float(input_data.get("exam_score", 0) or 0)
    language_score = float(input_data.get("language_score", 0) or 0)
    language_score_raw = float(input_data.get("language_score_raw", 0) or 0)
    language_type = str(input_data.get("language_type", ""))
    experience_details = input_data.get("experience_details")

    profile = classify_profile(
        {"similarity_results": sim, "cross_major_results": cross, "unified_results": unified}
    )

    # ── Compute percentile map early (before cache gate) ──
    stats = _get_school_stats()
    percentile_map: dict[str, dict[str, Any]] = {}
    gpa_for_percentile = gpa_raw if gpa_raw > 0 else gpa
    student_feat = {
        "gpa": gpa_for_percentile,
        "language_score": language_score,
        "research_count": int(input_data.get("research_count", 0) or 0),
        "internship_count": int(input_data.get("internship_count", 0) or 0),
        "award_count": int(input_data.get("award_count", 0) or 0),
        "paper_count": int(input_data.get("paper_count", 0) or 0),
    }
    if stats and unified:
        percentile_map = stats.get_school_feature_summary(student_feat, unified)

    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = _load_explain_cache()

    cache_key = _build_explain_cache_key(
        sim=sim,
        cross=cross,
        unified=unified,
        background_major=background_major,
        gpa=gpa,
        gpa_raw=gpa_raw,
        exam_type=exam_type,
        exam_score=exam_score,
        language_score=language_score,
        language_type=language_type,
        experience_details=experience_details,
        profile=profile,
    )

    # ── Cache hit: render immediately ──
    cached = st.session_state["explain_cache"].get(cache_key)
    if cached:
        products = render_static_frame(input_data, sim, cross, [])
        st.session_state["_ar_products"] = products
        render_ai_section(cached)
        _render_unified_school_cards(cached, unified, percentile_map)
        return

    # ── Button gate ──
    if "explain_generating" not in st.session_state:
        st.session_state["explain_generating"] = False

    is_generating = st.session_state["explain_generating"]
    c_left, _ = st.columns([2, 5])
    with c_left:
        label = "AI解读中..." if is_generating else "Pathfinder AI解读（beta）"
        clicked = st.button(label, key="explain_btn", width="stretch", disabled=is_generating)

    if not is_generating and not clicked:
        return

    if clicked:
        st.session_state["explain_generating"] = True

    # ── Fresh generation ──
    stream_placeholder = st.empty()
    try:
        products = render_static_frame(input_data, sim, cross, [])
        matched_products = products

        ctx = StudentContext(
            stage="match",
            background_university=background_university,
            background_major=background_major,
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
        )
        ctx.profile_type = profile
        ctx.matched_products = matched_products

        content_display_logger.info("使用ExplainAgent路径 | profile=%s", profile)
        agent = AgentRegistry.get("explain")
        result, stream_placeholder = _stream_explain_content(
            ctx, agent, percentile_data=percentile_map, unified_results=unified
        )

        if result and result.get("overview"):
            result["_ts"] = time.time()
            st.session_state["explain_cache"][cache_key] = result
            _persist_explain_cache(st.session_state["explain_cache"])
            _render_unified_school_cards(result, unified, percentile_map)
            with stream_placeholder.container():
                render_ai_section(result)
        else:
            stream_placeholder.empty()
            st.caption("解读暂不可用，稍后重试。")
    except Exception:
        content_display_logger.exception("AI解读生成失败")
        stream_placeholder.empty()
        st.caption("解读暂不可用，稍后重试。")
    finally:
        st.session_state["explain_generating"] = False

    st.rerun()


# ─── public API ──────────────────────────────────────────────────────────────


def display_content(
    session_manager: SessionManager,
    page_state: Any,
    submitted: bool,
    session_key_has_predicted: str = DEFAULT_SESSION_KEYS.has_predicted,
    session_key_input_data: str = DEFAULT_SESSION_KEYS.input_data,
    session_key_predict_lock: str = DEFAULT_SESSION_KEYS.predict_lock,
    session_key_form_data_changed: str = DEFAULT_SESSION_KEYS.form_data_changed,
) -> None:
    if not session_manager.get(session_key_has_predicted, False):
        return

    current_input_data = session_manager.get(session_key_input_data)
    if not current_input_data:
        content_display_logger.warning("has_predicted 为 True，但 session_state 中缺少输入数据。")
        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_has_predicted: False, session_key_predict_lock: False})
        st.rerun()

    form_changed = session_manager.get(session_key_form_data_changed, False)
    if not submitted and form_changed:
        st.caption("您的输入已更改，当前显示的是先前输入的预测结果。请点击预测按钮获取最新结果。")

    res_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)
    display_results_section(
        current_input_data,
        res_model.similarity_results,
        res_model.cross_major_results,
        res_model.user_specified_results,
        page_state.cases_df,
        submitted=submitted,
    )
    if res_model and (res_model.similarity_results or res_model.cross_major_results):
        _render_ai_explanation(res_model, current_input_data)

    if not submitted and form_changed:
        session_manager.set(**{session_key_form_data_changed: False})
