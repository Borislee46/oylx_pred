"""
预测结果展示编排层。

职责：
  1. display_content — 公共入口：检查状态 → 渲染结果 → 触发 AI 解读
  2. _render_ai_explanation — AI 解读编排：button-gate → 缓存检查 → 流式生成 → 缓存持久化
  3. _stream_explain_content — LLM 流式输出 + 渐进式渲染（partial JSON 提取）
  4. Explain 缓存：磁盘 JSON 文件（30min TTL, 最多 50 条）+ session_state 内存层

两层缓存架构：
  L1 (session_state)：页面级，重跑不丢失，刷新丢失
  L2 (磁盘)：跨 session 持久化，不同用户的相同输入可复用

流式渲染策略：
  DeepSeek 流式返回 JSON chunks。每 25ms 或 6+ 新字符时尝试部分解析：
  - 字符串字段 → 正则提取（快速）
  - 数组字段 → bracket counting（支持嵌套对象/数组）
  - 边解析边渲染，用户看到 AI 解读逐段出现而非白屏等待
"""

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
    render_ai_section,       # 渲染完整 AI 解读（静态）
    render_static_frame,     # 渲染 AI 解读的静态框架（产品卡片等非流式部分）
)
from src.pages.prediction.ai_report_sections import (
    render_ai_section_streaming,  # 流式渲染 AI 解读（每收到新字段时更新）
    render_school_cards,          # 渲染院校分析卡片
)
from src.pages.prediction.ai_school_stats import SchoolFeatureStats  # 院校统计特征（百分位等）
from src.pages.prediction.handler_config import DEFAULT_SESSION_KEYS, DEFAULT_UI_KEYS
from src.pages.prediction.page_components.result_section import display_results_section
from src.pages.prediction.results_handler import reset_prediction_results
from src.utils.logger import setup_logger
from src.utils.session_manager import SessionManager

content_display_logger = setup_logger("page3", "prediction")


# ── Fallback 通知 ──────────────────────────────────────────
def _check_and_render_fallback_notice(res_model: Any) -> None:
    """当预测结果来自 fallback（非模型推理）时显示警告通知。

    Fallback 层级（_fallback_level）：
      0 — 完全匹配：历史数据中恰好有此院校-专业组合
      1 — 同背景院校：基于相同本科院校的数据估算
      2 — 同目标组合：基于目标院校-专业的历史数据
      3 — 同目标院校：基于目标院校的全局数据
      4 — 全局兜底：全量历史数据的录取率

    层级越低说明匹配越精确，但本质上都不是个性化模型预测。
    触发条件：输入缺少 GPA 或语言成绩，或院校/专业无历史样本。
    """
    if res_model is None:
        return
    all_results = (
        (res_model.similarity_results or [])
        + (res_model.cross_major_results or [])
        + (res_model.user_specified_results or [])
    )
    if not all_results:
        return
    any_fallback = any(r.get("_is_fallback") for r in all_results)
    if not any_fallback:
        return
    # 取最低的 fallback level（最精确的匹配）
    fallback_level = min(
        (r.get("_fallback_level", 4) for r in all_results if r.get("_is_fallback")),
        default=4,
    )
    sample_counts = [
        r.get("_fallback_sample_count", 0)
        for r in all_results
        if r.get("_is_fallback") and r.get("_fallback_sample_count")
    ]
    min_n = min(sample_counts) if sample_counts else 0

    level_descriptions = {
        0: f"历史数据中有完全匹配的院校-专业组合（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        1: f"基于相同背景院校的数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        2: f"基于目标院校-专业的历史数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        3: f"基于目标院校的全局数据估算（n≥{min_n}），以下概率为历史录取率，非模型预测。",
        4: "基于全部历史数据的全局录取率估算，以下概率非个性化模型预测。",
    }
    desc = level_descriptions.get(fallback_level, level_descriptions[4])

    st.warning(
        f"⚠️ 数据不完整，无法运行个性化预测模型。{desc}\n\n"
        "录取概率仅供参考，建议补充完整的GPA和语言成绩信息后重新预测。",
        icon="⚠️",
    )


# ── Explain 缓存（双层：session_state + 磁盘）────────────
# L1: st.session_state["explain_cache"] — 页面级，重跑不丢失
# L2: .explain_cache/explain.json — 跨 session 持久化
#
# 缓存键：MD5(profile + top5 sim/cross/unified + background_major + gpa + language + experience)
# 相同输入 → 相同缓存键 → 跨用户复用（但不含 PII）
_EXPLAIN_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", ".explain_cache")
_EXPLAIN_CACHE_MAX = 50     # 最多 50 条（LRU 淘汰：按时间戳排序，删除最早）
_EXPLAIN_CACHE_TTL = 30 * 60  # 30 分钟过期


def _explain_cache_path() -> str:
    return os.path.join(_EXPLAIN_CACHE_DIR, "explain.json")


def _load_explain_cache() -> dict[str, Any]:
    """从磁盘加载 explain 缓存，过滤掉过期条目。"""
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
    """将 explain 缓存写入磁盘（原子写入：先写 .tmp 再 os.replace）。

    淘汰策略：按时间戳排序，保留最近 _EXPLAIN_CACHE_MAX 条。
    """
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
        os.replace(tmp, path)  # 原子替换，防止写一半时文件损坏
    except Exception:
        pass


# ── 院校统计数据加载（缓存）───────────────────────────────
@st.cache_resource(show_spinner=False)
def _get_school_stats() -> SchoolFeatureStats | None:
    """加载院校统计特征数据。

    @st.cache_resource：模型级缓存，所有用户共享同一实例。
    SchoolFeatureStats 用于计算学生在每个目标院校中的 GPA/语言/经历百分位。
    """
    try:
        from src.pages.prediction.page_data_loader import machine_learning_model

        mlm = machine_learning_model.resource_loader()
        return SchoolFeatureStats(mlm.cases_df)
    except Exception:
        return None


# ── 院校卡片渲染 ──────────────────────────────────────────
def _render_unified_school_cards(
    result: dict[str, Any] | None,
    unified: list[dict[str, Any]],
    percentile_map: dict[str, dict[str, Any]],
) -> None:
    """渲染统一院校分析卡片：每个目标院校一张卡片，含 AI 备注 + 概率 + 统计。"""
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


# ── 流式 JSON 部分提取 ────────────────────────────────────
# DeepSeek 流式返回 JSON chunks。为了渐进式渲染，需要从部分 JSON 中
# 提前提取已完成的字段。两个策略：
#   - 字符串字段（overview, summary）：正则匹配（快速，不处理转义边界）
#   - 数组字段（strengths, concerns, school_notes, products）：bracket counting
#     （逐字符扫描，处理字符串和转义，支持嵌套对象/数组）


def _extract_json_array(buffer: str, field: str) -> list | None:
    """从部分 JSON buffer 中提取指定字段的数组值。

    使用 bracket counting（括号计数）而非正则，正确处理：
    - 数组内嵌套对象/子数组
    - JSON 字符串内的 [ ] { }
    - 反斜杠转义

    Returns:
        成功解析的 list，或 None（字段不存在 / 数组未闭合 / 解析错误）。
    """
    import re

    # 定位字段名后的 ": [" 起始位置
    key_m = re.search(rf'"{field}"\s*:\s*', buffer)
    if not key_m:
        return None
    start = key_m.end()
    if start >= len(buffer) or buffer[start] != "[":
        return None

    depth = 0           # 括号嵌套深度（0 = 目标数组已闭合）
    in_string = False   # 是否在 JSON 字符串内
    escape = False      # 上一个字符是否为反斜杠
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
    return None  # 数组未闭合（还没收到完整数据）


def _try_extract_partial(buffer: str) -> dict[str, Any]:
    """从部分流式 JSON buffer 中提取所有可解析的字段。

    字符串字段用快速正则，数组/对象字段用 bracket counting。
    返回已成功解析的字段 dict，未就绪的字段不在 dict 中。
    """
    import re

    result: dict[str, Any] = {}
    # 字符串字段：快速正则匹配（假设值内不包含未转义的双引号）
    for field in ["overview", "summary"]:
        m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', buffer)
        if m:
            result[field] = re.sub(r"\\(.)", r"\1", m.group(1))
    # 数组字段：bracket counting
    for field in ["strengths", "concerns", "school_notes", "products"]:
        parsed = _extract_json_array(buffer, field)
        if parsed is not None:
            result[field] = parsed
    return result


# ── 缓存键构建 ────────────────────────────────────────────


def _compact_results(items: list[dict]) -> list[tuple[str, str, float]]:
    """压缩结果列表为 (university, major, probability) 三元组。

    只取前 N 条用于缓存键（避免长结果列表导致缓存键膨胀）。
    """
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
    """压缩经历详情为 {字段: 字符数}。

    缓存键不存经历全文（可能有几百字），只存字符数作为变化检测。
    """
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
    """构建 explain 缓存的 MD5 键。

    键由以下维度构成（覆盖 ExplainAgent 输出的所有决定因素）：
    - profile type（冲刺/稳妥/保底？）
    - top 5 similarity + top 3 cross + top 5 unified 结果
    - 背景专业、GPA、标化、语言、经历

    v=4: 缓存版本号，修改键构建逻辑时递增以强制失效旧缓存。
    """
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


# ── 流式 Explain 核心 ────────────────────────────────────
def _stream_explain_content(
    ctx: StudentContext,
    agent,
    percentile_data: dict | None = None,
    unified_results: list | None = None,
) -> tuple[dict[str, Any] | None, object]:
    """流式调用 ExplainAgent 并渐进式渲染。

    工作方式：
    1. 创建 st.empty() 占位符
    2. 迭代 agent.stream(ctx)，每次收到 chunk 追加到 buffer
    3. 每 25ms 或 6+ 新字符 → _try_extract_partial 尝试解析 buffer
    4. 解析成功的新字段 → render_ai_section_streaming 更新渲染
    5. 流式结束后用 parse_stream_result 做最终解析

    降级策略：
    - 流式异常 → 自动降级为同步 API 调用（agent.run）
    - 流式无输出 → 同上

    Returns:
        (parsed_result, stream_placeholder): 解析后的完整结果 + 占位符
    """
    stream_placeholder = st.empty()
    with stream_placeholder.container():
        render_ai_section_streaming("")  # 初始化空框架
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
            # 节流：25ms 内且新增不足 6 字符 → 跳过（减少 Streamlit 重渲染）
            if now - last_update < 0.025 and new_chars < 6:
                continue
            partial = _try_extract_partial(buffer)
            for key, val in partial.items():
                if val:
                    merged[key] = val
            new_seen = set(merged.keys()) - seen_fields  # 本轮新出现的字段
            seen_fields |= new_seen
            with stream_placeholder.container():
                render_ai_section_streaming(
                    merged,
                    seen_fields=frozenset(new_seen),  # 只传新字段（触发高亮动画）
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

    # 降级：流式无结果 → 同步 API 调用
    if result is None and (stream_error or not buffer.strip()):
        content_display_logger.info("降级到同步 API 调用")
        try:
            result = agent.run(ctx)
        except Exception:
            content_display_logger.exception("同步 API 降级也失败")
            result = None

    return result, stream_placeholder


# ── AI 解读编排器 ─────────────────────────────────────────
def _render_ai_explanation(
    prediction_results,
    input_data: dict[str, Any],
) -> None:
    """编排完整的 AI 解读流程。

    流程：
    1. 无结果 → 直接返回（无解读对象）
    2. 计算 profile type（冲刺/稳妥/保底）+ 百分位数据
    3. 构建缓存键 → 缓存命中直接渲染
    4. 缓存未命中 → 渲染"AI 解读"按钮（button gate）
    5. 用户点击 → 构建 StudentContext → ExplainAgent 流式生成
    6. 生成完成 → 缓存持久化 + 渲染结果
    7. st.rerun() 使按钮从"AI解读中..."恢复为可点击状态

    Button gate 模式：
      不自动调用 LLM（成本控制 + 用户主动触发），用户点击后才开始流式生成。
      生成中按钮 disabled，防止重复点击。
      生成完成后 st.rerun() → 下次渲染时缓存命中，按钮不再显示。
    """
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

    # 提前计算百分位数据（缓存命中时也需要渲染院校卡片）
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

    # L1 缓存初始化
    if "explain_cache" not in st.session_state:
        st.session_state["explain_cache"] = _load_explain_cache()

    cache_key = _build_explain_cache_key(
        sim=sim, cross=cross, unified=unified,
        background_major=background_major, gpa=gpa, gpa_raw=gpa_raw,
        exam_type=exam_type, exam_score=exam_score,
        language_score=language_score, language_type=language_type,
        experience_details=experience_details, profile=profile,
    )

    # 缓存命中：直接渲染，跳过 LLM 调用
    cached = st.session_state["explain_cache"].get(cache_key)
    if cached:
        products = render_static_frame(input_data, sim, cross, [])
        st.session_state["_ar_products"] = products
        render_ai_section(cached)
        _render_unified_school_cards(cached, unified, percentile_map)
        return

    # Button gate：用户主动触发 AI 解读
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

    # 流式生成
    stream_placeholder = st.empty()
    try:
        products = render_static_frame(input_data, sim, cross, [])
        matched_products = products

        ctx = StudentContext(
            stage="match",
            background_university=background_university,
            background_major=background_major,
            gpa=gpa, gpa_raw=gpa_raw,
            standardized_test_type=exam_type, standardized_test_score=exam_score,
            language_score=language_score, language_score_raw=language_score_raw,
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

    st.rerun()  # 重跑使按钮恢复可点击 + 缓存生效


# ── 公共 API ──────────────────────────────────────────────
def display_content(
    session_manager: SessionManager,
    page_state: Any,
    submitted: bool,
    session_key_has_predicted: str = DEFAULT_SESSION_KEYS.has_predicted,
    session_key_input_data: str = DEFAULT_SESSION_KEYS.input_data,
    session_key_predict_lock: str = DEFAULT_SESSION_KEYS.predict_lock,
    session_key_form_data_changed: str = DEFAULT_SESSION_KEYS.form_data_changed,
) -> None:
    """预测结果展示入口（hk.py → display_content 的唯一调用点）。

    职责：
    1. 检查 has_predicted 状态（无结果 → 不渲染）
    2. 检测状态不一致（has_predicted=True 但 input_data 丢失 → 重置）
    3. 检测表单变更（form_changed → 提示用户结果可能过时）
    4. 检查并渲染 fallback 通知
    5. 渲染结果表格（display_results_section）
    6. 触发 AI 解读（_render_ai_explanation）

    Args:
        submitted: 是否为新提交（True=首次展示含动画，False=回显既往结果）
    """
    if not session_manager.get(session_key_has_predicted, False):
        return

    current_input_data = session_manager.get(session_key_input_data)
    if not current_input_data:
        # 状态不一致：可能来自 session 序列化问题或手动清除
        content_display_logger.warning("has_predicted 为 True，但 session_state 中缺少输入数据。")
        reset_prediction_results(session_manager)
        session_manager.set(**{session_key_has_predicted: False, session_key_predict_lock: False})
        st.rerun()

    form_changed = session_manager.get(session_key_form_data_changed, False)
    if not submitted and form_changed:
        st.caption("您的输入已更改，当前显示的是先前输入的预测结果。请点击预测按钮获取最新结果。")

    res_model = session_manager.get(DEFAULT_UI_KEYS.prediction_results)

    _check_and_render_fallback_notice(res_model)

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

    # 消费 form_changed flag：提示已展示，下一轮不再重复提示
    if not submitted and form_changed:
        session_manager.set(**{session_key_form_data_changed: False})
