from __future__ import annotations

from typing import Any

from src.pages.prediction.core.utils import denormalize_language_score

PIPELINE_PHASE_MAP = {
    "init_engine": "prep",
    "wake_model": "prep",
    "check_consistency": "prep",
    "verify_data": "prep",
    "build_features": "prep",
    "search_cases": "match",
    "prepare_pool": "match",
    "load_similarity": "match",
    "extract_profile": "match",
    "running_calc": "infer",
    "initial_filter": "infer",
    "analyze_text": "infer",
    "merging": "deliver",
    "done": "deliver",
    "empty_results": "deliver",
}

PHASE_LABELS = {
    "prep": "初始化：加载模型、校验数据一致性、清洗输入并构造特征向量。",
    "match": "检索：相似案例召回、候选池构建、加载相似度缓存、抽取申请人表征。",
    "infer": "推理：批量概率计算、初筛、软背景文本特征与规则化权重调整。",
    "deliver": "汇总：合并相似/跨专业/指定志愿多路结果，去重排序并输出。",
}


def _shorten_program_label(text: str, max_len: int = 36) -> str:
    s = str(text).strip()
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def clip_join_labels(
    items: list[Any],
    max_items: int = 4,
    sep: str = "、",
    *,
    shorten_each: int | None = None,
) -> str:
    xs = [str(x).strip() for x in items if str(x).strip()]
    if not xs:
        return "—"
    head = xs[:max_items]
    parts = [_shorten_program_label(x, shorten_each) if shorten_each else x for x in head]
    out = sep.join(parts)
    if len(xs) > max_items:
        out += "…"
    return out


def _format_language_display(language_type: str | None, language_score: float | None) -> str:
    t = (language_type or "").strip()
    if language_score is None:
        return t or "—"
    try:
        x = float(language_score)
    except (TypeError, ValueError):
        return f"{t} {language_score}".strip() if t else str(language_score)
    if t in ("雅思", "托福") and 0.0 <= x <= 1.05:
        raw = denormalize_language_score(x, t, round_to_half=True)
        if t == "托福":
            disp = int(raw) if abs(raw - round(raw)) < 1e-6 else round(raw, 1)
            return f"{t} {disp}"
        disp = int(raw) if abs(raw - round(raw)) < 1e-6 else raw
        return f"{t} {disp}"
    if abs(x - round(x)) < 1e-9:
        return f"{t + ' ' if t else ''}{int(round(x))}".strip()
    s = f"{x:.3f}".rstrip("0").rstrip(".")
    return f"{t + ' ' if t else ''}{s}".strip() or "—"


def format_pipeline_prep_progress(
    *,
    bg_university: str | None,
    bg_major: str | None,
    language_type: str | None,
    language_score: float | None,
    target_universities: list[Any] | None,
    similarity_cache_loaded: bool,
) -> str:
    tu = target_universities or []
    scope = clip_join_labels(tu, 4) if tu else "未限定（全库）"
    cache = "相似度缓存就绪" if similarity_cache_loaded else "无相似度缓存"
    return (
        f"准备：{bg_university or '—'} / {bg_major or '—'}，"
        f"标化 {_format_language_display(language_type, language_score)}，"
        f"志愿 {scope}，{cache}"
    )


def format_pipeline_compute_progress(
    combinations: list[tuple[str, str]],
    hints: dict[str, Any],
) -> str:
    user_locked = bool(hints.get("user_locked_majors"))
    mode = "指定专业池" if user_locked else "匹配专业池（语义+模糊）"
    unis = clip_join_labels(hints.get("target_unis") or [], 4)
    majors = clip_join_labels(
        hints.get("target_majors") or [],
        3,
        shorten_each=34,
    )
    preview = _preview_combination_pairs(combinations, max_pairs=3)
    return f"推理：{mode}，院校 {unis}，专业 {majors}，示例 {preview}"


def _preview_combination_pairs(pairs: list[tuple[str, str]], max_pairs: int = 3) -> str:
    out: list[str] = []
    used_pairs: set[tuple[str, str]] = set()
    seen_uni: set[str] = set()

    for u, m in pairs:
        uu, mm = str(u).strip(), str(m).strip()
        if not uu or not mm:
            continue
        k = (uu, mm)
        if k in used_pairs or uu in seen_uni:
            continue
        used_pairs.add(k)
        seen_uni.add(uu)
        out.append(f"{uu} / {_shorten_program_label(mm, 32)}")
        if len(out) >= max_pairs:
            return "、".join(out)

    for u, m in pairs:
        uu, mm = str(u).strip(), str(m).strip()
        k = (uu, mm)
        if not uu or not mm or k in used_pairs:
            continue
        used_pairs.add(k)
        out.append(f"{uu} / {_shorten_program_label(mm, 32)}")
        if len(out) >= max_pairs:
            break
    return "、".join(out) if out else "—"


def format_pipeline_refine_progress(
    *,
    route_labels: list[str],
    bg_faculty: str | None,
    soft_background_on: bool,
) -> str:
    routes = "、".join(route_labels) if route_labels else "—"
    fac = (bg_faculty or "").strip() or "—"
    sb = "已纳入经历文本" if soft_background_on else "未纳入经历文本"
    return f"校准：分路 {routes}，背景学部 {fac}，{sb}"


def format_pipeline_done_progress() -> str:
    return "分析完成，请查看推荐结果"


def format_pipeline_empty_progress(
    *,
    bg_major: str | None,
    target_universities: list[Any] | None,
) -> str:
    tu = target_universities or []
    scope = clip_join_labels(tu, 3) if tu else "未限定"
    return f"当前无推荐结果：背景 {bg_major or '—'}，志愿 {scope}"


PIPELINE_MESSAGES = {
    "cross_check": ["跨学部/跨专业一致性校验中"],
    "empty_results": ["当前条件下无匹配结果"],
}

EXPERIENCE_ITEM_NAMES = {
    "research_details": "科研积淀",
    "internship_details": "实习经历",
    "award_details": "荣誉奖项",
    "paper_details": "学术产出",
}

EXPERIENCE_BOOST_TEMPLATE = "已检测到经历字段：{items}，纳入软背景特征"
EXPERIENCE_DEFAULT_MSG = "校验软背景文本字段"

FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

EXPERIENCE_ANALYSIS_MESSAGES = [
    "解析{field}文本",
    "{field}字段校验",
]

EXPERIENCE_VALIDATION_TEMPLATE = [
    "校验{field_name}（{idx}/{total}）",
]

RANKER_MESSAGES = {
    "basic": [
        "{tone}，概率测算：{target_major}",
    ],
    "cross_major": [
        "{tone}，疑似跨专业：{background_major_ori} → {target_major}",
    ],
    "faculty": [
        "{tone}，学部权重：{target_major}",
    ],
    "relax": [
        "{tone}，放宽边界：{target_major}",
    ],
    "tighten": [
        "{tone}，收紧阈值：{target_major}",
    ],
    "fallback": [
        "{tone}，综合排序：{target_major}",
    ],
}
