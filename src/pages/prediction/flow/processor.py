"""
预测结果处理器 — 候选组合生成 + 结果分路 + Agent 平衡调整。

这是 run_prediction.py 的直接下游，负责将原始 XGBoost 输出转换为
三路结构化结果（similarity / cross_major / user_specified）。

两个核心函数：
  generate_prediction_combinations — 构建 (院校, 专业) 候选池
  process_prediction_results — 原始推理结果 → 三路分类 + Agent 平衡

候选池构建策略：
  1. 用户指定了专业 → 直接用用户指定的组合
  2. 用户未指定 → 从全量目标专业中筛选"相关"的
     相关 = 语义相似度 ≥ 0.89 (COMBINATION_POOL_SEMANTIC_MIN)
          OR fuzzy token_sort_ratio > 50 (COMBINATION_POOL_FUZZY_MIN)
          OR 在 hot_paths.json 中配置的热门专业（如会计金融分析、信息技术）
  3. 最后确保每个目标院校至少有一个组合（_ensure_per_university_coverage）

Agent 平衡调整（_apply_agent_balance_adjustment_flat）：
  当 similarity 和 cross_major 两路结果数量差距过大时，
  用 BoundaryCaseAgent 重新评估相似路中处于边界的结果，
  把"其实不算相似"的从 sim 移到 cross。
"""

import math
from typing import Any

import pandas as pd
from rapidfuzz import fuzz

from src.agent.boundary_case_agent import BoundaryCaseAgent  # 边界案例重评估（sim vs cross 分类）
from src.pages.prediction.core.types import PredictionInput
from src.pages.prediction.core.utils import (
    _data_manager,                  # 院校-专业数据管理器单例
    denormalize_language_score,     # 归一化语言分数 → 原始分（用于 UI 展示）
)
from src.pages.prediction.flow.result_processor import SingleResultProcessor  # 单条结果处理器
from src.pages.prediction.result_modifier.config import (
    AGENT_MIN_BALANCE_DIFF_MIN,         # Agent 调整的最小绝对差阈值
    AGENT_MIN_BALANCE_DIFF_RATIO,       # Agent 调整的最小比例阈值（相对 max(sim, cross)）
    AGENT_NO_CHANGE_THRESHOLD,          # cross 数量低于此值时不触发调整
    COMBINATION_POOL_FUZZY_MIN,         # 模糊匹配最低分（fuzz.token_sort_ratio）
    COMBINATION_POOL_SEMANTIC_MIN,      # 语义匹配最低分（相似度缓存中的值）
    HIGHER_SIMILARITY_THRESHOLD,        # 院校数少时的更高相似度阈值
    MIN_SIMILARITY_THRESHOLD,           # 基础相似度阈值（0.89）
    UNIVERSITY_COUNT_THRESHOLD,         # 院校数阈值（区分高低相似度标准）
    USER_SPECIFIED_LARGE_RANGE_TOP_N,   # 用户指定范围大时的截断数
    USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD,  # 用户指定范围的"中等"阈值
)
from src.pages.prediction.result_modifier.faculty_filters import (
    get_allowed_target_faculties,   # 获取允许的目标学部（用于学部过滤）
)
from src.pages.prediction.result_modifier.filters import (
    get_cross_major_recommendations,    # 跨专业推荐结果提取
    get_similar_major_recommendations,  # 相似专业推荐结果提取
)
from src.pages.prediction.result_modifier.ranker import adjust_similarity_results_with_agent  # Agent 辅助排序

# 热门专业子串：即使相似度低也纳入候选池（来自 hot_paths.json）
_HOT_MAJOR_SUBSTRINGS: tuple | None = None


def _load_hot_paths():
    """延迟加载热门专业关键词（hot_paths.json），失败时使用硬编码默认值。

    热门专业无论相似度如何都纳入候选池，避免遗漏高频申请目标。
    """
    global _HOT_MAJOR_SUBSTRINGS
    if _HOT_MAJOR_SUBSTRINGS is not None:
        return
    import json
    from pathlib import Path

    cfg_path = Path(__file__).parent.parent.parent.parent.parent / "config" / "hot_paths.json"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        _HOT_MAJOR_SUBSTRINGS = tuple(cfg.get("hot_major_substrings", []))
    except Exception:
        _HOT_MAJOR_SUBSTRINGS = (
            "smart manufacturing",
            "accounting and finance analytics",
            "information technology",
        )


def _is_major_match(
    m_lower: str,
    bg_mapped: str,
    bg_orig: str,
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
) -> bool:
    """判断目标专业是否与背景专业"相关"。

    三级匹配（任一命中即返回 True）：
    1. 热门专业关键词匹配（无条件纳入候选池）
    2. 语义相似度 ≥ 0.89（预计算的相似度矩阵）
    3. fuzzy token_sort_ratio > 50（字符级模糊匹配兜底）
    """
    _load_hot_paths()
    if any(kw in m_lower for kw in _HOT_MAJOR_SUBSTRINGS):
        return True
    sim = float((bg_target_similarity_cache or {}).get((bg_mapped, m_lower), 0.0))
    if sim >= COMBINATION_POOL_SEMANTIC_MIN:
        return True
    return fuzz.token_sort_ratio(bg_orig, m_lower) > COMBINATION_POOL_FUZZY_MIN


def _resolve_prediction_target_lists(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[str], list[str]]:
    """解析目标院校和目标专业列表。

    专业解析逻辑：
    - 用户指定了 target_majors → 直接用（用户意图优先）
    - 用户未指定 → 从全量目标专业中筛选"相关"的
      → 一个都不匹配 → 回退到全量（不过滤）

    回退的原因：如果用户选了 10 个目标专业但一个都没匹配上，
    与其返回空，不如全量推理让用户自己看。
    """
    target_unis = list(input_data.get("target_universities") or all_universities_target)
    user_specified_majors = input_data.get("target_majors")

    if not user_specified_majors:
        bg_mapped = str(input_data.get("background_major", "")).strip().lower()
        target_majors: list[str] = []
        bg_orig = str(background_major_original or "").strip().lower()

        for m in all_majors_target:
            m_str = str(m).strip()
            if not m_str:
                continue
            if _is_major_match(m_str.lower(), bg_mapped, bg_orig, bg_target_similarity_cache):
                target_majors.append(m)

        # 回退：一个都不匹配 → 全量
        if not target_majors:
            target_majors = list(all_majors_target)
    else:
        target_majors = list(user_specified_majors)

    return target_unis, target_majors


def _count_fuzz_passing_majors(
    input_data: PredictionInput,
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
    background_major_original: str | None,
) -> int:
    """统计通过匹配筛选的专业数量（用于进度元数据，不做实际过滤）。"""
    bg_mapped = str(input_data.get("background_major", "")).strip().lower()
    bg_orig = (
        str(background_major_original or input_data.get("background_major") or "").strip().lower()
    )
    if not bg_orig:
        return 0
    n = 0
    seen: set[str] = set()
    for m in all_majors_target:
        m_str = str(m).strip()
        if not m_str:
            continue
        m_lower = m_str.lower()
        if m_lower in seen:
            continue
        if _is_major_match(m_lower, bg_mapped, bg_orig, bg_target_similarity_cache):
            seen.add(m_lower)
            n += 1
    return n


def prediction_progress_scope_meta(
    input_data: PredictionInput,
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None,
    background_major_original: str,
) -> tuple[int | None, int | None, bool]:
    """计算预测范围的元数据（用于进度 UI 展示院校数/专业数/用户锁定状态）。"""
    tu = input_data.get("target_universities") or []
    utrim = [str(u).strip() for u in tu if str(u).strip()]
    n_uni = len(set(utrim)) if utrim else None

    tm = input_data.get("target_majors") or []
    mtrim = [str(m).strip() for m in tm if str(m).strip()]
    if mtrim:
        return n_uni, len(set(mtrim)), True

    fuzz_n = _count_fuzz_passing_majors(
        input_data, all_majors_target, bg_target_similarity_cache, background_major_original,
    )
    if fuzz_n > 0:
        return n_uni, fuzz_n, False
    return n_uni, None, False


def _enumerate_valid_combinations(
    target_unis: list[str],
    target_majors: list[str],
) -> list[tuple[str, str]]:
    """生成所有有效的 (院校, 专业) 笛卡尔积组合。

    有效性检查：
    - 院校在 _data_manager.valid_universities 中
    - 专业在 _data_manager.valid_majors 中
    - (院校, 专业) 在 _data_manager.valid_combinations 中

    三层过滤确保只推理有详情数据的组合。
    """
    valid_unis = _data_manager.valid_universities
    valid_majors = _data_manager.valid_majors
    valid_set = _data_manager.valid_combinations
    return [
        (u, m)
        for u in target_unis
        if u in valid_unis
        for m in target_majors
        if m in valid_majors and (u, m) in valid_set
    ]


def _ensure_per_university_coverage(
    combos: list[tuple[str, str]],
    target_unis: list[str],
    target_majors: list[str],
) -> list[tuple[str, str]]:
    """确保每个目标院校至少有一个组合（"不漏学校"）。

    场景：用户选了 5 个院校，但 _resolve_prediction_target_lists 的专业过滤
    可能让某些院校落下（该院校的所有专业都没通过三级匹配）。
    此函数为每个缺失的院校补一个最常见的有效专业组合。

    策略：
    - 优先从 target_majors 中选（与用户意图相关）
    - 否则从全量有效专业中取第一个
    """
    covered = {u for u, _ in combos}
    missing = [u for u in target_unis if u not in covered and u in _data_manager.valid_universities]
    if not missing:
        return combos

    valid_set = _data_manager.valid_combinations
    result = list(combos)
    for u in missing:
        candidates = [m for m in target_majors if m in _data_manager.valid_majors and (u, m) in valid_set]
        if not candidates:
            candidates = [m for m in _data_manager.valid_majors if (u, m) in valid_set]
        if candidates:
            result.append((u, candidates[0]))
    return result


def generate_prediction_combinations(
    input_data: PredictionInput,
    all_universities_target: list[str],
    all_majors_target: list[str],
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
    background_major_original: str | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """生成预测候选组合列表 — 候选池构建的入口。

    三步：
    1. _resolve_prediction_target_lists：确定 target_unis + target_majors
    2. _enumerate_valid_combinations：笛卡尔积 + 有效性过滤
    3. _ensure_per_university_coverage：补漏

    Returns:
        (combinations, meta) where combinations 是 [(院校, 专业), ...]
    """
    target_unis, target_majors = _resolve_prediction_target_lists(
        input_data, all_universities_target, all_majors_target,
        bg_target_similarity_cache, background_major_original,
    )
    res = _enumerate_valid_combinations(target_unis, target_majors)
    res = _ensure_per_university_coverage(res, target_unis, target_majors)
    return res, {
        "combination_count": len(res),
        "progress_hints": {
            "target_unis": [str(u).strip() for u in target_unis if str(u).strip()],
            "target_majors": [str(m).strip() for m in target_majors if str(m).strip()],
            "user_locked_majors": bool(input_data.get("target_majors")),
        },
    }


def count_cases_with_similar_background(
    cases_df: pd.DataFrame | None,
    background_major: str,
    background_major_original: str,
    bg_target_similarity_cache: dict[tuple[str, str], float] | None = None,
) -> int:
    """统计历史数据中类似背景专业的案例数。

    用于判断是否有足够的历史数据支撑该背景专业的预测。
    匹配逻辑与 _is_major_match 一致（语义 + 模糊）。
    全不匹配时回退到精确匹配计数。
    """
    if cases_df is None or cases_df.empty or "background_major" not in cases_df.columns:
        return 0

    bg_mapped = str(background_major or "").strip().lower()
    bg_orig = str(background_major_original or background_major or "").strip().lower()
    if not bg_orig:
        return 0

    vc = cases_df["background_major"].astype(str).str.strip().value_counts(dropna=False)
    total = 0
    for maj_val, cnt in vc.items():
        m_lower = str(maj_val).strip().lower()
        if not m_lower:
            continue
        sim_score = 0.0
        if bg_target_similarity_cache is not None and bg_mapped:
            sim_score = float(bg_target_similarity_cache.get((bg_mapped, m_lower), 0.0))
        if sim_score >= COMBINATION_POOL_SEMANTIC_MIN:
            total += int(cnt)
            continue
        if fuzz.token_sort_ratio(bg_orig, m_lower) > COMBINATION_POOL_FUZZY_MIN:
            total += int(cnt)

    # 回退到精确匹配
    if total == 0 and bg_mapped:
        mask = cases_df["background_major"].astype(str).str.strip().str.lower() == bg_mapped
        return int(mask.sum())
    return total


def _get_user_specified_results(
    results: list,
    user_specified_combinations: list[tuple[str, str]] | None,
    allow_degraded: bool = True,
) -> list:
    """从所有结果中提取用户指定组合对应的结果。

    当用户范围很大时（>USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD），
    截断到 TOP_N 以避免结果页过长。
    """
    if not user_specified_combinations or not results:
        return []

    specified_set = set(user_specified_combinations)
    specified_results = [
        res
        for res in results
        if (res.get("university"), res.get("major")) in specified_set
        and (allow_degraded or res.get("_is_in_faculty_scope", True))
    ]

    if not specified_results:
        return []

    specified_results.sort(key=lambda x: x.get("probability", 0), reverse=True)

    if len(user_specified_combinations) <= USER_SPECIFIED_MEDIUM_RANGE_THRESHOLD:
        return specified_results

    return specified_results[:USER_SPECIFIED_LARGE_RANGE_TOP_N]


def process_prediction_results(
    results: list,
    background_major: str,
    bg_target_similarity_cache: dict,
    num_target_universities: int,
    cases_df: pd.DataFrame | None = None,
    user_specified_combinations: list[tuple[str, str]] | None = None,
    background_faculty: str | None = None,
    background_major_original: str | None = None,
    allow_degraded_user_specified: bool = False,
    probability_adjuster: Any | None = None,
    gpa: float | None = None,
    language_score: float | None = None,
    language_type: str | None = None,
    background_university: str | None = None,
    progress_reporter: Any | None = None,
    agent: Any | None = None,
    admitted_combinations: set[tuple[str, str]] | None = None,
) -> tuple[list, list, list]:
    """XGBoost 推理结果 → 三路结构化输出。

    处理流程：
    1. SingleResultProcessor 逐条处理原始输出（语言反归一化、学部范围标记等）
    2. _get_user_specified_results 提取用户指定组合
    3. get_similar_major_recommendations 提取相似专业推荐
    4. get_cross_major_recommendations 提取跨专业推荐
    5. _apply_agent_balance_adjustment_flat 用 BoundaryCaseAgent 平衡两路数量

    Returns:
        (similarity_results, cross_major_results, user_specified_results)
    """
    if not results:
        return [], [], []

    # 语言分数反归一化（用于 UI 展示原始分数）
    lang_type = language_type or "雅思"
    raw_lang = (
        denormalize_language_score(language_score, lang_type)
        if language_score is not None
        else None
    )

    # 学部范围限制（跨学部确认后放宽）
    allowed_faculties = (
        get_allowed_target_faculties(background_faculty)
        if background_faculty and not allow_degraded_user_specified
        else set()
    )

    result_processor = SingleResultProcessor(
        data_manager=_data_manager,
        bg_major=str(background_major or "").strip(),
        bg_major_orig=str(background_major_original or "").strip(),
        bg_orig_lower=str(background_major_original or background_major or "").strip().lower(),
        raw_lang=raw_lang,
        lang_type=lang_type,
        bg_target_similarity_cache=bg_target_similarity_cache,
        allowed_faculties=allowed_faculties,
        background_faculty=background_faculty,
    )

    # 逐条处理（walrus operator 过滤掉 process 返回 None 的无效结果）
    processed_results = [pr for r in results if (pr := result_processor.process(r))]

    if not processed_results:
        return [], [], []

    user_results = _get_user_specified_results(
        processed_results, user_specified_combinations, allow_degraded=allow_degraded_user_specified
    )
    # res_for_rec = 在学部范围内的结果（仅用于 sim/cross 推荐）
    res_for_rec = [r for r in processed_results if r.get("_is_in_faculty_scope", True)]

    bg_major_param = background_major_original or background_major

    # 相似专业推荐
    sim_rec = get_similar_major_recommendations(
        res_for_rec, num_target_universities,
        probability_adjuster=probability_adjuster,
        gpa=gpa, language_score=language_score,
        background_university=background_university,
        background_major=bg_major_param,
    )

    # 跨专业推荐
    cross_rec = get_cross_major_recommendations(
        res_for_rec, background_major, cases_df, background_faculty,
        probability_adjuster=probability_adjuster,
        gpa=gpa, language_score=language_score,
        background_university=background_university,
        admitted_combinations=admitted_combinations,
    )

    # Agent 平衡：两路结果数量差距过大时用 LLM 重评估边界案例
    sim_rec, cross_rec = _apply_agent_balance_adjustment_flat(
        sim_rec, cross_rec, processed_results, bg_major_param,
        background_faculty, num_target_universities, cases_df,
        progress_reporter,
        is_cross_faculty=allow_degraded_user_specified,
        agent=agent,
    )

    return sim_rec, cross_rec, user_results


def _apply_agent_balance_adjustment_flat(
    sim_rec: list,
    cross_rec: list,
    all_results: list,
    background_major: str,
    background_faculty: str | None,
    num_unis: int,
    cases_df: pd.DataFrame | None,
    reporter: Any | None,
    is_cross_faculty: bool = False,
    agent: Any | None = None,
) -> tuple[list, list]:
    """用 BoundaryCaseAgent 平衡 sim 和 cross 两路结果的数量。

    触发条件：
    1. |sim| - |cross| 的差大于阈值（max(5, 0.35×max(|sim|, |cross|))）
    2. cross 数量不低于 AGENT_NO_CHANGE_THRESHOLD（1）

    调整逻辑：
    - diff > 0（sim 过多）→ 从 sim 中挑选边界案例，移入 cross
    - diff < 0（cross 过多）且 cross 足够多 → 从 sim 的相似度边界处扩展

    相似度阈值：
    - 院校数 ≤ UNIVERSITY_COUNT_THRESHOLD：使用 HIGHER_SIMILARITY_THRESHOLD（更严格）
    - 院校数多：使用 MIN_SIMILARITY_THRESHOLD（0.89）
    """
    diff = len(cross_rec) - len(sim_rec)
    max_len = max(len(sim_rec), len(cross_rec))
    value = AGENT_MIN_BALANCE_DIFF_RATIO * max_len
    threshold = max(AGENT_MIN_BALANCE_DIFF_MIN, math.ceil(value))

    if abs(diff) < threshold or cases_df is None or not background_major:
        return sim_rec, cross_rec

    # cross 太少时不触发
    if diff < 0 and len(cross_rec) < AGENT_NO_CHANGE_THRESHOLD:
        return sim_rec, cross_rec

    if agent is None:
        agent = BoundaryCaseAgent(cases_df=cases_df)

    limit = (
        HIGHER_SIMILARITY_THRESHOLD
        if 0 < num_unis <= UNIVERSITY_COUNT_THRESHOLD
        else MIN_SIMILARITY_THRESHOLD
    )

    sim_rec = adjust_similarity_results_with_agent(
        sim_rec, all_results, diff, background_major, limit,
        agent, background_faculty,
        progress_reporter=reporter, is_cross_faculty=is_cross_faculty,
    )

    # 去重：从 cross 中移除已在 sim 中的组合
    sim_keys = {(r.get("university"), r.get("major")) for r in sim_rec}
    cross_rec = [r for r in cross_rec if (r.get("university"), r.get("major")) not in sim_keys]

    return sim_rec, cross_rec
