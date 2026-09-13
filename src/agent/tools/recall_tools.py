"""召回/统计类只读工具（第二波）。

## 为什么只做这一件事

系统里有**三个语义不同的"概率"**：

1. **历史基准率** —— 只看「目标院校 + 目标专业」，完全不看这个学生是谁（本模块暴露的）
2. **校准基模型概率** —— `PredictionModel.predict_batch`，且未知本科校/专业时会调 LLM 并**写盘**
3. **页面上最终展示的录取概率** —— ① + 个人条件 + GPA/语言/跨专业惩罚链 + 文本加成 + tier 修复

**没有任何轻量入口能复现 ③。** 所以本模块只暴露 ①，并且强制带 `source` 与 `caveat`：
不标注来源，AI 会说"这个专业 30%"，而页面显示 18%，用户立刻发现对不上。

## 实测成本与约束（决定了实现方式）

- tool 跑在 worker 线程：`@st.cache_data` 在那里**不生效**（`load_raw_cases_data()` 每次重读 21MB ≈ 147ms），
  而 `@st.cache_resource` 生效（`get_knn_stats()` 热 0.05ms）。
  → 本模块用**进程级单例缓存"派生统计量"**，算完就把原始 DataFrame 丢掉，绝不常驻 121MB。
- 样本量必须分级：1518 个组合里 **373 个 n<5**、只有 617 个 n≥30。
  n<5 给比例就是编数字。

## 明确不暴露（以及为什么）

- `PredictionModel.predict_batch`：未知校/专业会触发 LLM + 写 `cache/agent_cache/*.json`，延迟不可控。
- `retrieve_similar_cases` 的**原始返回**：含真实学生案例（隐私）→ 只暴露聚合，见
  `lookup_similar_admitted_cases`。
- `run_prediction_pipeline_with_progress`：内含 LLM 文本加成 + 边界案例 agent + 300s 池超时，会打爆单轮预算。
- "某专业录取 GPA 分位数"现算：654ms 且大量分组 n<5，统计上不可用。

## 一个必须记住的语义陷阱：KNN 候选池只有"已录取"案例

实测：`cases.feather` 全库录取率 **31.8%**（n=81,830），
而 `get_knn_stats()["pool"]` 是 **25,499 行、录取率 100%** —— 它是**成功案例展示库**，不是统计样本。

所以 `retrieve_similar_cases` **不能用来算录取率**：一不小心就会对用户说
"和你像的人 100% 被录取"。`lookup_similar_admitted_cases` 因此只提供**存在性证据**
（"有像你这样的人申上过"），返回体里带 `cases_are_admitted_only: true` 和明确禁止算率的 caveat。
"""

from __future__ import annotations

import json
import logging
import threading
from functools import lru_cache
from typing import Any

from pydantic_ai import RunContext

from src.agent.harness import HarnessDeps
from src.agent.tools.probe_tools import _resolve_school
from src.agent.tools.tool_trace import traced

_log = logging.getLogger("lead_in_recall")

# 样本量分级门槛：低于下限绝不给比例。
_MIN_N_FOR_RATE = 5
_MIN_N_FOR_CONFIDENT = 30
# 只知学校不知专业时，按样本量倒序回传多少个候选。
_MAX_MAJOR_LIST = 15

_CAVEAT = (
    "这是该「目标院校 + 目标专业」的**历史录取率**，只统计院校与专业，"
    "**不含这个学生的 GPA、语言成绩、本科背景**。"
    "页面上最终展示的录取概率会在这之上叠加个人条件与惩罚链，口径不同，不要互相引用。"
)

_CACHE_LOCK = threading.Lock()
_ADMIT_INDEX: dict[str, dict[str, tuple[float, int]]] | None = None


def _admit_index() -> dict[str, dict[str, tuple[float, int]]]:
    """学校 → {英文专业名: (录取率, 样本数)}，进程级单例。

    只缓存这个派生小字典（1518 条），原始 cases_df（81,830 行 / ~121MB）算完即弃。
    不要改成缓存 DataFrame —— tool 线程里 `@st.cache_data` 不生效，常驻两份大表不值得。
    """
    global _ADMIT_INDEX
    if _ADMIT_INDEX is None:
        with _CACHE_LOCK:
            if _ADMIT_INDEX is None:
                from src.adjustment.admission_cache import get_baseline_admit_lookup
                from src.pages.prediction.app_data import load_raw_cases_data

                cases_df = load_raw_cases_data()
                flat = get_baseline_admit_lookup(cases_df)
                index: dict[str, dict[str, tuple[float, int]]] = {}
                for (uni, major), (rate, n) in flat.items():
                    index.setdefault(str(uni), {})[str(major)] = (float(rate), int(n))
                _ADMIT_INDEX = index
                _log.info(
                    "ADMIT_INDEX | schools=%d combos=%d",
                    len(index),
                    sum(len(v) for v in index.values()),
                )
    return _ADMIT_INDEX


@lru_cache(maxsize=1)
def _all_combo_majors() -> dict[str, tuple[str, ...]]:
    """从 admit 索引倒出"每校有哪些专业"，用于专业名兜底解析。"""
    idx = _admit_index()
    return {school: tuple(sorted(majors)) for school, majors in idx.items()}


def _resolve_combo_major(school: str, raw: str) -> tuple[str, float, str]:
    """把专业名对齐到 admit 索引的键空间。

    优先复用 `school_major_details` 的中文名索引（覆盖约 89%），
    再对 admit 索引自己的专业名做模糊匹配兜底。
    """
    from src.agent.bridges.fuzzy_match import _fuzzy_match
    from src.agent.tools.probe_tools import _resolve_major

    candidates = _all_combo_majors().get(school, ())
    if not candidates:
        return "", 0.0, ""
    if raw in candidates:
        return raw, 1.0, "en"

    program, conf, lang = _resolve_major(school, raw)
    if program in candidates:
        return program, conf, lang

    matched, fallback_conf = _fuzzy_match(raw, list(candidates))
    if fallback_conf >= conf:
        return str(matched or raw), float(fallback_conf), "en"
    return program, conf, lang


def _sample_payload(rate: float, n: int) -> dict[str, Any]:
    """按样本量分级给出可负责的结论。"""
    from src.utils.numeric import wilson_score_ci

    payload: dict[str, Any] = {"sample_size": n}
    if n < _MIN_N_FOR_RATE:
        payload.update(
            {
                "admit_rate": None,
                "ci95": None,
                "reliable": False,
                "note": f"样本量 n={n} 太少（门槛 {_MIN_N_FOR_RATE}），不给录取率"
                "—— 给出来就是编数字。可以只告诉用户'这个方向我们的参考样本很少'。",
            }
        )
        return payload

    admitted = int(round(rate * n))
    lo, hi = wilson_score_ci(admitted, n)
    payload.update(
        {
            "admit_rate": round(rate, 3),
            "admitted": admitted,
            "ci95": [round(lo, 3), round(hi, 3)],
            "reliable": n >= _MIN_N_FOR_CONFIDENT,
        }
    )
    if n < _MIN_N_FOR_CONFIDENT:
        payload["note"] = (
            f"样本量 n={n} 偏少（稳健门槛 {_MIN_N_FOR_CONFIDENT}），区间很宽，"
            "只能作为大致参考，回答时要说明样本有限。"
        )
    return payload


@traced
def lookup_admission_stats(
    ctx: RunContext[HarnessDeps], university: str, major: str = ""
) -> str:
    """查某个「目标院校 + 专业」的历史录取率与参考样本量。

    用于回答「这个专业好不好申 / 录取率多少 / 我有没有戏」这类问题里**客观的那一半**：
    该专业的历史录取比例、样本量、95% 置信区间。

    **它不包含学生个人的 GPA / 语言 / 背景**，所以不要把它当成"你的录取概率"。
    要讲个人匹配度，只能说"结合你的条件还需要页面上的预测结果来判断"。

    Args:
        university: 目标院校，中文校名（如"香港大学""港大"）。
        major: 目标专业，中文或英文都可以（如"金融"或 "Master of Finance"）。
            留空会返回该校样本量最多的若干个专业供挑选。
    """
    try:
        school, school_conf = _resolve_school(university)
        index = _admit_index()
        combos = index.get(school, {})

        if not combos:
            return _json(
                {
                    "found": False,
                    "reason": "unknown_school",
                    "input_university": university,
                    "resolved_university": school,
                    "match_confidence": round(school_conf, 2),
                    "hint": "录取统计里没有这所目标院校；不要对不在库中的院校给录取率。",
                    "_source": "cases.feather:admission_rate",
                }
            )

        if not major:
            top = sorted(combos.items(), key=lambda kv: kv[1][1], reverse=True)[:_MAX_MAJOR_LIST]
            return _json(
                {
                    "found": False,
                    "reason": "major_required",
                    "resolved_university": school,
                    "available_majors": [m for m, _ in top],
                    "available_major_count": len(combos),
                    "hint": "请指定一个具体专业再查；上面按样本量从多到少列出。",
                    "_source": "cases.feather:admission_rate",
                }
            )

        program, conf, hit_lang = _resolve_combo_major(school, major)
        stats = combos.get(program)
        if stats is None:
            top = sorted(combos.items(), key=lambda kv: kv[1][1], reverse=True)[:_MAX_MAJOR_LIST]
            return _json(
                {
                    "found": False,
                    "reason": "unknown_major",
                    "resolved_university": school,
                    "input_major": major,
                    "match_confidence": round(conf, 2),
                    "available_majors": [m for m, _ in top],
                    "hint": "该校录取统计里没有这个专业名；从中挑一个最接近的再查，"
                    "或直接告诉用户查不到，不要编造录取率。",
                    "_source": "cases.feather:admission_rate",
                }
            )

        rate, n = stats
        payload: dict[str, Any] = {
            "found": True,
            "university": school,
            "major": program,
            "matched_by": hit_lang,
            "source": "history_base_rate",
            "caveat": _CAVEAT,
            "_source": "cases.feather:admission_rate",
        }
        payload.update(_sample_payload(rate, n))
        payload["school_reference_pool"] = _reference_pool(school)
        return _json(payload)
    except Exception as exc:
        _log.warning(
            "lookup_admission_stats failed | uni=%s major=%s", university, major, exc_info=True
        )
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


def _reference_pool(school: str) -> dict[str, Any] | None:
    """该校在召回池里的参考样本量（0.05ms 热态）。失败不影响主结果。"""
    try:
        from src.adjustment.knn_retrieval import reference_pool_size

        return {"sample_size": int(reference_pool_size(school))}
    except Exception:
        _log.debug("reference_pool_size failed | school=%s", school, exc_info=True)
        return None


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ── 表单背景读取（两个"和我有关"的工具都要用） ────────────────────────


def _form_present(ctx: RunContext[HarnessDeps]) -> dict[str, Any]:
    """读表单里已写入的字段。工具自己看表单，模型不用重复把背景报一遍。"""
    try:
        return dict((ctx.deps.gateway.read() or {}).get("present") or {})
    except Exception:
        _log.debug("_form_present failed", exc_info=True)
        return {}


def _normalized_lang_score(present: dict[str, Any]) -> float | None:
    score = present.get("language_score")
    if not isinstance(score, (int, float)) or score <= 0:
        return None
    try:
        from src.pages.prediction.core.utils import normalize_language_score

        return float(normalize_language_score(score, str(present.get("language_type") or "雅思")))
    except Exception:
        _log.debug("normalize_language_score failed", exc_info=True)
        return None


def _fallback_index(bg_uni: str, bg_major: str) -> dict[str, Any]:
    """按 (本科院校, 本科专业) 缓存分层计数索引。

    `compute_fallback_probabilities` 每次调用都会重建 4 层 groupby（实测 ~150ms）+
    重读 cases_df（worker 线程里 `@st.cache_data` 不生效，再 +147ms）。
    同一轮里对多所学校各查一次就会重复付这个成本，所以按背景缓存**计数索引**
    （只有计数字典，不含原始行），缓存命中后单次查询 ~0ms。
    """
    from src.adjustment.fallback import _build_fallback_index
    from src.pages.prediction.app_data import load_raw_cases_data

    return _build_fallback_index(load_raw_cases_data(), bg_uni, bg_major)


_fallback_index_cached = lru_cache(maxsize=8)(_fallback_index)


_ESTIMATE_CAVEAT = (
    "这是**按你的本科院校 + 本科专业分层**后的历史录取率（层级收缩估计），"
    "比专业整体录取率更贴近你，但**仍然不含你的 GPA、语言成绩、实习科研**，"
    "也和页面上最终的录取概率口径不同。回答时必须说明这是历史参考、不是系统给出的最终概率。"
)


@traced
def estimate_admission_chance(
    ctx: RunContext[HarnessDeps], target_university: str, target_major: str = ""
) -> str:
    """估算「以你目前的本科背景，申这个目标项目」的历史录取可能。

    用于回答「**我**申这个专业有戏吗 / 我的背景够不够」这类问题。
    它会**自动读取表单里已写入的本科院校与本科专业**做分层，你不需要把背景再报一遍。

    如果表单里还没有本科院校，先向用户问清楚，或先用 write_form 写入。

    Args:
        target_university: 目标院校，中文校名（如"香港大学""港大"）。
        target_major: 目标专业，中文或英文都可以（如"金融"或 "Master of Finance"）。
    """
    try:
        present = _form_present(ctx)
        bg_uni = str(present.get("university") or "").strip()
        bg_major = str(present.get("major") or "").strip()
        if not bg_uni:
            return _json(
                {
                    "found": False,
                    "reason": "missing_background_university",
                    "hint": "表单里还没有本科院校，无法做分层估计。先向用户问清楚本科院校，"
                    "或用 write_form 写进去再查。",
                    "_source": "cases.feather:fallback_base_rate",
                }
            )

        school, school_conf = _resolve_school(target_university)
        index = _admit_index()
        combos = index.get(school, {})
        if not combos:
            return _json(
                {
                    "found": False,
                    "reason": "unknown_school",
                    "resolved_university": school,
                    "match_confidence": round(school_conf, 2),
                    "hint": "录取统计里没有这所目标院校；不要对不在库中的院校给概率。",
                    "_source": "cases.feather:fallback_base_rate",
                }
            )

        program, conf, hit_lang = _resolve_combo_major(school, target_major) if target_major else ("", 0.0, "")
        if not target_major or program not in combos:
            top = sorted(combos.items(), key=lambda kv: kv[1][1], reverse=True)[:_MAX_MAJOR_LIST]
            return _json(
                {
                    "found": False,
                    "reason": "major_required" if not target_major else "unknown_major",
                    "resolved_university": school,
                    "available_majors": [m for m, _ in top],
                    "available_major_count": len(combos),
                    "hint": "请指定一个具体专业（中文或英文都可），上面按样本量从多到少列出。",
                    "_source": "cases.feather:fallback_base_rate",
                }
            )

        from src.adjustment.fallback import _resolve_fallback

        fb = _resolve_fallback(
            _fallback_index_cached(bg_uni, bg_major), bg_uni, bg_major, school, program
        )
        stratum_n = int(fb.sample_count)
        shrinkage = round(float(fb.shrinkage_weight), 3)

        payload: dict[str, Any] = {
            "found": True,
            "university": school,
            "major": program,
            "matched_by": hit_lang,
            "match_confidence": round(conf, 2),
            "background_used": {"university": bg_uni, "major": bg_major},
            "stratification": fb.level_description,
            "stratum_sample_size": stratum_n,
            "effective_n": round(float(fb.effective_n), 1),
            "shrinkage_weight": shrinkage,
            "source": "background_conditioned_base_rate",
            "caveat": _ESTIMATE_CAVEAT,
            "_source": "cases.feather:fallback_base_rate",
        }

        # 分层样本不足时，绝不给"个性化"数字：此时 shrinkage≈100%，
        # 那个数字其实已经退化成全局/专业基准，说成"你的录取可能"就是过度宣称。
        if stratum_n < _MIN_N_FOR_RATE:
            payload["probability"] = None
            payload["ci95"] = None
            payload["insufficient_stratum"] = True
            broader = index.get(school, {}).get(program)
            if broader:
                payload["broader_reference"] = {
                    "label": "该项目整体的历史录取率（不含你的背景分层）",
                    "admit_rate": round(broader[0], 3),
                    "sample_size": broader[1],
                }
            payload["hint"] = (
                f"你这一层（{fb.level_description}）只有 {stratum_n} 条样本，"
                "**不要给个性化的录取可能**。可以改用 broader_reference 说这个项目整体的历史情况，"
                "或者用 lookup_admission_stats 查询，并说明样本有限。"
            )
        else:
            payload["probability"] = round(float(fb.probability), 3)
            payload["ci95"] = [round(float(fb.ci_lower), 3), round(float(fb.ci_upper), 3)]
            if stratum_n < _MIN_N_FOR_CONFIDENT:
                payload["note"] = (
                    f"分层样本 n={stratum_n} 偏少（收缩权重 {shrinkage}，越高越接近整体基准），"
                    "区间较宽，只能作为大致参考。"
                )
        return _json(payload)
    except Exception as exc:
        _log.warning(
            "estimate_admission_chance failed | uni=%s major=%s",
            target_university,
            target_major,
            exc_info=True,
        )
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


_MAX_SIMILAR_K = 30
_MIN_SIMILAR_K = 5

_SIMILAR_CAVEAT = (
    "⚠️ 库里的参考案例**全部是已录取的**（实测候选池 25,499 条录取率 100%，而全库整体只有 31.8%）。"
    "所以**绝不能用它算录取率**，也不能说「和你像的人录取率 X%」——那会把 100% 说出来。"
    "它只说明一件事：**像你这样背景的人，有成功申上过的先例**。"
    "另外只返回聚合数字，不含任何个体申请者的原始信息。"
)


@traced
def lookup_similar_admitted_cases(
    ctx: RunContext[HarnessDeps],
    target_university: str,
    target_major: str = "",
    k: int = 20,
) -> str:
    """找「背景和你相近、并且已经申上」的参考案例（成功先例，不是录取率）。

    用于回答「有没有像我这样的人申上过 / 我这个背景方向对不对」——
    给用户一个存在性证据。会**自动读取表单里的本科院校/专业/GPA/语言成绩**算相似度。

    ⚠️ 案例库**只收录已录取的案例**，所以：
    - **不要**用它计算或暗示任何录取率/概率；
    - 只能说「我们库里有 N 条和你背景相近的成功案例，相似度在 X~Y 之间」。

    Args:
        target_university: 目标院校，中文校名（如"香港大学""港大"）。
        target_major: 目标专业，中文或英文都可以。留空时按该校整体样本检索。
        k: 取最相近的多少条案例，默认 20。系统会夹在 5~30 之间。
    """
    try:
        present = _form_present(ctx)
        bg_uni = str(present.get("university") or "").strip()
        if not present:
            return _json(
                {
                    "found": False,
                    "reason": "empty_form",
                    "hint": "表单里没有任何背景字段，无法计算相似度。先向用户问清楚，或用 write_form 写入。",
                    "_source": "cases.feather:admitted_reference_cases",
                }
            )

        school, school_conf = _resolve_school(target_university)
        program = ""
        if target_major:
            index = _admit_index()
            if school in index:
                program, _conf, _lang = _resolve_combo_major(school, target_major)

        k_clamped = max(_MIN_SIMILAR_K, min(_MAX_SIMILAR_K, int(k or 20)))
        student = {
            "target_university": school,
            "target_major": program or "?",
            "background_university": bg_uni,
            "background_major": str(present.get("major") or ""),
            "gpa": present.get("gpa"),
            "lang_score": _normalized_lang_score(present),
        }

        from src.adjustment.knn_retrieval import retrieve_similar_cases

        cases, level, note = retrieve_similar_cases(student, k=k_clamped)

        used = ["target_university"]
        for field, key in (
            ("background_university", "background_university"),
            ("background_major", "background_major"),
            ("gpa", "gpa"),
            ("lang_score", "lang_score"),
        ):
            if student.get(key) not in (None, "", "?"):
                used.append(field)

        if not cases:
            return _json(
                {
                    "found": False,
                    "reason": "no_similar_cases",
                    "resolved_university": school,
                    "resolved_major": program,
                    "match_confidence": round(school_conf, 2),
                    "pool_note": note,
                    "hint": "该项目在当前案例库里没有可比对的先例，直接告诉用户查不到，不要编造。",
                    "_source": "cases.feather:admitted_reference_cases",
                }
            )

        sims = [float(c.get("similarity") or 0.0) for c in cases]
        match_types: dict[str, int] = {}
        for c in cases:
            mt = str(c.get("match_type") or "unknown")
            match_types[mt] = match_types.get(mt, 0) + 1

        payload: dict[str, Any] = {
            "found": True,
            "university": school,
            "major": program or "(全校范围)",
            "similar_case_count": len(cases),
            "cases_are_admitted_only": True,
            "similarity": {
                "min": round(min(sims), 3),
                "mean": round(sum(sims) / len(sims), 3),
                "max": round(max(sims), 3),
            },
            "match_types": match_types,
            "upset_count": sum(1 for c in cases if c.get("is_upset")),
            "pool_level": level,
            "pool_note": note,
            "inputs_used": used,
            "source": "admitted_reference_cases",
            "caveat": _SIMILAR_CAVEAT,
            "_source": "cases.feather:admitted_reference_cases",
        }
        if len(cases) < _MIN_SIMILAR_K:
            payload["note"] = f"可比先例只有 {len(cases)} 条，只能作定性参考。"
        if "gpa" not in used or "lang_score" not in used:
            payload["note"] = (
                payload.get("note", "") + "（缺少 GPA 或语言成绩，相似度只用院校/专业，参考性下降。）"
            )
        return _json(payload)
    except Exception as exc:
        _log.warning(
            "lookup_similar_admitted_cases failed | uni=%s major=%s",
            target_university,
            target_major,
            exc_info=True,
        )
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


RECALL_TOOLS = [
    lookup_admission_stats,
    estimate_admission_chance,
    lookup_similar_admitted_cases,
]
