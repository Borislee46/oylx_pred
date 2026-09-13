"""只读数据探查工具。

设计原则（来自实测盘点，见 docs/agent_module_review_20260911.md）：
- 只放 **零重计算 / 无副作用 / 毫秒级** 的查询。任何会跑模型、写盘、读 100MB+ 数据的
  入口都不进这里（否则会打爆单轮预算，且 LLM 会把"查不到"包装成"查到了"）。
- 每个工具都必须能诚实地说"查不到"：返回体带 `found` / `reason`，
  让智能体据此回答"我查不到"，而不是编造。
- 口径必须自带 `_source` 与 `caveat`：同一个"概率/门槛"在本系统里有多种语义，
  不标注会让 AI 的数字和页面显示对不上。

第二波候选（需要预热，暂缓）：
  compute_fallback_probabilities（~150ms，带 Wilson CI）、retrieve_similar_cases（热 14.9ms，含隐私案例）
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from pydantic_ai import RunContext

from src.agent.harness import HarnessDeps
from src.agent.tools.tool_trace import traced

_log = logging.getLogger("lead_in_probe")

# 单校专业数最多也就几十个（港科 67）。宁可一次给全，也不要只给字母序前若干条——
# 实测：只给前 25 条时模型查不到目标专业，会自己换个名字再查一次，白付一轮 LLM 往返。
_MAX_MAJOR_SUGGESTIONS = 80


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


@lru_cache(maxsize=1)
def _valid_universities() -> tuple[str, ...]:
    from src.pages.prediction.core.data_manager import get_valid_school_major_set

    combos = get_valid_school_major_set() or set()
    return tuple(sorted({u for u, _ in combos}))


@lru_cache(maxsize=1)
def _school_to_majors() -> dict[str, tuple[str, ...]]:
    from src.pages.prediction.core.data_manager import get_valid_school_major_set

    combos = get_valid_school_major_set() or set()
    bucket: dict[str, list[str]] = {}
    for uni, major in combos:
        bucket.setdefault(uni, []).append(major)
    return {u: tuple(sorted(set(m))) for u, m in bucket.items()}


@lru_cache(maxsize=1)
def _school_cn_index() -> dict[str, dict[str, str]]:
    """学校 → {专业中文名: 专业英文名}。

    项目库每一行都有 `专业中文名称`（实测零缺失、基本 1:1），所以中文专业名
    可以直接在校内索引上做模糊匹配 —— 这正是 fuzzy 该干的活。
    没有这张索引，模型只能拿"金融"去撞英文库，撞不到就自己猜英文名，白付一轮 LLM 往返。
    """
    from src.pages.prediction.core.data_manager import _data_manager

    df = _data_manager.details_df
    if df is None or getattr(df, "empty", True):
        return {}
    out: dict[str, dict[str, str]] = {}
    schools = df["学校"].astype(str)
    english = df["专业英文名称"].astype(str)
    chinese = df["专业中文名称"].astype(str) if "专业中文名称" in df.columns else None
    if chinese is None:
        return {}
    for school, en, cn in zip(schools, english, chinese, strict=False):
        if not cn or cn == "nan" or not en or en == "nan":
            continue
        out.setdefault(school, {}).setdefault(cn, en)
    return out


def _resolve_school(raw: str) -> tuple[str, float]:
    """把用户/模型给的校名对齐到项目库里的正式中文校名。"""
    name = str(raw or "").strip()
    if not name:
        return "", 0.0
    options = list(_valid_universities())
    if name in options:
        return name, 1.0
    from src.agent.bridges.fuzzy_match import _fuzzy_match

    matched, conf = _fuzzy_match(name, options)
    return str(matched or name), float(conf or 0.0)


def _resolve_major(school: str, raw: str) -> tuple[str, float, str]:
    """把（中文或英文的）专业名对齐到项目库的英文专业名。

    返回 (英文专业名, 置信度, 命中的语言)。中文走校内中文名索引，英文走英文名列表。
    """
    name = str(raw or "").strip()
    majors_en = _school_to_majors().get(school, ())
    cn_index = _school_cn_index().get(school, {})
    if not name:
        return "", 0.0, ""

    if name in majors_en:
        return name, 1.0, "en"
    if name in cn_index:
        return cn_index[name], 1.0, "zh"

    from src.agent.bridges.fuzzy_match import _fuzzy_match

    best_en, conf_en = _fuzzy_match(name, list(majors_en))
    best_cn, conf_cn = (name, 0.0)
    if cn_index:
        best_cn, conf_cn = _fuzzy_match(name, list(cn_index))

    if conf_cn > conf_en:
        return str(cn_index.get(str(best_cn), best_en) or best_en), float(conf_cn), "zh"
    return str(best_en or name), float(conf_en), "en"


_PROGRAM_FIELDS = (
    "专业中文名称",
    "学习年限",
    "学费",
    "授课语言",
    "录取要求",
    "专业背景要求",
    "考试要求",
    "特殊要求",
    "申请方式",
    "推荐信方式",
    "成绩送分要求",
    "是否面试",
    "是否笔试",
    "考核形式",
    "申请注意事项",
    "专业网址",
)


def _shared_prefix_len(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _near_miss_majors(school: str, raw: str, *, limit: int = 3) -> list[dict[str, str]]:
    """中文专业名"接近但没过 fuzzy cutoff"时，给几个语义上相关的项目。

    例：港科没有叫"数据科学"的项目，但"大数据技术理学硕士""数据驱动建模硕士"是相关的。
    这能把模型从"重试猜名字"引到"从相关项里挑"，省一轮 LLM 往返。
    用与 `_fuzzy_match` 相同的中文 scorer（partial_ratio），保证口径一致；
    排序再额外偏向**与输入共享前缀**的项——"数据科学" 应该先落到"数据驱动建模"，
    而不是只共享"科学"二字的"科学（A组）"。
    """
    cn_index = _school_cn_index().get(school, {})
    if not cn_index or not raw:
        return []
    try:
        from rapidfuzz import fuzz, process
    except Exception:  # pragma: no cover
        return []
    hits = process.extract(
        raw, list(cn_index), scorer=fuzz.partial_ratio, limit=max(limit * 4, 12), score_cutoff=40
    )
    hits.sort(key=lambda h: (_shared_prefix_len(raw, str(h[0])), h[1]), reverse=True)
    return [
        {"cn": str(cn), "en": cn_index[str(cn)], "score": str(int(score))}
        for cn, score, _ in hits[:limit]
    ]


def _major_hint_payload(
    school: str, majors: tuple[str, ...], *, reason: str, extra: dict[str, Any] | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "found": False,
        "reason": reason,
        "resolved_university": school,
        "available_majors": list(majors[:_MAX_MAJOR_SUGGESTIONS]),
        "available_major_count": len(majors),
        "available_majors_truncated": len(majors) > _MAX_MAJOR_SUGGESTIONS,
        "hint": "上面是该校全部可选专业（英文全名）。请从中挑一个最接近的再查一次；"
        "如果确实没有对应方向，就直接告诉用户查不到，不要编造门槛。"
        "也可以用中文专业名再传一次，工具会自动匹配。",
        "_source": "school_major_details.feather",
    }
    if extra:
        payload.update(extra)
    return payload


@traced
def lookup_program(ctx: RunContext[HarnessDeps], university: str, major: str = "") -> str:
    """查某个「目标院校 + 专业」项目的官方要求。

    用于回答：语言门槛（雅思/托福）、要不要 GRE/GMAT、要不要面试笔试、学费学制、
    录取要求原文。**不要凭常识回答这类问题，先查这个工具。**

    返回值里 `GPA要求` 字段不可靠（多数只是"是/否"），要讲 GPA 限制请读
    「录取要求」「专业背景要求」的原文。

    Args:
        university: 目标院校，中文校名（如"香港大学""港大"），系统会自动对齐到库内正式名。
        major: 目标专业。**中文或英文都可以**（如"金融""数据科学"或 "Master of Finance"），
            工具会同时按专业中文名和英文项目名做模糊匹配。只知道学校不知道专业时留空，
            会返回该校全部可选专业。
    """
    try:
        school, school_conf = _resolve_school(university)
        majors = _school_to_majors().get(school, ())

        if not majors:
            return _json(
                {
                    "found": False,
                    "reason": "unknown_school",
                    "input_university": university,
                    "resolved_university": school,
                    "match_confidence": round(school_conf, 2),
                    "supported_school_count": len(_valid_universities()),
                    "hint": "项目库里没有这所目标院校；请用 list_supported_targets 查支持名单，"
                    "不要对不在库中的院校给出门槛数字。",
                    "_source": "school_major_details.feather",
                }
            )

        if not major:
            return _json(_major_hint_payload(school, majors, reason="major_required"))

        program, major_conf, hit_lang = _resolve_major(school, major)

        from src.pages.prediction.core.data_manager import _data_manager

        row = _data_manager.get_row(school, program)
        if row is None:
            extra: dict[str, Any] = {
                "input_major": major,
                "match_confidence": round(major_conf, 2),
                "hint": "中文或英文专业名都可以直接传。可以从 available_majors（英文全名）"
                "或 near_miss_majors（本案里语义相近的项目）里挑一个再查。"
                "确实没有对应方向就告诉用户查不到，不要编造门槛。",
            }
            near = _near_miss_majors(school, major)
            if near:
                extra["near_miss_majors"] = near
            return _json(
                _major_hint_payload(school, majors, reason="unknown_major", extra=extra)
            )

        fields = {
            f: str(row.get(f)).strip()
            for f in _PROGRAM_FIELDS
            if row.get(f) is not None and str(row.get(f)).strip() not in ("", "nan", "None")
        }
        lang_reqs = row.get("_lang_reqs") or {}

        result: dict[str, Any] = {
            "found": True,
            "university": school,
            "major": program,
            "matched_by": hit_lang,
            "fields": fields,
            "language_requirements": {
                k: v for k, v in dict(lang_reqs).items() if v not in (None, "")
            },
            "notes": [
                "语言门槛以 language_requirements 为准（已从结构化字段解析）。",
                "「录取要求」「专业背景要求」为官方原文，可直接引用其中的 GPA / 背景限制。",
                "本项目库**没有可靠的 GPA 分数门槛字段**，不要引用「GPA要求」列（多数是 是/否）。",
                "库中没有的信息（排名、就业率、真实录取概率）一律不要编造。",
            ],
            "_source": "school_major_details.feather",
        }
        return _json(result)
    except Exception as exc:
        _log.warning("lookup_program failed | uni=%s major=%s", university, major, exc_info=True)
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


@traced
def lookup_school(ctx: RunContext[HarnessDeps], school: str) -> str:
    """查本科院校的层次与别名归一。

    用于回答：这所学校是 985/211/双非 吗、是不是海本、系统认不认这个校名。
    用户写「985/双非」这类**层次别名**时不要查，直接写进表单即可。

    Args:
        school: 本科院校名或简称（如"中山大学""中大""北大"）。
            如果给的是"学校+学院"（如"北大光华"），本工具可能查不到层次，
            此时按用户原话写入表单即可，不要猜层次。
    """
    try:
        from src.utils.schools.alias_resolver import (
            is_school_category_alias,
            resolve_background_school,
        )
        from src.utils.schools.level_service import get_school_level_service

        raw = str(school or "").strip()
        if not raw:
            return _json({"found": False, "reason": "empty_input"})

        if is_school_category_alias(raw):
            return _json(
                {
                    "found": True,
                    "input": raw,
                    "is_level_alias": True,
                    "hint": "这是院校层次别名，不是具体学校名；直接原样写入表单即可，无需归一。",
                    "_source": "alias_resolver",
                }
            )

        service = get_school_level_service()
        info = service.get_school_info(raw) or {}
        level = str(info.get("school_level") or "未知")
        resolved = ""
        try:
            resolved = str(resolve_background_school(raw) or "")
        except Exception:
            _log.debug("lookup_school | resolve_background_school failed", exc_info=True)

        known = level not in ("", "未知")
        return _json(
            {
                "found": known,
                "input": raw,
                "resolved_school": resolved or raw,
                "school_level": level,
                "is_overseas": bool(known and service.is_overseas_school(raw)),
                "priority": info.get("priority"),
                "hint": (
                    "可直接把 resolved_school 写入表单。"
                    if known
                    else "层次库里没有这所学校，不要猜它是 985/211；按用户原话写入即可。"
                ),
                "_source": "school_base.feather + alias_resolver",
            }
        )
    except Exception as exc:
        _log.warning("lookup_school failed | school=%s", school, exc_info=True)
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


@traced
def list_supported_targets(ctx: RunContext[HarnessDeps], country: str = "") -> str:
    """列出产品支持的目标地区与院校。

    用于回答「你们能申哪些国家/地区」「XX 地区有哪些学校可以申」，
    以及判断某个目标院校是否越界（越界目标静默丢弃，不要写入表单）。

    Args:
        country: 可选。只查某个地区时传中文地区名（如"中国香港""新加坡""英国"）。
            留空返回全部支持地区。不在支持范围内时会返回 found=false。
    """
    try:
        from src.utils.schools.config_loader import (
            TARGET_COUNTRY_UNIVERSITY_MAP,
            get_target_countries,
        )

        countries = get_target_countries()
        key = str(country or "").strip()
        if key:
            matched = next((c for c in countries if key in c or c in key), "")
            if not matched:
                return _json(
                    {
                        "found": False,
                        "reason": "unsupported_country",
                        "input_country": key,
                        "supported_countries": countries,
                        "hint": "该地区不在产品支持范围内，作为目标时不要写入表单，也不要特意强调。",
                        "_source": "config/prediction_rules.json",
                    }
                )
            return _json(
                {
                    "found": True,
                    "country": matched,
                    "universities": list(TARGET_COUNTRY_UNIVERSITY_MAP.get(matched, [])),
                    "_source": "config/prediction_rules.json",
                }
            )

        return _json(
            {
                "found": True,
                "supported_countries": countries,
                "universities_by_country": {
                    c: list(v) for c, v in TARGET_COUNTRY_UNIVERSITY_MAP.items()
                },
                "_source": "config/prediction_rules.json",
            }
        )
    except Exception as exc:
        _log.warning("list_supported_targets failed", exc_info=True)
        return _json({"found": False, "reason": "lookup_error", "error": str(exc)[:200]})


PROBE_TOOLS = [
    lookup_program,
    lookup_school,
    list_supported_targets,
]
