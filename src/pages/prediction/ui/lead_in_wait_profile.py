from __future__ import annotations

import re
from typing import Any

from src.utils.numeric import clip_probability

_FINANCE = ("金融", "商科", "会计", "mba", "finance", "business", "经济", "管理")
_STEM = ("计算机", "工程", "理工", "数据", "电子", "cs", "ai", "统计", "物理", "化学")
_CREATIVE = ("设计", "传媒", "艺术", "语言", "教育", "心理", "新闻", "广告", "文学")


def infer_persona(*parts: str) -> str:
    blob = " ".join(p for p in parts if p).lower()
    if any(k in blob for k in _FINANCE):
        return "finance"
    if any(k in blob for k in _STEM):
        return "stem"
    if any(k in blob for k in _CREATIVE):
        return "creative"
    return "signal"


def extract_tags(details: list[str], partial_text: str) -> list[str]:
    tags: list[str] = []
    for line in details:
        m = re.search(r"(?:已抓住|识别到)[：:]\s*(.+)$", line)
        if m:
            for bit in re.split(r"[,，、/|]", m.group(1)):
                t = bit.strip()
                if t and t not in tags:
                    tags.append(t)
    if not tags and partial_text:
        for bit in re.split(r"[,，、\s]+", partial_text):
            t = bit.strip()
            if 1 < len(t) <= 8 and t not in tags:
                tags.append(t)
            if len(tags) >= 4:
                break
    return tags[:6]


def parse_score_norms(*parts: str) -> tuple[float, float]:
    blob = " ".join(p for p in parts if p)
    gpa = 0.65
    lang = 0.65

    m = re.search(r"(?:gpa|GPA)\s*[:=]?\s*([0-4](?:\.\d+)?)", blob)
    if m:
        gpa = clip_probability(float(m.group(1)) / 4.0)
    else:
        m = re.search(r"均分\s*([0-9]{2}(?:\.\d+)?)", blob)
        if m:
            gpa = clip_probability(float(m.group(1)) / 100.0)

    m = re.search(r"(?:雅思|IELTS)\s*([0-9](?:\.\d+)?)", blob, re.I)
    if m:
        lang = clip_probability((float(m.group(1)) - 4.0) / 5.0)
    else:
        m = re.search(r"(?:托福|TOEFL)\s*([0-9]{2,3})", blob, re.I)
        if m:
            lang = clip_probability((float(m.group(1)) - 40.0) / 80.0)

    return gpa, lang


def _dump(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return dict(obj.model_dump())
    return {}


def _norm_gpa(value: Any, scale: str = "") -> float | None:
    if value is None or value == "":
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    scale_l = (scale or "").lower()
    if "100" in scale_l or v > 4.5:
        return clip_probability(v / 100.0)
    return clip_probability(v / 4.0)


def _norm_language(score: Any, lang_type: str = "") -> float | None:
    if score is None or score == "":
        return None
    try:
        v = float(score)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    t = (lang_type or "").lower()
    if "toefl" in t or "托福" in t:
        return max(0.0, min(1.0, (v - 40.0) / 80.0))
    if "ielts" in t or "雅思" in t or v <= 9.5:
        return max(0.0, min(1.0, (v - 4.0) / 5.0))
    return max(0.0, min(1.0, (v - 40.0) / 80.0))


def _fmt_num(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _score_labels(bg: dict[str, Any]) -> tuple[str, str]:
    gpa_label = ""
    gpa = bg.get("gpa")
    if gpa not in (None, ""):
        try:
            gv = float(gpa)
            if gv > 0:
                scale_l = str(bg.get("gpa_scale") or "").lower()
                if "100" in scale_l or gv > 4.5:
                    gpa_label = f"{_fmt_num(gv)}/100"
                else:
                    gpa_label = f"{_fmt_num(gv)}/4"
        except (TypeError, ValueError):
            pass

    lang_label = ""
    score = bg.get("language_score")
    if score not in (None, ""):
        try:
            lv = float(score)
            if 0 < lv <= 120:
                t = str(bg.get("language_type") or "").lower()
                if "toefl" in t or "托福" in t:
                    lang_label = f"TOEFL {_fmt_num(lv)}"
                elif "ielts" in t or "雅思" in t or lv <= 9.5:
                    lang_label = f"IELTS {_fmt_num(lv)}"
                else:
                    lang_label = f"LANG {_fmt_num(lv)}"
        except (TypeError, ValueError):
            pass

    return gpa_label, lang_label


def profile_from_sources(
    ctx: Any | None,
    applied: dict[str, Any] | None,
) -> dict[str, Any]:
    applied = dict(applied or {})
    bg: dict[str, Any] = {}
    raw_input = ""
    if ctx is not None:
        bg = _dump(getattr(ctx, "extracted_background", None))
        raw_input = str(getattr(ctx, "raw_input", "") or "")
        if not bg.get("gpa") and getattr(ctx, "gpa_raw", 0):
            bg["gpa"] = getattr(ctx, "gpa_raw", None) or getattr(ctx, "gpa", None)
        if not bg.get("language_score") and getattr(ctx, "language_score_raw", 0):
            bg["language_score"] = getattr(ctx, "language_score_raw", None) or getattr(
                ctx, "language_score", None
            )
        if not bg.get("language_type") and getattr(ctx, "language_type", ""):
            bg["language_type"] = getattr(ctx, "language_type", "")
        if not bg.get("university") and getattr(ctx, "background_university", ""):
            bg["university"] = getattr(ctx, "background_university", "")
        if not bg.get("major") and getattr(ctx, "background_major", ""):
            bg["major"] = getattr(ctx, "background_major", "")
        if not bg.get("target_majors") and getattr(ctx, "target_majors", None):
            bg["target_majors"] = list(getattr(ctx, "target_majors", []) or [])
        if not bg.get("target_schools") and getattr(ctx, "target_universities", None):
            bg["target_schools"] = list(getattr(ctx, "target_universities", []) or [])

    for k, v in applied.items():
        if v not in (None, "", [], {}):
            bg[k] = v
            if k == "background_university":
                bg["university"] = v
            if k == "background_major":
                bg["major"] = v

    tags: list[str] = []
    for key in ("university", "major", "language_type", "country"):
        val = bg.get(key)
        if isinstance(val, str) and val.strip() and val.strip() not in tags:
            tags.append(val.strip())
    for key in ("target_schools", "target_majors"):
        for item in bg.get(key) or []:
            s = str(item).strip()
            if s and s not in tags:
                tags.append(s)
            if len(tags) >= 6:
                break

    major_blob = " ".join(
        str(x)
        for x in (
            bg.get("major"),
            bg.get("major_2"),
            bg.get("background_major"),
            *(bg.get("target_majors") or []),
        )
        if x
    )

    if len(tags) < 2 and raw_input:
        for bit in re.split(r"[,，、\s/|]+", raw_input):
            t = bit.strip()
            if 1 < len(t) <= 10 and t not in tags and not t.isdigit():
                tags.append(t)
            if len(tags) >= 5:
                break

    gpa_label, lang_label = _score_labels(bg)

    return {
        "gpa_norm": _norm_gpa(bg.get("gpa"), str(bg.get("gpa_scale") or "")),
        "lang_norm": _norm_language(bg.get("language_score"), str(bg.get("language_type") or "")),
        "gpa_label": gpa_label,
        "lang_label": lang_label,
        "tags": tags[:6],
        "major_blob": major_blob,
        "raw_input": raw_input,
    }
