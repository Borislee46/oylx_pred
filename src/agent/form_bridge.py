"""Bridge: LeadInAgent extracted_background → prediction form session_state.

Strategy: directly set st.session_state widget keys so Streamlit picks them up on rerun.
Also set session_manager values so form init logic sees non-None defaults.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import streamlit as st
from rapidfuzz import fuzz, process

from src.agent.context import StudentContext
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_WIDGET_KEYS
from src.utils.school_alias_resolver import (
    is_school_category_alias,
    resolve_background_school,
    resolve_target_schools,
)

# Minimal alias map for English abbreviations that can't match via substring or char-order.
# Chinese abbreviations (港大→香港大学) are handled by _chars_in_order().
_UNIVERSITY_ALIAS_MAP: dict[str, str] = {
    "HKU": "香港大学",
    "CUHK": "香港中文大学",
    "HKUST": "香港科技大学",
    "PolyU": "香港理工大学",
    "CityU": "香港城市大学",
    "NUS": "新加坡国立大学",
    "NTU": "新加坡南洋理工大学",
    "SMU": "新加坡管理大学",
}

# Major discipline alias → English discipline name for target major fuzzy matching.
# Maps Chinese names, abbreviations, and short forms to the core discipline term
# that appears inside "Master of Science in <Discipline>".
_MAJOR_DISCIPLINE_MAP: dict[str, str] = {
    # CS / IT
    "CS": "Computer Science",
    "计算机": "Computer Science",
    "计算机科学": "Computer Science",
    "计算机科学与技术": "Computer Science",
    "信息技术": "Information Technology",
    "软件工程": "Software Engineering",
    "人工智能": "Artificial Intelligence",
    "AI": "Artificial Intelligence",
    "数据科学": "Data Science",
    "DS": "Data Science",
    "大数据": "Data Science",
    # Business
    "金融": "Finance",
    "金融学": "Finance",
    "金融工程": "Financial Engineering",
    "会计": "Accounting",
    "商业分析": "Business Analytics",
    "BA": "Business Analytics",
    "市场营销": "Marketing",
    "管理": "Management",
    "工商管理": "Business Administration",
    "MBA": "Business Administration",
    "经济": "Economics",
    "经济学": "Applied Economics",
    # Engineering
    "电子工程": "Electronic Engineering",
    "电子": "Electronic Engineering",
    "EE": "Electronic Engineering",
    "电气": "Electrical Engineering",
    "机械": "Mechanical Engineering",
    "土木": "Civil Engineering",
    "材料": "Materials Engineering",
    "生物医学": "Biomedical Engineering",
    "化学工程": "Chemical Engineering",
    # Science
    "统计": "Statistics",
    "统计学": "Statistics",
    "数学": "Mathematics",
    "物理": "Physics",
    "化学": "Chemistry",
    "生物": "Biology",
    # Other
    "法律": "Law",
    "法学": "Law",
    "传媒": "Media",
    "传播": "Communication",
    "设计": "Design",
    "建筑": "Architecture",
    "教育": "Education",
    "公共管理": "Public Management",
    "供应链": "Supply Chain Management",
    "心理学": "Psychology",
}


def apply_lead_in_to_form(ctx: StudentContext, session_manager: Any) -> dict[str, Any]:
    bg = ctx.extracted_background or {}
    if not bg or not any(v for v in bg.values() if v):
        return {}

    applied: dict[str, Any] = {}

    # ── university ──
    uni = bg.get("university")
    if uni and isinstance(uni, str) and uni.strip():
        uni_raw = uni.strip()
        # Category alias → concrete school (e.g. "985" → "北京大学")
        if is_school_category_alias(uni_raw):
            uni_raw = resolve_background_school(uni_raw)
        uni_options = _get_list(session_manager, DEFAULT_FORM_KEYS.background_universities_cache)
        matched = _fuzzy_match(uni_raw, uni_options) if uni_options else uni_raw
        st.session_state[DEFAULT_WIDGET_KEYS.background_university] = matched
        session_manager.set(background_university=matched)
        applied["background_university"] = matched

    # ── major ──
    major = bg.get("major")
    if major and isinstance(major, str) and major.strip():
        major_raw = major.strip()
        major_cache = session_manager.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
        major_options = major_cache.get("majors_display", [])
        matched = _fuzzy_match(major_raw, major_options) if major_options else major_raw
        st.session_state[DEFAULT_WIDGET_KEYS.background_major] = matched
        # Also store in user_history_data for the selectbox index fallback
        _update_user_history(session_manager, "background_major_original", matched)
        applied["background_major"] = matched

    # ── gpa ──
    gpa = bg.get("gpa")
    if gpa is not None and isinstance(gpa, (int, float)) and float(gpa) > 0:
        gpa_val = round(float(gpa), 2)
        if gpa_val > 10:
            session_manager.set(gpa_scale="100")
            st.session_state[DEFAULT_WIDGET_KEYS.gpa_scale] = "100"
        elif gpa_val > 4.0:
            session_manager.set(gpa_scale="5.0")
            st.session_state[DEFAULT_WIDGET_KEYS.gpa_scale] = "5.0"
        session_manager.set(gpa_raw_input=gpa_val)
        st.session_state[DEFAULT_WIDGET_KEYS.gpa_raw_input] = gpa_val
        applied["gpa"] = gpa_val

    # ── language type ──
    lang_type = bg.get("language_type")
    if lang_type and isinstance(lang_type, str) and lang_type.strip() in ("雅思", "托福"):
        lt = lang_type.strip()
        session_manager.set(language_type=lt)
        st.session_state[DEFAULT_WIDGET_KEYS.language_type] = lt
        applied["language_type"] = lt

    # ── language score ──
    lang_score = bg.get("language_score")
    if lang_score is not None and isinstance(lang_score, (int, float)) and float(lang_score) > 0:
        ls = float(lang_score)
        is_ielts = (lang_type or "") == "雅思"
        ls = round(ls, 1) if is_ielts else round(ls)
        session_manager.set(language_score_input=ls)
        st.session_state[DEFAULT_WIDGET_KEYS.language_score] = ls
        applied["language_score"] = ls

    # ── target country ──
    country = bg.get("country")
    if country:
        countries = [country] if isinstance(country, str) else list(country)
        session_manager.set(selected_target_countries=countries)
        st.session_state[DEFAULT_WIDGET_KEYS.target_countries] = countries
        applied["target_country"] = countries

    # ── target schools ──
    schools = bg.get("target_schools")
    if schools:
        schools_list = _to_list(schools)
        if schools_list:
            # Expand "港3" → ["香港大学", "香港中文大学", "香港科技大学"]
            expanded: list[str] = []
            for s in schools_list:
                resolved = resolve_target_schools(s)
                if resolved:
                    expanded.extend(resolved)
                else:
                    expanded.append(s)
            session_manager.set(selected_target_universities=expanded)
            st.session_state[DEFAULT_WIDGET_KEYS.target_universities] = expanded
            applied["target_schools"] = expanded

    # ── target majors ──
    majors = bg.get("target_majors")
    if majors:
        majors_list = _to_list(majors)
        if majors_list:
            # Fuzzy-match each alias to actual option names (e.g. "CS" → "Master of Science in Computer Science")
            matched_majors = [_fuzzy_match_major(m) for m in majors_list]
            session_manager.set(selected_target_majors=matched_majors)
            st.session_state[DEFAULT_WIDGET_KEYS.target_majors] = matched_majors
            applied["target_majors"] = matched_majors

    # ── standardized test (GRE / GMAT) ──
    exam_type = bg.get("standardized_test_type")
    exam_score = bg.get("standardized_test_score")
    if exam_type and isinstance(exam_type, str) and exam_type.strip() in ("GRE", "GMAT"):
        et = exam_type.strip()
        session_manager.set(standardized_test_type=et)
        st.session_state[DEFAULT_WIDGET_KEYS.standardized_test_type] = et
        applied["standardized_test_type"] = et
        if (
            exam_score is not None
            and isinstance(exam_score, (int, float))
            and float(exam_score) > 0
        ):
            es = float(exam_score)
            session_manager.set(current_exam_score=es)
            applied["standardized_test_score"] = es

    # ── experiences (routed through user_history_data, read by experience_ui) ──
    for exp_key in ("research", "internship", "paper", "award"):
        _apply_experience(session_manager, bg, exp_key)

    # ── build expander summary ──
    _set_form_summary(session_manager, applied)

    return applied


def _set_form_summary(session_manager: Any, applied: dict) -> None:
    parts: list[str] = []
    if "background_university" in applied:
        uni = applied["background_university"]
        parts.append(uni if len(uni) <= 12 else uni[:10] + "…")
    if "background_major" in applied:
        maj = applied["background_major"]
        parts.append(maj if len(maj) <= 10 else maj[:8] + "…")
    if "gpa" in applied:
        parts.append(f"GPA{applied['gpa']}")
    if "language_type" in applied:
        lt = applied["language_type"]
        if "language_score" in applied:
            ls = applied["language_score"]
            parts.append(f"{lt}{ls}")
        else:
            parts.append(lt)
    if "standardized_test_type" in applied:
        et = applied["standardized_test_type"]
        if "standardized_test_score" in applied:
            es = applied["standardized_test_score"]
            parts.append(f"{et}{es}")
        else:
            parts.append(et)
    if "target_schools" in applied:
        schools = applied["target_schools"]
        if len(schools) == 1:
            parts.append(schools[0][:10])
        elif len(schools) > 1:
            parts.append(f"{schools[0][:8]}等{len(schools)}所")
    if "target_country" in applied:
        parts.append(
            str(applied["target_country"][0])
            if isinstance(applied["target_country"], list)
            else str(applied["target_country"])
        )

    summary = " · ".join(parts) if parts else ""
    if summary:
        session_manager.set(lead_in_form_summary=summary)

    # Track that auto-fill was applied (enables the clear button to reset it)
    session_manager.set(lead_in_form_filled=bool(summary))


# ── helpers ────────────────────────────────────────────────────────────────


def _fuzzy_match(query: str, candidates: list[str]) -> str:
    """Exact → substring → rapidfuzz → char-order → alias map. Falls back to original query."""
    if not candidates:
        return query

    for c in candidates:
        if c == query or c.casefold() == query.casefold():
            return c

    for c in candidates:
        if query in c or c in query:
            return c

    # rapidfuzz partial_ratio: best substring match, CJK-aware
    match = process.extractOne(query, candidates, scorer=fuzz.partial_ratio, score_cutoff=65)
    if match:
        return match[0]

    # Chinese abbreviation: check if all query chars appear in-order in candidate.
    if query and candidates:
        for c in candidates:
            if _chars_in_order(query, c):
                return c

    # English alias fallback: NUS → 新加坡国立大学, etc.
    alias = _UNIVERSITY_ALIAS_MAP.get(query.upper())
    if alias and alias in candidates:
        return alias
    if alias:
        # Alias resolved but not in candidates — still use it (better than raw)
        return alias

    return query


def _chars_in_order(query: str, candidate: str) -> bool:
    """Return True if all chars of `query` appear in `candidate` in order."""
    if not query or not candidate:
        return False
    pos = 0
    for ch in query:
        pos = candidate.find(ch, pos)
        if pos == -1:
            return False
        pos += 1
    return True


@lru_cache(maxsize=1)
def _load_target_major_options() -> list[str]:
    """All target major English names from school_major_details, cached once."""
    try:
        from src.utils.app_data_loader import load_school_major_details_df

        df = load_school_major_details_df()
        if df is not None and "专业英文名称" in df.columns:
            return sorted(df["专业英文名称"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    return []


def _fuzzy_match_major(raw: str) -> str:
    """Convert a raw major alias to the closest target major option name.

    Strategy:
    1. Strip degree suffix (硕士/博士/学士/Master/PhD) → retry
    2. Direct match against options (exact / partial_ratio)
    3. Map alias → discipline name → partial_ratio against options
    4. Char-order fallback
    """
    if not raw or not raw.strip():
        return raw

    query = raw.strip()
    options = _load_target_major_options()
    if not options:
        return query

    # 0. Strip degree suffix and retry — "计算机科学硕士" → "计算机科学"
    _DEGREE_SUFFIX_RE = r"(硕士|博士|学士|研究生|Master|PhD|Doctor|Bachelor|MSc|MPhil|MA|MBA).*$"
    stripped = re.sub(_DEGREE_SUFFIX_RE, "", query).strip()
    if stripped and stripped != query:
        # Try alias lookup with stripped version first
        discipline = _MAJOR_DISCIPLINE_MAP.get(stripped)
        if discipline:
            match = process.extractOne(
                discipline, options, scorer=fuzz.partial_ratio, score_cutoff=70
            )
            if match:
                return match[0]
        # Also try direct match with stripped
        match = process.extractOne(stripped, options, scorer=fuzz.partial_ratio, score_cutoff=75)
        if match:
            return match[0]

    # 1. Direct fuzzy match against full option names
    match = process.extractOne(query, options, scorer=fuzz.partial_ratio, score_cutoff=75)
    if match:
        return match[0]

    # 2. Try alias → discipline name → match
    discipline = _MAJOR_DISCIPLINE_MAP.get(query)
    if not discipline:
        for k, v in _MAJOR_DISCIPLINE_MAP.items():
            if k.casefold() == query.casefold():
                discipline = v
                break
    if discipline:
        match = process.extractOne(discipline, options, scorer=fuzz.partial_ratio, score_cutoff=70)
        if match:
            return match[0]

    # 3. Try char-order match (for Chinese → English cases)
    for opt in options:
        if _chars_in_order(query, opt):
            return opt

    return query


def _get_list(session_manager: Any, cache_key: str) -> list[str]:
    data = session_manager.get(cache_key)
    if not data:
        return []
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return list(data.keys())
    return []


def _to_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def _update_user_history(session_manager: Any, key: str, value: Any) -> None:
    uh = session_manager.get(DEFAULT_FORM_KEYS.user_history_data, {})
    uh[key] = value
    session_manager.set(user_history_data=uh)


def _apply_experience(session_manager: Any, bg: dict, key: str) -> None:
    text = bg.get(key)
    if not text or not isinstance(text, str) or not text.strip():
        return
    uh = session_manager.get(DEFAULT_FORM_KEYS.user_history_data, {})
    exp = uh.setdefault("experience_details", {})
    exp[key] = text.strip()
    uh[f"{key}_count"] = max(uh.get(f"{key}_count", 0), 1)
    session_manager.set(user_history_data=uh)
    # Set widget states directly so number_input / text_input pick them up
    st.session_state[f"{key}_count_input"] = uh[f"{key}_count"]
    st.session_state[f"{key}_details_input"] = text.strip()
