"""Bridge: LeadInAgent extracted_background → prediction form session_state.

Strategy: directly set st.session_state widget keys so Streamlit picks them up on rerun.
Also set session_manager values so form init logic sees non-None defaults.
"""

from __future__ import annotations

import difflib
from typing import Any

import streamlit as st

from src.agent.context import StudentContext

_FUZZY_CUTOFF = 0.6

# ── widget key constants (must match form components) ──────────────────────
_W_UNIVERSITY = "background_university_selectbox"
_W_MAJOR = "background_major_selectbox"
_W_GPA_SCALE = "gpa_scale_widget_key"
_W_GPA = "gpa_raw_input_widget"
_W_LANG_TYPE = "language_type_widget_key"
_W_LANG_SCORE = "language_score_input_widget"
_W_COUNTRY = "target_countries_multiselect"
_W_UNI = "target_universities_multiselect"
_W_MAJORS = "target_majors_multiselect"


def apply_lead_in_to_form(ctx: StudentContext, session_manager: Any) -> dict[str, Any]:
    bg = ctx.extracted_background or {}
    if not bg or not any(v for v in bg.values() if v):
        return {}

    applied: dict[str, Any] = {}

    # ── university ──
    uni = bg.get("university")
    if uni and isinstance(uni, str) and uni.strip():
        uni_raw = uni.strip()
        uni_options = _get_list(session_manager, "background_universities_cache")
        matched = _fuzzy_match(uni_raw, uni_options) if uni_options else uni_raw
        st.session_state[_W_UNIVERSITY] = matched
        session_manager.set(background_university=matched)
        applied["background_university"] = matched

    # ── major ──
    major = bg.get("major")
    if major and isinstance(major, str) and major.strip():
        major_raw = major.strip()
        major_cache = session_manager.get("background_majors_cache", {}) or {}
        major_options = major_cache.get("majors_display", [])
        matched = _fuzzy_match(major_raw, major_options) if major_options else major_raw
        st.session_state[_W_MAJOR] = matched
        # Also store in user_history_data for the selectbox index fallback
        _update_user_history(session_manager, "background_major_original", matched)
        applied["background_major"] = matched

    # ── gpa ──
    gpa = bg.get("gpa")
    if gpa is not None and isinstance(gpa, (int, float)) and float(gpa) > 0:
        gpa_val = round(float(gpa), 2)
        if gpa_val > 4.0:
            session_manager.set(gpa_scale="5.0")
            st.session_state[_W_GPA_SCALE] = "5.0"
        session_manager.set(gpa_raw_input=gpa_val)
        st.session_state[_W_GPA] = gpa_val
        applied["gpa"] = gpa_val

    # ── language type ──
    lang_type = bg.get("language_type")
    if lang_type and isinstance(lang_type, str) and lang_type.strip() in ("雅思", "托福"):
        lt = lang_type.strip()
        session_manager.set(language_type=lt)
        st.session_state[_W_LANG_TYPE] = lt
        applied["language_type"] = lt

    # ── language score ──
    lang_score = bg.get("language_score")
    if lang_score is not None and isinstance(lang_score, (int, float)) and float(lang_score) > 0:
        ls = float(lang_score)
        is_ielts = (lang_type or "") == "雅思"
        ls = round(ls, 1) if is_ielts else round(ls)
        session_manager.set(language_score_input=ls)
        st.session_state[_W_LANG_SCORE] = ls
        applied["language_score"] = ls

    # ── target country ──
    country = bg.get("country")
    if country:
        countries = [country] if isinstance(country, str) else list(country)
        session_manager.set(selected_target_countries=countries)
        st.session_state[_W_COUNTRY] = countries
        applied["target_country"] = countries

    # ── target schools ──
    schools = bg.get("target_schools")
    if schools:
        schools_list = _to_list(schools)
        if schools_list:
            session_manager.set(selected_target_universities=schools_list)
            st.session_state[_W_UNI] = schools_list
            applied["target_schools"] = schools_list

    # ── target majors ──
    majors = bg.get("target_majors")
    if majors:
        majors_list = _to_list(majors)
        if majors_list:
            session_manager.set(selected_target_majors=majors_list)
            st.session_state[_W_MAJORS] = majors_list
            applied["target_majors"] = majors_list

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
    """Exact → substring → difflib. Falls back to original query."""
    if not candidates:
        return query

    for c in candidates:
        if c == query or c.casefold() == query.casefold():
            return c

    for c in candidates:
        if query in c or c in query:
            return c

    matches = difflib.get_close_matches(query, candidates, n=1, cutoff=_FUZZY_CUTOFF)
    return matches[0] if matches else query


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
    uh = session_manager.get("user_history_data", {})
    uh[key] = value
    session_manager.set(user_history_data=uh)


def _apply_experience(session_manager: Any, bg: dict, key: str) -> None:
    text = bg.get(key)
    if not text or not isinstance(text, str) or not text.strip():
        return
    uh = session_manager.get("user_history_data", {})
    exp = uh.setdefault("experience_details", {})
    exp[key] = text.strip()
    uh[f"{key}_count"] = max(uh.get(f"{key}_count", 0), 1)
    session_manager.set(user_history_data=uh)
    # Set widget states directly so number_input / text_input pick them up
    st.session_state[f"{key}_count_input"] = uh[f"{key}_count"]
    st.session_state[f"{key}_details_input"] = text.strip()
