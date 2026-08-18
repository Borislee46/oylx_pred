"""Bridge: LeadInAgent extracted_background → prediction form session_state.

This module implements a 5-layer fuzzy-match pipeline that translates unstructured
background fields from LLM extraction into validated Streamlit form widget state:

1. **Exact match** — case-folded equality against canonical option list.
2. **Substring match** — query is a substring of a candidate or vice versa.
3. **rapidfuzz partial_ratio** — CJK-aware best substring alignment (threshold 0.65).
4. **Char-order match** — Chinese abbreviation heuristic (e.g., 港大→香港大学).
5. **Alias map** — English abbreviations (HKU, CUHK, NUS, etc.) → full Chinese names.

For target majors, there is an additional layer: major-discipline alias mapping
(e.g., "CS" → "Computer Science" → "Master of Science in Computer Science").

Strategy: directly set st.session_state widget keys so Streamlit picks them up
on rerun. Also set session_manager values so form init logic sees non-None defaults.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import streamlit as st
from rapidfuzz import fuzz, process

from src.agent.context import StudentContext
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS, DEFAULT_WIDGET_KEYS
from src.pages.prediction.input_form_components.background_ui import _generate_major_options
from src.pages.prediction.page_data_loader import machine_learning_model
from src.utils.logger import setup_logger
from src.utils.schools.alias_resolver import (
    get_category_display_label,
    is_school_category_alias,
    resolve_background_school,
    resolve_target_schools,
)

_log = setup_logger("page3", "prediction")

# Re-export pure fuzzy-match logic from the canonical agent/bridges location.
# These are used by external callers (tests, form_tools); kept here for
# backward-compatible import paths.
from src.agent.bridges.fuzzy_match import (  # noqa: E402, F401
    _BG_MAJOR_ALIAS_MAP,
    _MAJOR_DISCIPLINE_MAP,
    _UNIVERSITY_ALIAS_MAP,
    _chars_in_order,
    _fuzzy_match,
    _fuzzy_match_major,
    _infer_gpa_scale_from_text,
    _load_target_major_options,
    cross_reference_school_majors,
    fuzzy_match_major_smart,
)


def reset_lead_in_profile(
    session_manager: Any,
    ctx: StudentContext,
    *,
    reset_state_machine: bool = True,
) -> None:
    """预测完成后开始新学生背景：清空 agent 内存与表单 widget，避免上轮残留。

    When ``reset_state_machine=False`` (mid-dispatch from ``_try_tools``), only
    form/profile fields are cleared — the active ``LeadInTurnStateMachine`` phase
    must not be wiped while GATING/EXTRACTING is in progress.
    """
    ctx.extracted_background = {}
    session_manager.set(
        lead_in_form_summary=None,
        lead_in_form_filled=False,
        lead_in_missing_fields=None,
        lead_in_low_confidence_fields=None,
        lead_in_low_confidence_labels=None,
        background_university_display_label=None,
        background_university_alias=None,
        background_university=None,
        gpa_raw_input=None,
        language_score_input=None,
        **{DEFAULT_FORM_KEYS.language_score_user_provided: False},
    )
    for widget_key in (
        DEFAULT_WIDGET_KEYS.background_university,
        DEFAULT_WIDGET_KEYS.background_major,
        DEFAULT_WIDGET_KEYS.background_major_2,
        DEFAULT_WIDGET_KEYS.language_score,
        DEFAULT_WIDGET_KEYS.language_type,
        DEFAULT_WIDGET_KEYS.gpa_raw_input,
        DEFAULT_WIDGET_KEYS.gpa_scale,
        DEFAULT_WIDGET_KEYS.standardized_test_type,
        DEFAULT_WIDGET_KEYS.research_count,
        DEFAULT_WIDGET_KEYS.award_count,
        DEFAULT_WIDGET_KEYS.internship_count,
        DEFAULT_WIDGET_KEYS.paper_count,
        DEFAULT_WIDGET_KEYS.research_details,
        DEFAULT_WIDGET_KEYS.award_details,
        DEFAULT_WIDGET_KEYS.internship_details,
        DEFAULT_WIDGET_KEYS.paper_details,
        DEFAULT_WIDGET_KEYS.target_universities,
        DEFAULT_WIDGET_KEYS.target_majors,
    ):
        if widget_key in st.session_state:
            del st.session_state[widget_key]
    # Also clear the LeadIn persisted chips key (fallback for _render_persisted)
    st.session_state.pop("_lead_in_last_applied", None)
    st.session_state.pop("_lead_in_low_conf_display", None)
    uh = session_manager.get(DEFAULT_FORM_KEYS.user_history_data, {}) or {}
    for hist_key in (
        "background_university",
        "background_major_original",
        "background_major_2_original",
    ):
        uh.pop(hist_key, None)
    exp = uh.get("experience_details", {}) or {}
    for exp_key in ("research", "internship", "paper", "award"):
        exp.pop(exp_key, None)
        uh.pop(f"{exp_key}_count", None)
    uh["experience_details"] = exp
    session_manager.set(user_history_data=uh)

    # Always clear display-relevant fields from LeadIn state machine,
    # even when reset_state_machine=False (called from apply_fresh_session).
    # The phase must NOT be reset mid-dispatch, but stale chips / low-conf
    # warnings / clarify bubbles from the previous student must be removed.
    try:
        from src.agent.lead_in.state_machine import LeadInTurnStateMachine

        sm = LeadInTurnStateMachine(session_manager)
        display_clear = {
            "last_applied_fields": {},
            "low_confidence_display": {},
            "clarifying_questions": [],
            "last_trace": [],
            "feedback_dismissed": False,
        }
        if reset_state_machine:
            state = sm.get_state()
            state.reset_all()
            sm.update(
                phase="idle",
                turn=0,
                pending_text="",
                running_hash="",
                running_ts=0.0,
                retry_count=0,
                tools_failed=False,
                progress_steps=[],
                progress_text="",
                progress_variant="default",
                intent_blocked=False,
                intent_gate_result=None,
                pydantic_messages=[],
                conversation_turns=[],
                last_path="",
                last_error=None,
                form_expander_open=False,
                **display_clear,
            )
        else:
            sm.update(**display_clear)
    except Exception:
        pass

    _log.info("APPLY_FORM | reset_lead_in_profile 已清空上轮 LeadIn 状态")


def _warm_university_cache(session_manager: Any) -> None:
    """预热院校候选缓存，避免 LeadIn 写入时 selectbox 因选项未就绪回退到 index=0。"""
    existing = session_manager.get(DEFAULT_FORM_KEYS.background_universities_cache)
    if existing:
        n = len(existing.keys()) if isinstance(existing, dict) else len(existing)
        if n > 0:
            return
    try:
        from src.pages.prediction.input_form_components.background_ui import (
            _generate_university_options,
        )

        page_state = machine_learning_model.resource_loader()
        if page_state is not None and page_state.cases_df is not None:
            options = _generate_university_options(session_manager, page_state.cases_df)
            if options:
                session_manager.set(background_universities_cache=options)
                _log.info("APPLY_FORM university | 预热院校缓存 %d 条", len(options))
    except Exception as exc:
        _log.warning("APPLY_FORM university | 无法预热院校缓存: %s", exc)


def _clear_unset_language(session_manager: Any, applied: dict[str, Any]) -> None:
    """LeadIn 未写入语言时，清除 UI 占位默认分，避免自动预测误用 6.5。"""
    if "language_score" in applied or "language_type" in applied:
        return
    session_manager.set(
        language_score_input=None,
        **{DEFAULT_FORM_KEYS.language_score_user_provided: False},
    )
    if DEFAULT_WIDGET_KEYS.language_score in st.session_state:
        del st.session_state[DEFAULT_WIDGET_KEYS.language_score]


def apply_lead_in_to_form(ctx: StudentContext, session_manager: Any) -> dict[str, Any]:
    bg = ctx.extracted_background or {}
    if not bg or not any(v for v in bg.values() if v):
        _log.info("APPLY_FORM | extracted_background 为空，跳过")
        return {}

    uni_cache_n = len(_get_list(session_manager, DEFAULT_FORM_KEYS.background_universities_cache))
    major_cache_raw = session_manager.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
    major_cache_n = (
        len(major_cache_raw.get("majors_display", [])) if isinstance(major_cache_raw, dict) else 0
    )
    _log.info(
        "APPLY_FORM START | bg_keys=%s | uni_cache=%d major_cache=%d%s",
        list(bg.keys()),
        uni_cache_n,
        major_cache_n,
        "  [警告:院校缓存空,将写入原值未经候选校验]" if uni_cache_n == 0 else "",
    )

    applied: dict[str, Any] = {}

    _warm_university_cache(session_manager)

    # ── university ──
    uni = bg.get("university")
    if uni and isinstance(uni, str) and uni.strip():
        uni_raw = uni.strip()
        # Category alias → concrete school (e.g. "985" → "北京大学")
        if is_school_category_alias(uni_raw):
            resolved = resolve_background_school(uni_raw)
            _log.info("APPLY_FORM university | 类别别名 %s → %s", uni_raw, resolved)
            # 保存别名信息供 UI 展示层次标签
            session_manager.set(background_university_alias=uni_raw)
            session_manager.set(
                background_university_display_label=get_category_display_label(uni_raw)
            )
            uni_raw = resolved
        else:
            # 非类别别名时清除之前的标签
            session_manager.set(background_university_alias=None)
            session_manager.set(background_university_display_label=None)
        uni_options = _get_list(session_manager, DEFAULT_FORM_KEYS.background_universities_cache)
        if uni_options:
            matched, conf = _fuzzy_match(uni_raw, uni_options)
        else:
            matched, conf = uni_raw, 0.0
        value = matched if conf >= 0.65 else uni_raw
        st.session_state[DEFAULT_WIDGET_KEYS.background_university] = value
        session_manager.set(background_university=value)
        _update_user_history(session_manager, "background_university", value)
        applied["background_university"] = value
        if conf < 0.65 and uni_options:
            _track_low_confidence(session_manager, "background_university", uni_raw, matched)
            _log.warning(
                "APPLY_FORM university | raw=%s best=%s conf=%.2f < 0.65 WRITE_RAW(low_conf)",
                uni_raw,
                matched,
                conf,
            )
        else:
            _log.info(
                "APPLY_FORM university | raw=%s → matched=%s conf=%.2f options=%d WRITE",
                uni_raw,
                matched,
                conf,
                len(uni_options),
            )

    # ── major ──
    # 预热专业缓存：首次加载时 background_majors_cache 可能尚未填充，
    # 导致 fuzzy_match 无选项可用 → 写入原始 LLM 值 → selectbox 渲染时
    # 值不在 options 列表中被 Streamlit 静默丢弃，最终 form_data 丢失该字段。
    _major_cache = session_manager.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
    if not _major_cache or not _major_cache.get("majors_display"):
        try:
            page_state = machine_learning_model.resource_loader()
            if page_state is not None and page_state.cases_df is not None:
                generated = _generate_major_options(page_state.cases_df)
                if generated and generated.get("majors_display"):
                    session_manager.set(background_majors_cache=generated)
                    _log.info(
                        "APPLY_FORM major | 预热专业缓存 %d 条",
                        len(generated.get("majors_display", [])),
                    )
        except Exception as exc:
            _log.warning("APPLY_FORM major | 无法预热专业缓存: %s", exc)

    major = bg.get("major")
    if major and isinstance(major, str) and major.strip():
        major_raw = major.strip()
        # Expand ASCII abbreviations to Chinese canonical name before fuzzy match.
        # isascii() guard: Chinese names like "软件工程" are already canonical and
        # should go directly to fuzzy match — translating them to English via the
        # discipline map and then matching against Chinese options produces false positives.
        query = major_raw
        if major_raw.isascii():
            for key, val in _BG_MAJOR_ALIAS_MAP.items():
                if key.casefold() == major_raw.casefold():
                    query = val
                    _log.info("APPLY_FORM major | alias %s → %s", major_raw, query)
                    break
        major_cache = session_manager.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
        major_options = major_cache.get("majors_display", [])
        if major_options:
            matched, conf = _fuzzy_match(query, major_options)
        else:
            matched, conf = query, 0.0
        value = matched if conf >= 0.65 else major_raw
        st.session_state[DEFAULT_WIDGET_KEYS.background_major] = value
        _update_user_history(session_manager, "background_major_original", value)
        applied["background_major"] = value
        if conf < 0.65 and major_options:
            _track_low_confidence(session_manager, "background_major", major_raw, matched)
            _log.warning(
                "APPLY_FORM major | raw=%s best=%s conf=%.2f < 0.65 WRITE_RAW(low_conf)",
                major_raw,
                matched,
                conf,
            )
        else:
            _log.info(
                "APPLY_FORM major | raw=%s → matched=%s conf=%.2f options=%d WRITE%s",
                major_raw,
                matched,
                conf,
                len(major_options),
                "  [注意:matched≠raw,缓存映射改写]" if matched != major_raw else "",
            )

    # ── major_2 (second major / dual degree) ──
    major_2 = bg.get("major_2")
    if major_2 and isinstance(major_2, str) and major_2.strip():
        major_2_raw = major_2.strip()
        major_cache = session_manager.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
        major_options_2 = major_cache.get("majors_display", [])
        if major_options_2:
            matched_2, conf_2 = _fuzzy_match(major_2_raw, major_options_2)
        else:
            matched_2, conf_2 = major_2_raw, 0.0
        value_2 = matched_2 if conf_2 >= 0.65 else major_2_raw
        st.session_state[DEFAULT_WIDGET_KEYS.background_major_2] = value_2
        _update_user_history(session_manager, "background_major_2_original", value_2)
        session_manager.set(background_major_2_original=value_2)
        # 从 agent 推断的 degree_type 决定辅修/双学位；无法判断时默认辅修
        inferred_degree_type = bg.get("degree_type", "")
        is_dual = inferred_degree_type == "双学位"
        if DEFAULT_WIDGET_KEYS.dual_degree_type not in st.session_state:
            st.session_state[DEFAULT_WIDGET_KEYS.dual_degree_type] = inferred_degree_type or "辅修"
        session_manager.set(is_dual_degree=is_dual, dual_alpha=1.0 if is_dual else 0.85)
        applied["background_major_2"] = value_2
        if conf_2 < 0.65 and major_options_2:
            _track_low_confidence(session_manager, "background_major_2", major_2_raw, matched_2)

    # ── gpa ──
    gpa = bg.get("gpa")
    if gpa is not None and isinstance(gpa, (int, float)) and float(gpa) > 0:
        gpa_val = round(float(gpa), 2)
        # Trust LLM's gpa_scale inference; fall back to value-based heuristic.
        llm_scale = bg.get("gpa_scale")
        if (
            llm_scale
            and isinstance(llm_scale, str)
            and llm_scale.strip() in ("4.0", "4.3", "5.0", "10", "100")
        ):
            scale = llm_scale.strip()
        elif text_scale := _infer_gpa_scale_from_text(ctx.raw_input or ""):
            scale = text_scale
        elif gpa_val > 10:
            scale = "100"
        elif gpa_val > 5.0:
            scale = "10"
        elif gpa_val > 4.3:
            scale = "5.0"
        elif gpa_val > 4.0:
            scale = "4.3"
        else:
            scale = "4.0"
        session_manager.set(gpa_scale=scale)
        st.session_state[DEFAULT_WIDGET_KEYS.gpa_scale] = scale
        session_manager.set(gpa_raw_input=gpa_val)
        st.session_state[DEFAULT_WIDGET_KEYS.gpa_raw_input] = gpa_val
        applied["gpa"] = gpa_val
        applied["gpa_scale"] = scale

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
        session_manager.set(
            language_score_input=ls,
            **{DEFAULT_FORM_KEYS.language_score_user_provided: True},
        )
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
            dropped: list[str] = []
            for s in schools_list:
                resolved = resolve_target_schools(s)
                if resolved:
                    expanded.extend(resolved)
                    continue
                # 非"港N"类别名：可能是受支持院校的规范全名（如"香港大学"），
                # resolve_target_schools 只认带数字的别名，会误丢全名 → 此处兜底匹配。
                canon = _match_supported_school(s)
                if canon:
                    expanded.append(canon)
                else:
                    dropped.append(s)
            expanded = list(dict.fromkeys(expanded))  # 去重，保序
            if dropped:
                applied["_dropped_schools"] = dropped
                _log.warning("APPLY_FORM target_schools | 丢弃越界/未知院校: %s", dropped)
            if expanded:
                session_manager.set(selected_target_universities=expanded)
                st.session_state[DEFAULT_WIDGET_KEYS.target_universities] = expanded
                applied["target_schools"] = expanded
                _log.info("APPLY_FORM target_schools | %s → 展开 %s", schools_list, expanded)

    # ── target majors ──
    majors = bg.get("target_majors")
    if majors:
        majors_list = _to_list(majors)
        if majors_list:
            # Smart match: exact/alias match first, LLM-expanded multi-search
            # as fallback (e.g. "语言学" → all Linguistics-related programs)
            matched_majors: list[str] = []
            for m in majors_list:
                matched_majors.extend(fuzzy_match_major_smart(m, limit=10))
            # Deduplicate while preserving order
            seen: set[str] = set()
            unique: list[str] = []
            for name in matched_majors:
                if name not in seen:
                    seen.add(name)
                    unique.append(name)
            session_manager.set(selected_target_majors=unique)
            st.session_state[DEFAULT_WIDGET_KEYS.target_majors] = unique
            applied["target_majors"] = unique

            # ── cross-reference: filter majors to those offered by target schools ──
            target_schools = applied.get("target_schools") or []
            if target_schools and len(unique) > 1:
                cross = cross_reference_school_majors(target_schools, unique)
                if cross != unique:
                    session_manager.set(selected_target_majors=cross)
                    st.session_state[DEFAULT_WIDGET_KEYS.target_majors] = cross
                    applied["target_majors"] = cross

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
            # Sync the widget-level session_state key used by
            # standardized_test_ui.render_standardized_test_section().
            # Without this, the text_input renders empty and the UI's
            # cleanup logic (line 82-83) overwrites current_exam_score → None.
            widget_key = f"standardized_test_score_text_{et}"
            st.session_state[widget_key] = str(int(es))

    # ── experiences (routed through user_history_data, read by experience_ui) ──
    for exp_key in ("research", "internship", "paper", "award"):
        _apply_experience(session_manager, bg, exp_key)

    # ── build expander summary ──
    _set_form_summary(session_manager, applied)

    _clear_unset_language(session_manager, applied)

    # Track which key fields are missing after lead_in extraction
    _track_missing_fields(session_manager, bg, applied)

    written = {k: v for k, v in applied.items() if not k.startswith("_")}
    _log.info(
        "APPLY_FORM DONE | written_fields=%s | dropped_schools=%s | low_conf=%s",
        list(written.keys()),
        applied.get("_dropped_schools", []),
        list((session_manager.get("lead_in_low_confidence_fields", {}) or {}).keys()),
    )

    # Phase 3 dual-write: sync extraction results to state machine
    try:
        from src.agent.lead_in.state_machine import LeadInTurnStateMachine

        sm = LeadInTurnStateMachine(session_manager)
        sm.update(
            last_applied_fields=written,
            form_expander_open=True,
        )
    except Exception:
        pass

    return applied


KEY_FIELDS = {
    "university": "院校",
    "major": "专业",
    "gpa": "GPA",
    "language_type": "语言成绩",
    "country": "目标地区",
}


def _track_missing_fields(session_manager: Any, bg: dict, applied: dict) -> None:
    """Record key fields that are still missing or need user confirmation."""
    missing: list[str] = []
    for key, label in KEY_FIELDS.items():
        if not bg.get(key) and key not in applied:
            missing.append(label)
    if missing:
        session_manager.set(lead_in_missing_fields=missing)
    else:
        session_manager.set(lead_in_missing_fields=None)
    # Flag low-confidence fields that need user review
    low_conf = session_manager.get("lead_in_low_confidence_fields", {}) or {}
    confirm: list[str] = []
    for field_key in low_conf:
        if field_key == "background_university":
            confirm.append("院校(需确认)")
        elif field_key == "background_major":
            confirm.append("专业(需确认)")
        elif field_key == "background_major_2":
            confirm.append("辅修专业(需确认)")
    session_manager.set(lead_in_low_confidence_labels=confirm if confirm else None)


def _set_form_summary(session_manager: Any, applied: dict) -> None:
    parts: list[str] = []
    if "background_university" in applied:
        display_label = session_manager.get("background_university_display_label")
        uni = display_label or applied["background_university"]
        parts.append(uni if len(uni) <= 12 else uni[:10] + "…")
    if "background_major" in applied:
        maj = applied["background_major"]
        parts.append(maj if len(maj) <= 10 else maj[:8] + "…")
    if "background_major_2" in applied:
        m2 = applied["background_major_2"]
        m2_short = m2 if len(m2) <= 8 else m2[:6] + "…"
        parts.append(f"+{m2_short}")
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


def _track_low_confidence(
    session_manager: Any, field: str, raw_value: str, best_guess: str
) -> None:
    """Record fields where fuzzy_match had low confidence so UI can flag them."""
    tracked: dict[str, dict[str, str]] = (
        session_manager.get("lead_in_low_confidence_fields", {}) or {}
    )
    tracked[field] = {"raw": raw_value, "best_guess": best_guess}
    session_manager.set(lead_in_low_confidence_fields=tracked)


@lru_cache(maxsize=1)
def _supported_target_schools() -> tuple[str, ...]:
    """Target school canonical names from config_loader, cached once."""
    from src.utils.schools.config_loader import TARGET_COUNTRY_UNIVERSITY_MAP

    out: list[str] = []
    for schools in TARGET_COUNTRY_UNIVERSITY_MAP.values():
        out.extend(schools)
    return tuple(out)


def _match_supported_school(name: str) -> str:
    """Match a target school name to canonical supported school name.

    Uses exact match → substring → rapidfuzz partial_ratio (85 cutoff).
    Returns canonical name on match, or '' if out of scope.
    """
    s = str(name).strip()
    if not s:
        return ""
    supported = _supported_target_schools()
    if s in supported:
        _log.debug("MATCH_SCHOOL | %s → exact match", s)
        return s
    for k in supported:
        if s in k or k in s:
            _log.debug("MATCH_SCHOOL | %s → substring match %s", s, k)
            return k
    m = process.extractOne(s, list(supported), scorer=fuzz.partial_ratio, score_cutoff=85)
    if m:
        _log.debug("MATCH_SCHOOL | %s → rapidfuzz %.0f → %s", s, m[1], m[0])
        return m[0]
    _log.debug("MATCH_SCHOOL | %s → no match (out of scope)", s)
    return ""


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
    # Use router-extracted count if available, else fall back to existing or 1
    extracted = bg.get(f"{key}_count")
    if isinstance(extracted, int) and extracted > 0:
        count = extracted
    else:
        count = max(uh.get(f"{key}_count", 0), 1)
    uh[f"{key}_count"] = count
    session_manager.set(user_history_data=uh)
    st.session_state[f"{key}_count_input"] = count
    st.session_state[f"{key}_details_input"] = text.strip()
