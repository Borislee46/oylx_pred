from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.app_data import load_school_major_details_df
from src.pages.prediction.result_display._similar_cases import render_similar_cases
from src.pages.prediction.result_display.major_detail import (
    render_major_detail_compact_html,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce, prob_to_pct

_logger = setup_logger("page3", "prediction")

_TOP_SCHOOLS = 12

_TIER_KEY_MAP = {"保底": "safety", "适中": "target", "冲刺": "reach"}

_TIER_CONFIG: dict[str, dict[str, str]] = {
    "reach": {
        "label": "冲刺",
        "color": "#f97316",
        "desc": "录取把握较低，建议重点准备",
        "accent_bar": "#f97316",
    },
    "target": {
        "label": "目标",
        "color": "#3b82f6",
        "desc": "录取把握适中，可纳入申请主力",
        "accent_bar": "#3b82f6",
    },
    "safety": {
        "label": "保底",
        "color": "#22c55e",
        "desc": "录取把握较高，可设为稳妥选项",
        "accent_bar": "#22c55e",
    },
}

_TIER_ORDER = ["reach", "target", "safety"]

_KNN_RESULT_CACHE = "_hk_knn_results_cache"
_KNN_LOADED_SLOTS = "_hk_knn_loaded_slots"


def _knn_cache_key(student: dict) -> str:
    return "|".join(
        [
            str(student.get("background_university", "")),
            str(student.get("background_major", "")),
            str(student.get("gpa", "")),
            str(student.get("lang_score", "")),
            str(student.get("target_university", "")),
            str(student.get("target_major", "")),
        ]
    )


def _render_knn_lazy(student: dict, slot: str, major_idx: int) -> None:
    slot_id = f"{slot}_{major_idx}"
    cache_key = _knn_cache_key(student)
    loaded: set[str] = set(st.session_state.get(_KNN_LOADED_SLOTS, ()))
    result_cache: dict = dict(st.session_state.get(_KNN_RESULT_CACHE, {}))
    show_content = slot_id in loaded or cache_key in result_cache

    if not show_content:
        label = "查看参考案例"
        with st.container(key=f"hk_knn_cta_{slot_id}"):
            if st.button(
                label,
                key=f"hk_knn_btn_{slot_id}",
                type="secondary",
                width="content",
                icon=":material/groups:",
            ):
                st.session_state[_KNN_LOADED_SLOTS] = loaded | {slot_id}
                st.rerun(scope="fragment")
        return

    if cache_key not in result_cache:
        from src.adjustment.knn_retrieval import retrieve_similar_cases

        st.session_state[_KNN_RESULT_CACHE] = {
            **result_cache,
            cache_key: retrieve_similar_cases(student, k=3),
        }
        result_cache = st.session_state[_KNN_RESULT_CACHE]

    render_similar_cases(student, slot, cached_result=result_cache[cache_key])


@st.cache_data(show_spinner=False)
def _cached_school_details() -> pd.DataFrame:
    return load_school_major_details_df()


def _get_major_reqs(university: str, major: str) -> dict[str, str]:
    df = _cached_school_details()
    if df.empty or "学校" not in df.columns:
        return {}
    sub = df[df["学校"] == university]
    if sub.empty:
        return {}
    for col in ("专业英文名称", "专业中文名称"):
        if col in sub.columns:
            hit = sub[sub[col] == major]
            if not hit.empty:
                row = hit.iloc[0]
                return {
                    "gpa": str(row.get("GPA要求", "") or "").strip(),
                    "ielts": str(row.get("IELTS", "") or "").strip(),
                    "toefl": str(row.get("TOEFL", "") or "").strip(),
                }
    return {}


def _prob_bar_color(tier_key: str) -> str:
    cfg = _TIER_CONFIG.get(tier_key, _TIER_CONFIG["target"])
    c = cfg["color"]
    return f"linear-gradient(90deg, {c}, {c}dd)"


def _group_by_school(
    unified: list[dict[str, Any]],
) -> list[tuple[str, list[dict[str, Any]]]]:
    from src.pages.prediction.core import UNIVERSITY_ORDER_MAP

    groups: dict[str, list[dict[str, Any]]] = {}
    for r in unified:
        uni = str(r.get("university", ""))
        groups.setdefault(uni, []).append(r)
    for majors in groups.values():
        majors.sort(key=lambda r: float(r.get("similarity", 0) or 0), reverse=True)
        if len(majors) > 3:
            majors[:] = majors[:3]
    return sorted(
        groups.items(),
        key=lambda kv: (
            UNIVERSITY_ORDER_MAP.get(kv[0], 9999),
            -max(clip_probability_coerce(r.get("probability")) for r in kv[1]),
        ),
    )


def _render_reqs_tags(reqs: dict[str, str]) -> str:
    tags: list[str] = []
    gpa = reqs.get("gpa", "")
    if gpa:
        tags.append(f'<span class="hk-sc-req-tag">GPA ≥ {html.escape(gpa)}</span>')
    ielts = reqs.get("ielts", "")
    if ielts:
        tags.append(f'<span class="hk-sc-req-tag">IELTS {html.escape(ielts)}</span>')
    else:
        toefl = reqs.get("toefl", "")
        if toefl:
            tags.append(f'<span class="hk-sc-req-tag">TOEFL {html.escape(toefl)}</span>')
    if not tags:
        return ""
    return " ".join(tags)


def _ref_badge_html(school_name: str, majors: list[dict[str, Any]]) -> str:
    from src.adjustment.knn_retrieval import reference_pool_size

    major = str(majors[0].get("major", "")) if majors else ""
    try:
        n = reference_pool_size(school_name, major)
    except Exception:
        return ""
    if n <= 0:
        return ""
    text = f"{n} 位背景相似的学长学姐被录取"
    return f'<span class="hk-sc-ref">{text}</span>'


def _render_school_card_html(
    school_name: str,
    majors: list[dict[str, Any]],
    tier_key: str,
    ref_badge: str = "",
) -> str:
    cfg = _TIER_CONFIG[tier_key]
    accent = cfg["accent_bar"]
    prob_color = cfg["color"]
    school_name_esc = html.escape(school_name)

    major_rows: list[str] = []
    for m in majors:
        prob = clip_probability_coerce(m.get("probability"))
        major_name_raw = str(m.get("major", ""))
        major_name = html.escape(major_name_raw)
        bar_color = _prob_bar_color(tier_key)
        uni = str(m.get("university", ""))
        reqs_html = _render_reqs_tags(_get_major_reqs(uni, major_name_raw))

        major_rows.append(
            f'<div class="hk-sc-major-row">'
            f'<div class="hk-sc-major-top">'
            f'<span class="hk-sc-major-name">{major_name}</span>'
            f'<span class="hk-sc-major-prob" style="color:{prob_color};">{prob:.0%}</span>'
            f"</div>"
            f'<div class="hk-sc-major-bar-wrap">'
            f'<div class="hk-sc-major-bar-fill" style="width:{prob_to_pct(prob)}%;background:{bar_color};"></div>'
            f"</div>"
            + (f'<div class="hk-sc-major-reqs">{reqs_html}</div>' if reqs_html else "")
            + "</div>"
        )

    return (
        f'<div class="hk-school-card">'
        f'<div class="hk-sc-accent" style="background:{accent};"></div>'
        f'<div class="hk-sc-body">'
        f'<div class="hk-sc-header">'
        f'<span class="hk-sc-name">{school_name_esc}</span>' + ref_badge + f"</div>"
        f'<div class="hk-sc-major-list">{"".join(major_rows)}</div>'
        f"</div>"
        f"</div>"
    )


@st.fragment
def render_school_explorer(
    unified_results: list[dict[str, Any]] | None,
    input_data: dict[str, Any],
    canonical_tiers: dict[str, str] | None = None,
) -> None:
    unified = unified_results or []
    _logger.info("render_school_explorer: %d results", len(unified))
    if not unified:
        _logger.info("render_school_explorer: no results, skipping")
        return

    ranked_schools = _group_by_school(unified)[:_TOP_SCHOOLS]
    _logger.info(
        "render_school_explorer: %d schools after grouping (top %d)",
        len(ranked_schools),
        _TOP_SCHOOLS,
    )
    show_knn = True

    bg_uni = str(input_data.get("background_university", ""))
    bg_major = str(input_data.get("background_major", "") or "")
    gpa = float(input_data.get("gpa", 0) or 0)
    lang_score = float(input_data.get("language_score", 0) or 0)

    if canonical_tiers:
        tier_labels = [
            canonical_tiers.get(school_name, "适中") for school_name, _ in ranked_schools
        ]
    else:
        from src.agent.schemas import compute_tiers

        best_probs = [
            max(clip_probability_coerce(r.get("probability")) for r in majors)
            for _, majors in ranked_schools
        ]
        tier_labels = compute_tiers(best_probs)

    tier_buckets: dict[str, list[tuple[str, list[dict[str, Any]]]]] = {
        "reach": [],
        "target": [],
        "safety": [],
    }
    for idx, (school_name, majors) in enumerate(ranked_schools):
        cn_label = tier_labels[idx]
        tier_key = _TIER_KEY_MAP.get(cn_label, "target")
        tier_buckets[tier_key].append((school_name, majors))

    _logger.info(
        "render_school_explorer: tier dist — reach=%d target=%d safety=%d",
        len(tier_buckets["reach"]),
        len(tier_buckets["target"]),
        len(tier_buckets["safety"]),
    )

    section_sub = "按录取把握分层，展开可查看专业详情与参考案例"
    detail_label = "查看详情 · 参考案例"

    with st.container(key="hk_school_explorer"):
        st.html(
            '<div class="hk-section-head">院校推荐</div>'
            f'<div class="hk-section-sub">{section_sub}</div>'
        )

        for tier_key in _TIER_ORDER:
            schools_in_tier = tier_buckets[tier_key]
            cfg = _TIER_CONFIG[tier_key]

            st.html(
                f'<div class="hk-tier-header">'
                f'<span class="hk-tier-label" style="color:{cfg["color"]};">{cfg["label"]}</span>'
                f'<span class="hk-tier-desc">{cfg["desc"]}</span>'
                f'<span class="hk-tier-count">{len(schools_in_tier)} 所</span>'
                f"</div>"
            )

            if not schools_in_tier:
                st.html(
                    f'<div class="hk-tier-empty">当前背景暂无{cfg["label"]}层级的院校推荐</div>'
                )
                continue

            card_idx = 0
            with st.container(key=f"hk_sc_grid_{tier_key}"):
                for school_name, majors in schools_in_tier:
                    with st.container(key=f"hk_sc_wrap_{tier_key}_{card_idx}"):
                        ref_badge = _ref_badge_html(school_name, majors)
                        st.html(_render_school_card_html(school_name, majors, tier_key, ref_badge))

                        expander_key = f"hk_sc_exp_{tier_key}_{card_idx}"
                        exp = st.expander(detail_label, key=expander_key, on_change="rerun")
                        if exp.open:
                            with exp:
                                for k, m in enumerate(majors):
                                    uni = str(m.get("university", ""))
                                    major = str(m.get("major", ""))
                                    prob = clip_probability_coerce(m.get("probability"))
                                    prob_color = _TIER_CONFIG[tier_key]["color"]

                                    st.html(
                                        f'<div class="hk-sc-detail-major-header">'
                                        f'<span class="hk-sc-detail-major-name">{html.escape(major)}</span>'
                                        f'<span class="hk-sc-detail-major-prob" style="color:{prob_color};">录取概率 {prob:.0%}</span>'
                                        f"</div>"
                                    )

                                    detail_html = render_major_detail_compact_html(uni, major)
                                    if detail_html:
                                        st.html(detail_html)
                                    else:
                                        st.caption("暂无详细信息")

                                    if show_knn:
                                        st.html(
                                            '<div class="hk-sc-detail-section-label">相似案例</div>'
                                        )
                                        try:
                                            _render_knn_lazy(
                                                {
                                                    "background_university": bg_uni,
                                                    "background_major": bg_major,
                                                    "gpa": gpa,
                                                    "lang_score": lang_score,
                                                    "target_university": uni,
                                                    "target_major": major,
                                                },
                                                slot=f"{tier_key}_{card_idx}",
                                                major_idx=k,
                                            )
                                        except Exception:
                                            st.caption("相似案例暂不可用")

                                    if k < len(majors) - 1:
                                        st.divider()
                    card_idx += 1
