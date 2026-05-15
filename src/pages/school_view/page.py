"""School Perspective page — 学校画像 + What-If 模拟 + 学校对比."""

import pandas as pd
import streamlit as st

from .data_loader import FEATURE_LABELS, PROFILE_FEATURES, load_full_cases
from .school_form import render_school_form
from .school_profiles import MIN_SAMPLES_FOR_PROFILE, SchoolProfileCalculator
from .what_if_simulator import SCENARIOS, WhatIfSimulator


@st.cache_resource
def _get_simulator() -> WhatIfSimulator:
    return WhatIfSimulator()


def _fmt_val(val, feat: str) -> str:
    """Format a feature value for display."""
    if val is None:
        return "—"
    if feat == "gpa":
        return f"{val:.2f}"
    if feat == "language_score":
        return f"{val:.2f}"
    if feat in ("research_count", "internship_count", "paper_count", "award_count"):
        return f"{val:.0f}段"
    return str(val)


def _fmt_pct(pct) -> str:
    if pct is None:
        return "—"
    return f"{pct:.0f}%"


def _render_profile_tab(profiler: SchoolProfileCalculator, form_data: dict):
    """Tab 1: School profile cards with percentile bars and gap analysis."""
    schools = form_data["target_schools"]
    student = form_data["student_values"]

    if not schools:
        st.info("请先在表单中选择至少一所目标院校")
        return

    for school in schools:
        profile = profiler.compute_school_profile(school)
        if profile.get("insufficient"):
            st.warning(f"**{school}**: 录取样本不足 ({profile.get('n_admitted', 0)} < {MIN_SAMPLES_FOR_PROFILE})")
            continue

        n = profile["n_admitted"]
        insufficient = profile.get("insufficient", False)

        with st.container(border=True):
            st.subheader(f"🎯 {school}")
            if insufficient:
                st.caption(f"录取样本不足 ({n} < {MIN_SAMPLES_FOR_PROFILE})，无法生成画像")
                continue
            st.caption(f"基于 {n} 份历史录取案例")

            # Profile stats + percentile table
            rows = []
            for feat in PROFILE_FEATURES:
                feat_info = profile["features"].get(feat, {})
                p50 = feat_info.get("p50")
                p25 = feat_info.get("p25")
                p75 = feat_info.get("p75")
                student_val = student.get(feat)

                pct = profiler.get_student_percentile(school, feat, student_val) if student_val is not None else None

                if p50 is not None:
                    rows.append({
                        "维度": FEATURE_LABELS.get(feat, feat),
                        "录取者P50": _fmt_val(p50, feat),
                        "P25-P75": f"{_fmt_val(p25, feat)} — {_fmt_val(p75, feat)}" if p25 and p75 else "—",
                        "你的值": _fmt_val(student_val, feat) if student_val else "—",
                        "你的百分位": _fmt_pct(pct),
                    })

            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

            # Gap analysis
            gaps = profiler.get_gap_analysis(school, student)
            if gaps:
                st.caption("**差距分析**")
                worst = [g for g in gaps if g["gap"] < 0][:3]
                if worst:
                    for g in worst:
                        label = g["label"]
                        pct_str = f"(P{_fmt_pct(g['percentile'])})" if g["percentile"] is not None else ""
                        st.write(f"- {label}: 差 {abs(g['gap']):.3f} {pct_str}，录取者中位 {_fmt_val(g['p50'], g['feature'])}")
                else:
                    st.success("所有维度均高于录取者中位水平")


def _render_whatif_tab(form_data: dict):
    """Tab 2: What-If counterfactual simulation."""
    schools = form_data["target_schools"]

    if not schools:
        st.info("请先在表单中选择至少一所目标院校")
        return

    with st.spinner("加载模型并运行模拟..."):
        try:
            sim = _get_simulator()
        except Exception as e:
            st.error(f"模型加载失败: {e}")
            return
        result = sim.simulate(form_data, schools)

    if not result:
        st.warning("无法运行模拟，请检查输入数据")
        return

    roi_df = sim.compute_roi_table(result)

    st.subheader("📊 各场景概率对比")
    st.dataframe(roi_df, use_container_width=True, hide_index=True)

    # Simple ROI summary
    st.subheader("📈 边际增益排序")
    school_baseline = result.get("school_baseline", {})
    school_scenarios = result.get("school_scenarios", {})
    scenarios_excl_baseline = [s for s in SCENARIOS if s["key"] != "baseline"]

    roi_rows = []
    for sc in scenarios_excl_baseline:
        total_delta = 0
        for school in schools:
            base = school_baseline.get(school, 0)
            sc_prob = school_scenarios.get(sc["key"], {}).get(school, base)
            total_delta += sc_prob - base
        avg_delta = total_delta / len(schools) if schools else 0
        roi_rows.append({
            "干预": sc["label"],
            "平均概率变化": f"{avg_delta:+.1%}",
        })

    roi_rows.sort(key=lambda r: float(r["平均概率变化"].rstrip("%").replace("+", "")), reverse=True)
    st.dataframe(pd.DataFrame(roi_rows), use_container_width=True, hide_index=True)

    if roi_rows:
        best = roi_rows[0]
        st.success(f"ROI 最高: **{best['干预']}** → 平均概率变化 **{best['平均概率变化']}**")


def _render_comparison_tab(profiler: SchoolProfileCalculator, form_data: dict):
    """Tab 3: Side-by-side school comparison."""
    schools = form_data["target_schools"]
    student = form_data["student_values"]

    if not schools:
        st.info("请先在表单中选择至少一所目标院校")
        return

    if len(schools) < 2:
        st.info("请至少选择 2 所目标院校进行对比")
        return

    profiles = []
    for school in schools:
        p = profiler.compute_school_profile(school)
        if p:
            profiles.append(p)

    if len(profiles) < 2:
        st.warning("有数据的院校不足 2 所，无法对比")
        return

    # Side-by-side comparison for each feature
    for feat in PROFILE_FEATURES:
        label = FEATURE_LABELS.get(feat, feat)
        st.caption(f"**{label}**")
        cols = st.columns(len(profiles))
        for col, p in zip(cols, profiles):
            uni = p["university"]
            feat_info = p.get("features", {}).get(feat, {})
            p50 = feat_info.get("p50")
            n = feat_info.get("n_valid", 0)
            student_val = student.get(feat)
            pct_val = profiler.get_student_percentile(uni, feat, student_val) if student_val else None

            with col:
                st.metric(
                    label=uni,
                    value=_fmt_val(p50, feat) if p50 else "—",
                    delta=_fmt_pct(pct_val) + " 百分位" if pct_val is not None else None,
                )
                st.caption(f"n={n}" if n else "数据不足")


def render_school_view():
    st.title("🏫 学校视角")
    st.caption("了解目标院校的录取画像，模拟背景提升效果")

    # Load data
    cases_df = load_full_cases()
    profiler = SchoolProfileCalculator(cases_df)

    # Form
    form_data = render_school_form(cases_df)

    if form_data is None:
        return

    if not form_data.get("target_schools"):
        st.warning("请至少选择一所目标院校")
        return

    st.divider()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["学校画像", "What-If 模拟", "学校对比"])

    with tab1:
        _render_profile_tab(profiler, form_data)

    with tab2:
        _render_whatif_tab(form_data)

    with tab3:
        _render_comparison_tab(profiler, form_data)
