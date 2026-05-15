"""Lightweight standalone form for school view. Reuses GPAConverter + normalize_language_score."""

import pandas as pd
import streamlit as st

from src.pages.prediction.core.utils import normalize_language_score
from src.pages.prediction.input_form_components.gpa_converter import GPAConverter


def render_school_form(cases_df: pd.DataFrame) -> dict | None:
    """Render a simplified background + target form. Returns form_data dict or None."""

    st.subheader("📋 输入背景信息")

    bg_uni_list = sorted(cases_df["background_university"].dropna().astype(str).unique().tolist())

    col1, col2 = st.columns(2)

    with col1:
        bg_uni = st.selectbox("本科院校", options=bg_uni_list, key="sv_bg_uni")

        bg_majors = sorted(
            cases_df.loc[
                cases_df["background_university"] == bg_uni, "background_major"
            ].dropna().astype(str).unique().tolist()
        ) if bg_uni else []
        bg_major_default = bg_majors[0] if bg_majors else None
        bg_major = st.selectbox(
            "本科专业", options=bg_majors if bg_majors else ["请选择"],
            index=0 if bg_major_default else None,
            key="sv_bg_major",
        )

        gpa_scale = st.selectbox("GPA 分制", options=["4.0", "5.0", "100"], key="sv_gpa_scale")
        gpa_raw = st.number_input("GPA", min_value=0.0, max_value=100.0, value=3.0, step=0.1, key="sv_gpa")

        # Convert GPA
        gpa_4 = None
        if gpa_raw > 0:
            try:
                result = GPAConverter.convert_gpa_by_rules(float(gpa_raw), gpa_scale, bg_uni)
                if result is not None:
                    gpa_4 = result
            except Exception:
                pass
            if gpa_4 is None:
                if gpa_scale == "4.0":
                    gpa_4 = float(gpa_raw)
                elif gpa_scale == "5.0":
                    gpa_4 = float(gpa_raw) * 4.0 / 5.0
                else:
                    gpa_4 = float(gpa_raw) / 100 * 4.0
        st.caption(f"→ 4.0制: {gpa_4:.2f}" if gpa_4 else "请输入GPA")

    with col2:
        lang_type = st.selectbox("语言类型", options=["雅思", "托福"], key="sv_lang_type")
        if lang_type == "雅思":
            lang_score = st.number_input("雅思分数", min_value=0.0, max_value=9.0, value=6.5, step=0.5, key="sv_ielts")
        else:
            lang_score = st.number_input("托福分数", min_value=0.0, max_value=120.0, value=90.0, step=1.0, key="sv_toefl")

        lang_norm = normalize_language_score(lang_score, lang_type) if lang_score > 0 else None
        st.caption(f"→ 归一化: {lang_norm:.3f}" if lang_norm else "请输入语言成绩")

        research_count = int(st.number_input("科研经历(段)", min_value=0, max_value=20, value=0, step=1, key="sv_research"))
        internship_count = int(st.number_input("实习经历(段)", min_value=0, max_value=20, value=0, step=1, key="sv_internship"))
        paper_count = int(st.number_input("论文发表(篇)", min_value=0, max_value=20, value=0, step=1, key="sv_paper"))
        award_count = int(st.number_input("获奖经历(项)", min_value=0, max_value=20, value=0, step=1, key="sv_award"))

    st.divider()
    st.subheader("🎯 选择目标院校")

    target_school_list = sorted(cases_df["target_university"].dropna().astype(str).unique().tolist())
    target_schools = st.multiselect(
        "目标院校（可多选）",
        options=target_school_list,
        key="sv_targets",
    )

    submit = st.button("开始分析", type="primary", use_container_width=True)

    if not submit:
        return None

    student_values = {
        "gpa": gpa_4,
        "language_score": lang_norm,
        "research_count": float(research_count) if research_count > 0 else 0.0,
        "internship_count": float(internship_count) if internship_count > 0 else 0.0,
        "paper_count": float(paper_count) if paper_count > 0 else 0.0,
        "award_count": float(award_count) if award_count > 0 else 0.0,
    }

    return {
        "background_university": bg_uni,
        "background_major": bg_major if bg_major != "请选择" else None,
        "gpa": gpa_4,
        "language_type": lang_type,
        "language_score_raw": lang_score,
        "language_score": lang_norm,
        "research_count": research_count,
        "internship_count": internship_count,
        "paper_count": paper_count,
        "award_count": award_count,
        "student_values": student_values,
        "target_schools": target_schools,
    }
