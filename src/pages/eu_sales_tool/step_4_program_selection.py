import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_CURRENT_STEP,
    KEY_EDUCATION_DATA,
    KEY_SCORE_KEY,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_STUDY_PROGRAM,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.steps import reset_selections_from_step


def render_step_4():
    education_data = st.session_state[KEY_EDUCATION_DATA]
    raw_data = education_data.get(st.session_state[KEY_SELECTED_COUNTRY], {})
    score_key = st.session_state.get(KEY_SCORE_KEY)

    available_programs = [
        p for p in raw_data.get("study_programs", []) if score_key in p.get("required_scores", [])
    ]

    if not available_programs:
        st.success("根据您当前选择的分数段，暂无匹配的留学项目。请尝试返回上一步更改分数评估。")
    else:
        for i, program in enumerate(available_programs):
            program_name = program.get("program_name", "未命名项目")
            program_desc = program.get("description", "暂无描述")

            short_desc = program_desc[:80] + "..." if len(program_desc) > 80 else program_desc

            button_text = f"**{program_name}**\n\n{short_desc}"

            help_info = f"项目详情：{program_desc}\n"
            if program.get("available_schools"):
                help_info += f"可申请院校：{', '.join(program['available_schools'][:3])}等\n"
            if program.get("available_majors"):
                help_info += f"专业方向：{', '.join(program['available_majors'][:3])}等"

            if st.button(
                button_text,
                key=f"study_program_{i}_{program_name}",
                type="primary",
                help=help_info,
            ):
                st.session_state[KEY_SELECTED_STUDY_PROGRAM] = program_name
                st.session_state[KEY_CURRENT_STEP] = 5
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        if len(available_programs) > 1:
            with st.expander("项目详细对比", expanded=True):
                for i, program in enumerate(available_programs, 1):
                    program_name = program.get("program_name", "未命名项目")
                    description = program.get("description", "暂无描述")
                    school_count = len(program.get("available_schools", []))
                    major_count = len(program.get("available_majors", []))

                    program_info = f"""**{i}. {program_name}**  
{description}   
院校数量：{school_count} 所 | 专业数量：{major_count} 个"""

                    st.success(program_info)

    if st.button("上一步", type="primary", key="change_score_step4"):
        reset_selections_from_step(3)
        st.session_state[KEY_CURRENT_STEP] = 3
        st.rerun()


def _render_statistics(study_program, country_data, raw_data):
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        with st.expander("可选院校", expanded=True):
            school_count = len(study_program.get("available_schools", []))
            st.metric("院校总数", school_count, "所", label_visibility="collapsed")
    with col2:
        with st.expander("专业方向", expanded=True):
            major_count = len(study_program.get("available_majors", []))
            st.metric("专业总数", major_count, "类", label_visibility="collapsed")
    with col3:
        with st.expander("授课语言", expanded=True):
            language_count = len(country_data.get("languages", []))
            st.metric("语言总数", language_count, "种", label_visibility="collapsed")
    with col4:
        with st.expander("留学优势", expanded=True):
            if raw_data.get("descriptions"):
                advantage_count = len(raw_data["descriptions"])
                st.metric("优势总数", advantage_count, "条", label_visibility="collapsed")
            else:
                st.metric("数据完整度", "100%", "", label_visibility="collapsed")
