import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_CURRENT_STEP,
    KEY_EDUCATION_DATA,
    KEY_SELECTED_COUNTRY,
    KEY_SHOW_COUNTRY_DETAILS,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.get_countries import get_standardized_countries


def render_step_1():
    COUNTRIES = get_standardized_countries()
    education_data = st.session_state[KEY_EDUCATION_DATA]

    for i, (country_key, country_data) in enumerate(COUNTRIES.items()):
        full_description = country_data["description"]
        short_description = (
            full_description[:60] + "..." if len(full_description) > 60 else full_description
        )

        button_text = f"**{country_data['display_name']}**\n\n{short_description}"

        if st.button(
            button_text,
            key=f"country_{i}_{country_key}",
            type="primary",
            help=f"完整介绍：{full_description}",
        ):
            st.session_state[KEY_SELECTED_COUNTRY] = country_key
            st.session_state[KEY_CURRENT_STEP] = 2
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("查看国家详细信息", type="primary", key="toggle_country_details"):
            st.session_state[KEY_SHOW_COUNTRY_DETAILS] = not st.session_state.get(
                KEY_SHOW_COUNTRY_DETAILS, False
            )
            st.rerun()

    if st.session_state.get("show_quick_preview", False):
        with st.expander("所有国家完整介绍", expanded=True):
            for country_key, country_data in COUNTRIES.items():
                st.markdown(f"**{country_data['display_name']}**")
                st.write(country_data["description"])

    if st.session_state.get(KEY_SHOW_COUNTRY_DETAILS, False):
        selected_country_for_preview = st.selectbox(
            "选择国家查看详情",
            list(COUNTRIES.keys()),
            format_func=lambda key: COUNTRIES[key]["display_name"],
            key="country_preview",
        )

        if selected_country_for_preview:
            raw_data_preview = education_data.get(selected_country_for_preview, {})

            country_detail = COUNTRIES[selected_country_for_preview]
            raw_data = education_data.get(selected_country_for_preview, {})

            with st.expander(f"{country_detail['display_name']} 国家优势", expanded=False):
                if raw_data.get("descriptions"):
                    for i, desc in enumerate(raw_data["descriptions"]):
                        with st.container(border=True):
                            st.success(f"**优势 {i + 1}:** {desc}")

        if selected_country_for_preview:
            raw_data_for_lists = education_data.get(selected_country_for_preview, {})
            all_schools = set()
            for program in raw_data_for_lists.get("study_programs", []):
                for school in program.get("available_schools", []):
                    all_schools.add(school.strip())
            schools_full = list(all_schools)

            all_majors = set()
            for program in raw_data_for_lists.get("study_programs", []):
                for major in program.get("available_majors", []):
                    all_majors.add(major.strip())
            majors_full = list(all_majors)

            col1, col2 = st.columns(2)

            with col1:
                with st.expander(
                    f"查看 {COUNTRIES[selected_country_for_preview]['display_name']} 所有院校 ({len(schools_full)} 所)",
                    expanded=False,
                ):
                    for i, school in enumerate(schools_full, 1):
                        st.write(f"**{i}.** {school}")

            with col2:
                with st.expander(
                    f"查看 {COUNTRIES[selected_country_for_preview]['display_name']} 所有专业 ({len(majors_full)} 个)",
                    expanded=False,
                ):
                    for i, major in enumerate(majors_full, 1):
                        st.write(f"**{i}.** {major}")
