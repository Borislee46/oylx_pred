import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_CURRENT_STEP,
    KEY_EDUCATION_DATA,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_LANGUAGE,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.get_countries import (
    get_standardized_countries,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.steps import reset_selections_from_step


def render_step_2():
    COUNTRIES = get_standardized_countries()
    education_data = st.session_state[KEY_EDUCATION_DATA]

    country_data = COUNTRIES[st.session_state[KEY_SELECTED_COUNTRY]]
    raw_data = education_data.get(st.session_state[KEY_SELECTED_COUNTRY], {})

    for i, language in enumerate(country_data["languages"]):
        language_info = f"选择{language}授课，可申请该国使用{language}教学的院校和专业"
        if st.button(
            f"**{language}**",
            key=f"lang_{i}_{language}",
            type="primary",
            help=language_info,
        ):
            st.session_state[KEY_SELECTED_LANGUAGE] = language
            st.session_state[KEY_CURRENT_STEP] = 3
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(f"{country_data['display_name']} 语言环境", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
            <h4 style="
                font-family: 'Playfair Display', 'Crimson Text', 'Georgia', serif;
                color: #2c3e50; 
                font-size: 1.3em; 
                font-weight: 600;
                margin-bottom: 15px;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 8px;
            ">📚 可选语言</h4>
            """,
                unsafe_allow_html=True,
            )
            for lang in raw_data.get("languages", []):
                st.success(f"• {lang}")

            if raw_data.get("descriptions"):
                language_related = [
                    desc
                    for desc in raw_data["descriptions"]
                    if "语言" in desc or "英语" in desc or "授课" in desc
                ]
                if language_related:
                    for desc in language_related:
                        st.success(desc)
                elif raw_data["descriptions"]:
                    st.success(raw_data["descriptions"][0])

        with col2:
            st.markdown(
                """
            <h4 style="
                font-family: 'Playfair Display', 'Crimson Text', 'Georgia', serif;
                color: #2c3e50; 
                font-size: 1.3em; 
                font-weight: 600;
                margin-bottom: 15px;
                letter-spacing: 0.5px;
                border-bottom: 2px solid #e9ecef;
                padding-bottom: 8px;
            ">🏛️ 国家优势概览</h4>
            """,
                unsafe_allow_html=True,
            )
            if raw_data.get("descriptions"):
                for i, desc in enumerate(raw_data.get("descriptions", [])[:3]):
                    st.success(f"**优势 {i + 1}:** {desc}")
            else:
                st.success("该国暂无详细优势描述。")

    if st.button("上一步", type="primary", key="change_country_step2"):
        reset_selections_from_step(1)
        st.session_state[KEY_CURRENT_STEP] = 1
        st.rerun()
