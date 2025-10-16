import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    KEY_CURRENT_STEP,
    KEY_EDUCATION_DATA,
    KEY_SCORE_KEY,
    KEY_SCORE_RANGE,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_LANGUAGE,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.get_countries import (
    get_standardized_countries,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.steps import reset_selections_from_step


def render_step_3():
    COUNTRIES = get_standardized_countries()
    education_data = st.session_state[KEY_EDUCATION_DATA]

    country_data = COUNTRIES[st.session_state[KEY_SELECTED_COUNTRY]]
    raw_data = education_data.get(st.session_state[KEY_SELECTED_COUNTRY], {})
    selected_language = st.session_state.get(KEY_SELECTED_LANGUAGE)

    score_options = [
        ("优秀 (600+)", "高分段，可申请顶尖院校，有更多奖学金机会", "high"),
        ("良好 (500-600)", "中高分段，可申请优质院校，选择范围广泛", "mid"),
        ("一般 (400-500)", "中等分段，有多种选择，可通过预科提升", "low"),
        (
            "较低 (<400)",
            "可通过预科、语言班等多种途径入学，同样有机会进入好学校",
            "very_low",
        ),
    ]

    for i, (score_range, description, score_key) in enumerate(score_options):
        detailed_help = f"选择{score_range}：{description}。该分数段学生通常可以申请相应层次的院校，具体录取要求因学校而异。"
        if st.button(
            f"**{score_range}**\n\n{description}",
            key=f"score_{i}_{score_range}",
            type="primary",
            help=detailed_help,
        ):
            st.session_state[KEY_SCORE_RANGE] = score_range
            st.session_state[KEY_SCORE_KEY] = score_key
            st.session_state[KEY_CURRENT_STEP] = 4
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander(f"{country_data['display_name']} 入学要求详情", expanded=True):
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
            ">📊 分数要求</h4>
            """,
                unsafe_allow_html=True,
            )
            st.success("该国家入学门槛相对灵活，具体要求视院校而定")

            if raw_data.get("requirements"):
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
                ">✅ 入学条件</h4>
                """,
                    unsafe_allow_html=True,
                )
                for i, req in enumerate(raw_data["requirements"]):
                    if "分数" in req or "高考" in req:
                        st.success(f"**条件 {i + 1}:** {req}")

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
            ">💡 申请建议</h4>
            """,
                unsafe_allow_html=True,
            )
            if raw_data.get("descriptions"):
                score_related = [
                    desc
                    for desc in raw_data["descriptions"]
                    if "分数" in desc or "成绩" in desc or "门槛" in desc or "入学" in desc
                ]
                if score_related:
                    for desc in score_related:
                        st.success(f"{desc}")
                else:
                    general_advantage = (
                        raw_data["descriptions"][0] if raw_data["descriptions"] else ""
                    )
                    if general_advantage:
                        st.success(f"{general_advantage}")

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
            ">🛤️ 申请路径</h4>
            """,
                unsafe_allow_html=True,
            )
            application_paths = raw_data.get("application_paths", [])
            if application_paths:
                for path in application_paths:
                    st.success(f"• {path}")
            else:
                st.success("• 直接申请本科")
                st.success("• 预科 + 本科")
                st.success("• 语言班 + 本科")

    if raw_data.get("descriptions"):
        score_related = [
            desc
            for desc in raw_data["descriptions"]
            if "分数" in desc or "成绩" in desc or "门槛" in desc or "入学" in desc
        ]
        if score_related and len(score_related) > 1:
            with st.expander("查看更多申请建议", expanded=False):
                for i, desc in enumerate(score_related[1:], 2):
                    st.success(f"**建议 {i}:** {desc}")

    if st.button("上一步", type="primary", key="change_language_step3"):
        reset_selections_from_step(2)
        st.session_state[KEY_CURRENT_STEP] = 2
        st.rerun()
