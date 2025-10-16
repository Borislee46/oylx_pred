import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.constants import (
    COUNTRY_TO_FLAG_NAME,
    KEY_CURRENT_STEP,
    KEY_EDUCATION_DATA,
    KEY_SCORE_RANGE,
    KEY_SELECTED_COUNTRY,
    KEY_SELECTED_LANGUAGE,
    KEY_SELECTED_PROGRAM,
    KEY_SELECTED_STUDY_PROGRAM,
    LANGUAGE_TO_FLAG_NAME,
    LANGUAGE_TO_SYMBOL,
    LANGUAGE_TO_TITLE,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.get_countries import (
    get_standardized_countries,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.steps import reset_selections_from_step
from src.pages.eu_sales_tool.eu_sales_tool_utils.utils import get_image_as_base64


def render_step_5():
    COUNTRIES = get_standardized_countries()
    education_data = st.session_state[KEY_EDUCATION_DATA]

    country_data = COUNTRIES[st.session_state[KEY_SELECTED_COUNTRY]]
    selected_study_program_name = st.session_state.get(KEY_SELECTED_STUDY_PROGRAM)
    raw_data = education_data.get(st.session_state.get(KEY_SELECTED_COUNTRY), {})

    study_program = next(
        (
            p
            for p in raw_data.get("study_programs", [])
            if p.get("program_name") == selected_study_program_name
        ),
        None,
    )

    if not study_program:
        st.success("无法找到所选留学项目的信息，请返回重选。")
        if st.button("返回上一步", type="primary"):
            reset_selections_from_step(4)
            st.session_state[KEY_CURRENT_STEP] = 4
            st.rerun()
        return

    if not st.session_state.get(KEY_SELECTED_PROGRAM):
        _render_major_selection(study_program)
    else:
        _render_final_plan(country_data, study_program, raw_data)

    st.markdown("</div>", unsafe_allow_html=True)


def _render_major_selection(study_program):
    majors = study_program.get("available_majors", [])
    if majors:
        for i, major in enumerate(majors):
            major_help = f"选择 {major} 专业，可申请该项目下所有开设此专业的院校"
            if st.button(
                f"**{major}**",
                key=f"major_{i}_{major}",
                type="primary",
                help=major_help,
            ):
                st.session_state[KEY_SELECTED_PROGRAM] = major
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.success("该留学项目暂未配置专业方向，请联系管理员在后台添加。")

    with st.expander(f"项目详情: {study_program.get('program_name')}", expanded=True):
        if study_program.get("description"):
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
            ">🎓 项目介绍</h4>
            """,
                unsafe_allow_html=True,
            )
            st.success(study_program.get("description"))

    col1, col2 = st.columns(2)
    with col1:
        schools = study_program.get("available_schools", [])
        with st.expander(f"查看所有 {len(schools)} 所可申请院校", expanded=True):
            for i, school in enumerate(schools, 1):
                st.write(f"**{i}.** {school}")

    with col2:
        majors = study_program.get("available_majors", [])
        with st.expander(f"查看所有 {len(majors)} 个专业方向", expanded=True):
            for i, major in enumerate(majors, 1):
                st.write(f"**{i}.** {major}")

    if st.button("上一步", type="primary", key="change_study_program"):
        reset_selections_from_step(4)
        st.session_state[KEY_CURRENT_STEP] = 4
        st.rerun()


def _render_final_plan(country_data, study_program, raw_data):
    country_key = st.session_state[KEY_SELECTED_COUNTRY]
    selected_language = st.session_state.get(KEY_SELECTED_LANGUAGE)

    background_style = _get_background_style(country_key, selected_language)

    title_text, language_symbol = _get_language_title(selected_language)

    _render_plan_header(background_style, title_text, language_symbol, country_data, study_program)

    _render_cost_information(raw_data)

    _render_schools_and_advantages(study_program, raw_data)

    _render_statistics(study_program, country_data, raw_data)

    if st.button("重新开始咨询", type="primary"):
        reset_selections_from_step(1)
        st.session_state[KEY_CURRENT_STEP] = 1
        st.rerun()


def _get_background_style(country_key, selected_language):
    flag_name = None
    if country_key == "欧英（爱荷北欧瑞匈比）":
        flag_name = LANGUAGE_TO_FLAG_NAME.get(selected_language, "Ireland")
    else:
        flag_name = COUNTRY_TO_FLAG_NAME.get(country_key)

    background_style = ""
    if flag_name:
        flag_path = f"assets/flags/{flag_name}.png"
        img_b64 = get_image_as_base64(flag_path)
        if img_b64:
            background_style = f"""
            background-image: linear-gradient(rgba(255, 255, 255, 0.3), rgba(255, 255, 255, 0.3)), url(data:image/png;base64,{img_b64});
            background-size: cover;
            background-position: center;
            padding: 20px;
            border-radius: 10px;
            color: #333;
            """
    return background_style


def _get_language_title(selected_language):
    language_titles = LANGUAGE_TO_TITLE
    language_symbols = LANGUAGE_TO_SYMBOL

    title_text = language_titles.get(selected_language, "您的留学方案已生成")
    language_symbol = language_symbols.get(selected_language, "🎓")

    return title_text, language_symbol


def _render_plan_header(background_style, title_text, language_symbol, country_data, study_program):
    selected_study_program_name = st.session_state.get(KEY_SELECTED_STUDY_PROGRAM)

    st.markdown(
        f"""
    <div class="final-result" style="{background_style}">
        <h2 style="
            font-family: 'Playfair Display', 'Crimson Text', 'Georgia', 'Times New Roman', serif;
            font-size: 2.4em;
            font-weight: 700;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            background: linear-gradient(45deg, #1a252f, #2c3e50, #1a252f);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 25px;
            letter-spacing: 1.5px;
            font-style: italic;
        ">{language_symbol} {title_text}</h2>
        <div style="
            text-align: left; 
            margin: 20px 0;
            font-family: 'Playfair Display', 'Georgia', 'Times New Roman', serif;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.25) 50%, rgba(255,255,255,0.15) 100%);
            padding: 20px;
            border-radius: 12px;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.3);
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <p style="
                color: #2c3e50; 
                margin: 12px 0; 
                font-size: 1.15em; 
                font-weight: 500;
                letter-spacing: 0.5px;
                font-family: 'Crimson Text', 'Playfair Display', 'Georgia', serif;
            "><strong style="color: #1a252f; font-weight: 600;">Destination:</strong> {country_data["display_name"]}</p>
            <p style="
                color: #2c3e50; 
                margin: 12px 0; 
                font-size: 1.15em; 
                font-weight: 500;
                letter-spacing: 0.5px;
                font-family: 'Crimson Text', 'Playfair Display', 'Georgia', serif;
            "><strong style="color: #1a252f; font-weight: 600;">Language:</strong> {st.session_state[KEY_SELECTED_LANGUAGE]}</p>
            <p style="
                color: #2c3e50; 
                margin: 12px 0; 
                font-size: 1.15em; 
                font-weight: 500;
                letter-spacing: 0.5px;
                font-family: 'Crimson Text', 'Playfair Display', 'Georgia', serif;
            "><strong style="color: #1a252f; font-weight: 600;">Academic Level:</strong> {st.session_state[KEY_SCORE_RANGE]}</p>
            <p style="
                color: #2c3e50; 
                margin: 12px 0; 
                font-size: 1.15em; 
                font-weight: 500;
                letter-spacing: 0.5px;
                font-family: 'Crimson Text', 'Playfair Display', 'Georgia', serif;
            "><strong style="color: #1a252f; font-weight: 600;">Program:</strong> {selected_study_program_name}</p>
            <p style="
                color: #2c3e50; 
                margin: 12px 0; 
                font-size: 1.15em; 
                font-weight: 500;
                letter-spacing: 0.5px;
                font-family: 'Crimson Text', 'Playfair Display', 'Georgia', serif;
            "><strong style="color: #1a252f; font-weight: 600;">Specialization:</strong> {st.session_state[KEY_SELECTED_PROGRAM]}</p>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )


def _render_cost_information(raw_data):
    cost_data = raw_data.get("cost_estimation", {})
    if cost_data.get("tuition") or cost_data.get("living_cost"):
        with st.container(border=True):
            st.success(f"**参考学费：** {cost_data.get('tuition', 'N/A')}")
            st.success(f"**预估生活费：** {cost_data.get('living_cost', 'N/A')}")

            if cost_data.get("description"):
                with st.expander("费用详细说明", expanded=False):
                    st.success(cost_data["description"])


def _render_schools_and_advantages(study_program, raw_data):
    col1, col2 = st.columns(2)

    with col1:
        schools = study_program.get("available_schools", [])
        if schools:
            with st.expander(f"{len(schools)} 所推荐院校", expanded=False):
                for i, school in enumerate(schools, 1):
                    st.write(f"**{i}.** {school}")
        else:
            st.success("该项目暂未配置推荐院校。")

    with col2:
        if raw_data.get("descriptions"):
            with st.expander(f"{len(raw_data['descriptions'])}条优势", expanded=False):
                for i, desc in enumerate(raw_data["descriptions"], 1):
                    st.success(f"**优势 {i}:** {desc}")

        if study_program.get("description"):
            with st.expander("详细申请要求", expanded=False):
                st.success(study_program.get("description"))
        else:
            st.success("该项目暂无详细申请要求描述。")


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
