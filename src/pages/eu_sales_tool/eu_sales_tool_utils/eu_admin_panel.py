import json

import streamlit as st

from src.pages.eu_sales_tool.eu_sales_tool_utils.load_config import load_config_data
from src.pages.eu_sales_tool.eu_sales_tool_utils.load_data import load_education_data
from src.pages.eu_sales_tool.eu_sales_tool_utils.modify_permission_check import (
    check_admin_permission,
)
from src.pages.eu_sales_tool.eu_sales_tool_utils.save_config import (
    save_config_data,
    save_education_data,
)


def show_admin_panel():
    if not check_admin_permission():
        return

    with st.expander(
        "业务老师编辑面板（不能编辑的部分请联系 李佳鹏 lijiapeng8@xdf.cn 进行添加）",
        expanded=False,
    ):
        config = st.session_state.get("site_config", load_config_data())
        education_data = st.session_state.get("education_data", load_education_data())

        tab1, tab3, tab4 = st.tabs(["标题文案编辑", "国家描述编辑", "国家详情编辑"])

        with tab1:
            st.subheader("页面文案编辑")

            new_title = st.text_input("页面标题", value=config.get("page_title", ""))
            new_welcome = st.text_area("欢迎文字", value=config.get("welcome_text", ""), height=100)

            if st.button("保存文案修改"):
                config["page_title"] = new_title
                config["welcome_text"] = new_welcome

                if save_config_data(config):
                    st.session_state.site_config = config
                    st.success("文案修改已保存！")
                    st.rerun()
                else:
                    st.error("保存失败，请重试")

        with tab3:
            st.subheader("国家选择卡片描述编辑")

            country_descriptions = config.get("country_descriptions", {})

            for country_key in education_data.keys():
                new_desc = st.text_area(
                    f"{country_key} 卡片描述",
                    value=country_descriptions.get(country_key, ""),
                    key=f"desc_{country_key}",
                    height=80,
                )
                country_descriptions[country_key] = new_desc

            if st.button("保存国家描述"):
                config["country_descriptions"] = country_descriptions

                if save_config_data(config):
                    st.session_state.site_config = config
                    st.success("国家描述已保存！")
                    st.rerun()
                else:
                    st.error("保存失败，请重试")

        with tab4:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.info("编辑老师无需关心这个下载功能")
            with col2:
                education_json = json.dumps(education_data, ensure_ascii=False, indent=2)
                st.download_button(
                    label="下载教育数据JSON文件",
                    data=education_json,
                    file_name="education_data.json",
                    mime="application/json",
                    help="下载当前教育数据文件",
                )

            countries = list(education_data.keys())
            if countries:
                selected_country_for_edit = st.selectbox(
                    "选择要编辑的国家/地区", countries, key="admin_select_country"
                )

                if selected_country_for_edit:
                    country_data = education_data[selected_country_for_edit]

                    st.markdown(f"### 编辑 {selected_country_for_edit} 的留学项目")

                    if "study_programs" not in country_data:
                        country_data["study_programs"] = []

                    for i, program in enumerate(country_data["study_programs"]):
                        with st.container(border=True):
                            st.markdown(
                                f"**项目 {i + 1}: {program.get('program_name', '未命名项目')}**"
                            )
                            st.text_input(
                                "项目名称",
                                value=program.get("program_name", ""),
                                key=f"pname_{selected_country_for_edit}_{i}",
                            )
                            st.text_area(
                                "项目描述",
                                value=program.get("description", ""),
                                key=f"pdesc_{selected_country_for_edit}_{i}",
                                height=100,
                            )

                            score_options = ["high", "mid", "low", "very_low"]
                            st.multiselect(
                                "适用分数段",
                                options=score_options,
                                default=program.get("required_scores", []),
                                key=f"pscore_{selected_country_for_edit}_{i}",
                            )

                            st.text_area(
                                "可申请院校（每行一所）",
                                value="\n".join(program.get("available_schools", [])),
                                key=f"pschools_{selected_country_for_edit}_{i}",
                                height=150,
                            )
                            st.text_area(
                                "可申请专业（每行一个）",
                                value="\n".join(program.get("available_majors", [])),
                                key=f"pmajors_{selected_country_for_edit}_{i}",
                                height=100,
                            )

                            if st.button(
                                "删除此项目",
                                key=f"pdel_{selected_country_for_edit}_{i}",
                                type="secondary",
                            ):
                                country_data["study_programs"].pop(i)
                                st.rerun()

                    if st.button("新增留学项目", key=f"add_program_{selected_country_for_edit}"):
                        country_data["study_programs"].append(
                            {
                                "program_name": "新项目",
                                "description": "",
                                "required_scores": [],
                                "available_schools": [],
                                "available_majors": [],
                            }
                        )
                        st.rerun()

                    st.markdown("#### 国家通用信息编辑")

                    new_descriptions = st.text_area(
                        "国家优势描述（每行一条）",
                        value="\n".join(country_data.get("descriptions", [])),
                        height=120,
                        key=f"descriptions_{selected_country_for_edit}",
                    )

                    new_application_paths = st.text_area(
                        "国家通用申请路径（每行一条）",
                        value="\n".join(
                            country_data.get(
                                "application_paths",
                                ["直接申请本科", "预科 + 本科", "语言班 + 本科"],
                            )
                        ),
                        height=100,
                        key=f"paths_{selected_country_for_edit}",
                    )

                    st.text_area(
                        "入学要求（每行一条）",
                        value="\n".join(country_data.get("requirements", [])),
                        height=120,
                        key=f"requirements_{selected_country_for_edit}",
                    )

                    st.markdown("#### 可用语言")
                    st.text_area(
                        "授课语言列表（每行一种）",
                        value="\n".join(country_data.get("languages", [])),
                        height=80,
                        key=f"languages_{selected_country_for_edit}",
                    )

                    st.markdown("#### 留学费用估算")
                    cost_data = country_data.get("cost_estimation", {})
                    new_tuition = st.text_input(
                        "参考学费",
                        value=cost_data.get("tuition", "公立大学免费 / 私立大学€10,000+"),
                        key=f"cost_tuition_{selected_country_for_edit}",
                    )
                    new_living_cost = st.text_input(
                        "预估生活费",
                        value=cost_data.get("living_cost", "每年 €10,000 - €15,000"),
                        key=f"cost_living_{selected_country_for_edit}",
                    )
                    new_cost_desc = st.text_area(
                        "费用描述",
                        value=cost_data.get("description", "具体费用因城市和生活方式而异。"),
                        key=f"cost_desc_{selected_country_for_edit}",
                        height=100,
                    )

                    st.info("请注意：点击下方按钮将保存对该国家的所有修改（除了留学项目修改）。")
                    button_label = f'保存对 "{selected_country_for_edit}" 的所有修改'
                    if st.button(button_label, type="primary"):
                        updated_programs = []
                        for i, _ in enumerate(country_data["study_programs"]):
                            program_name = st.session_state[
                                f"pname_{selected_country_for_edit}_{i}"
                            ].strip()
                            if not program_name:
                                st.error(f"错误：项目 {i + 1} 的项目名称不能为空。请填写后再保存。")
                                st.stop()

                            updated_programs.append(
                                {
                                    "program_name": program_name,
                                    "description": st.session_state[
                                        f"pdesc_{selected_country_for_edit}_{i}"
                                    ],
                                    "required_scores": st.session_state[
                                        f"pscore_{selected_country_for_edit}_{i}"
                                    ],
                                    "available_schools": [
                                        s.strip()
                                        for s in st.session_state[
                                            f"pschools_{selected_country_for_edit}_{i}"
                                        ].split("\n")
                                        if s.strip()
                                    ],
                                    "available_majors": [
                                        m.strip()
                                        for m in st.session_state[
                                            f"pmajors_{selected_country_for_edit}_{i}"
                                        ].split("\n")
                                        if m.strip()
                                    ],
                                }
                            )

                        country_data["study_programs"] = updated_programs

                        country_data["descriptions"] = [
                            d.strip()
                            for d in st.session_state[
                                f"descriptions_{selected_country_for_edit}"
                            ].split("\n")
                            if d.strip()
                        ]
                        country_data["application_paths"] = [
                            p.strip()
                            for p in st.session_state[f"paths_{selected_country_for_edit}"].split(
                                "\n"
                            )
                            if p.strip()
                        ]
                        country_data["requirements"] = [
                            r.strip()
                            for r in st.session_state[
                                f"requirements_{selected_country_for_edit}"
                            ].split("\n")
                            if r.strip()
                        ]

                        current_languages = [
                            l.strip()
                            for l in st.session_state[
                                f"languages_{selected_country_for_edit}"
                            ].split("\n")
                            if l.strip()
                        ]
                        country_data["languages"] = current_languages

                        country_data["cost_estimation"] = {
                            "tuition": st.session_state[
                                f"cost_tuition_{selected_country_for_edit}"
                            ],
                            "living_cost": st.session_state[
                                f"cost_living_{selected_country_for_edit}"
                            ],
                            "description": st.session_state[
                                f"cost_desc_{selected_country_for_edit}"
                            ],
                        }

                        education_data[selected_country_for_edit] = country_data

                        if save_education_data(education_data):
                            st.session_state.education_data = education_data
                            st.cache_data.clear()
                            st.success(f"{selected_country_for_edit} 的数据已保存！")
                            st.rerun()
                        else:
                            st.error("保存失败，请重试")

            uploaded_education_file = st.file_uploader(
                "选择教育数据文件",
                type=["json"],
                help="仅支持JSON格式的教育数据文件",
                key="education_upload",
            )

            if uploaded_education_file is not None:
                try:
                    uploaded_education_data = json.load(uploaded_education_file)

                    if isinstance(uploaded_education_data, dict):
                        st.success("教育数据文件格式验证通过")

                        with st.expander("预览教育数据内容"):
                            for country, data in uploaded_education_data.items():
                                all_schools = set()
                                all_majors = set()
                                for program in data.get("study_programs", []):
                                    for school in program.get("available_schools", []):
                                        all_schools.add(school)
                                    for major in program.get("available_majors", []):
                                        all_majors.add(major)
                                st.write(
                                    f"**{country}**: {len(all_schools)} 所院校, {len(all_majors)} 个专业方向"
                                )

                        if st.button("应用教育数据", type="primary", key="apply_education"):
                            if save_education_data(uploaded_education_data):
                                st.session_state.education_data = uploaded_education_data
                                st.cache_data.clear()
                                st.success("教育数据已成功应用！")
                                st.rerun()
                            else:
                                st.error("应用教育数据失败，请重试")
                    else:
                        st.error("教育数据文件格式不正确")
                except json.JSONDecodeError:
                    st.error("文件不是有效的JSON格式")
                except Exception as e:
                    st.error(f"读取文件失败: {e}")
