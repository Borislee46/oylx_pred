from functools import partial

import streamlit as st


def _log_experience_change(session_manager, form_state_manager, experience_type):
    value = session_manager.get_widget_value(f"{experience_type}_count_input", 0)
    form_state_manager.on_form_change(session_manager, change_type="text")


def _log_experience_details_change(session_manager, form_state_manager, experience_type):
    _ = session_manager.get_widget_value(f"{experience_type}_details_input", "")
    form_state_manager.on_form_change(session_manager, change_type="text")


def render_experience_section(session_manager, form_state_manager, logger):
    st.markdown("**其他经历**")

    user_history_data = session_manager.get("user_history_data", {})

    def render_experience_item(label_prefix, count_key, details_key, placeholder_text):
        col_count, col_details = st.columns([1, 5], gap="small")
        with col_count:
            count_val = st.number_input(
                f"{label_prefix}数量",
                min_value=0,
                max_value=99,
                value=user_history_data.get(count_key, 0),
                on_change=partial(
                    _log_experience_change,
                    session_manager,
                    form_state_manager,
                    count_key.split("_")[0],
                ),
                placeholder="",
                key=f"{count_key}_input",
            )
        with col_details:
            details_val = st.text_input(
                f"{label_prefix}详细信息（选填）",
                value=user_history_data.get(details_key, ""),
                on_change=partial(
                    _log_experience_details_change,
                    session_manager,
                    form_state_manager,
                    count_key.split("_")[0],
                ),
                placeholder=placeholder_text,
                key=f"{details_key}_input",
            )
        return count_val, details_val

    research_count, research_details = render_experience_item(
        "科研",
        "research_count",
        "research_details",
        "例如：参与国家重点实验室项目，发表SCI论文等",
    )
    award_count, award_details = render_experience_item(
        "获奖", "award_count", "award_details", "例如：国家奖学金、省级一等奖、国际竞赛金奖等"
    )
    internship_count, internship_details = render_experience_item(
        "实习",
        "internship_count",
        "internship_details",
        "例如：腾讯、阿里巴巴、500强企业实习经历等",
    )
    paper_count, paper_details = render_experience_item(
        "论文", "paper_count", "paper_details", "例如：第一作者SCI论文、核心期刊发表等"
    )

    experience_details = {
        "research_details": session_manager.get_widget_value("research_details_input", ""),
        "award_details": session_manager.get_widget_value("award_details_input", ""),
        "internship_details": session_manager.get_widget_value("internship_details_input", ""),
        "paper_details": session_manager.get_widget_value("paper_details_input", ""),
    }

    return research_count, award_count, internship_count, paper_count, experience_details
