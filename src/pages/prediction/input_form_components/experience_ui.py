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
    experience_details_data = user_history_data.get("experience_details", {})

    experience_items = [
        {
            "label": "科研",
            "type": "research",
            "placeholder": "例如：参与国家重点实验室项目，发表SCI论文等",
        },
        {
            "label": "获奖",
            "type": "award",
            "placeholder": "例如：国家奖学金、省级一等奖、国际竞赛金奖等",
        },
        {
            "label": "实习",
            "type": "internship",
            "placeholder": "例如：腾讯、阿里巴巴、500强企业实习经历等",
        },
        {
            "label": "论文",
            "type": "paper",
            "placeholder": "例如：第一作者SCI论文、核心期刊发表等",
        },
    ]

    results = {}
    details = {}

    for item in experience_items:
        label_prefix = item["label"]
        item_type = item["type"]
        placeholder = item["placeholder"]
        count_key = f"{item_type}_count"
        details_key = f"{item_type}_details"

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
                    item_type,
                ),
                placeholder="",
                key=f"{count_key}_input",
            )
        with col_details:
            details_val = st.text_input(
                f"{label_prefix}详细信息（选填）",
                value=experience_details_data.get(details_key, ""),
                on_change=partial(
                    _log_experience_details_change,
                    session_manager,
                    form_state_manager,
                    item_type,
                ),
                placeholder=placeholder,
                key=f"{details_key}_input",
            )
        results[count_key] = count_val
        details[details_key] = details_val

    return (
        results["research_count"],
        results["award_count"],
        results["internship_count"],
        results["paper_count"],
        details,
    )
