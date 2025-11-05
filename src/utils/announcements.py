import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

ANNOUNCEMENTS_CONFIG_PATH = "announcements_config.json"


@st.cache_data
def load_announcements_config() -> dict[str, Any]:
    if os.path.exists(ANNOUNCEMENTS_CONFIG_PATH):
        with open(ANNOUNCEMENTS_CONFIG_PATH, encoding="utf-8") as f:
            config = json.load(f)
            if isinstance(config, dict) and "announcements" in config:
                return config
            return {"announcements": []}
    return {"announcements": []}


def save_announcements_config(config: dict[str, Any]) -> tuple[bool, str]:
    """
    保存公告配置到文件

    Args:
        config: 公告配置字典

    Returns:
        tuple[bool, str]: (是否成功, 消息)
    """
    try:
        with open(ANNOUNCEMENTS_CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        load_announcements_config.clear()
        return True, "公告配置保存成功"
    except Exception as e:
        return False, f"保存公告配置失败: {str(e)}"


def get_user_group(
    user_email: str, is_admin_user: bool, accessible_modules: dict[str, bool]
) -> list[str]:
    groups = ["all"]
    if is_admin_user:
        groups.append("admin")

    if accessible_modules.get("hk", False):
        groups.append("hk")

    return groups


def filter_announcements_for_user(
    user_email: str, is_admin_user: bool, accessible_modules: dict[str, bool]
) -> list[dict[str, Any]]:
    config = load_announcements_config()
    all_announcements = config.get("announcements", [])

    user_groups = get_user_group(user_email, is_admin_user, accessible_modules)
    filtered_announcements = []

    current_date = datetime.now().strftime("%Y-%m-%d")

    for announcement in all_announcements:
        if not announcement.get("enabled", True):
            continue

        start_date = announcement.get("start_date")
        end_date = announcement.get("end_date")

        if start_date and current_date < start_date:
            continue
        if end_date and current_date > end_date:
            continue

        target_groups = announcement.get("target_groups", ["all"])
        if any(group in user_groups for group in target_groups):
            filtered_announcements.append(announcement)

    return filtered_announcements


def generate_announcement_html(announcements: list[dict[str, Any]]) -> str:
    if not announcements:
        return ""

    announcement_items = []
    for announcement in announcements:
        title = announcement.get("title", "")
        content = announcement.get("content", "")

        if title and content:
            announcement_items.append(f"{title}: {content}")
        elif content:
            announcement_items.append(content)

    if not announcement_items:
        return ""

    separator = " &nbsp; &nbsp; "
    combined_text = separator.join(announcement_items)

    scrolling_text = f"{combined_text}{separator}" * 2

    html = f"""
    <div class="announcement-container">
        <div class="announcement-scroll">
            <div class="announcement-text">{scrolling_text}</div>
        </div>
    </div>
    """

    return html


def generate_announcement_css() -> str:
    css_file_path = "assets/announcements_style.css"
    try:
        with open(css_file_path, encoding="utf-8") as f:
            css = f.read()
        return f"<style>{css}</style>"
    except FileNotFoundError:
        return ""
