import json
import os
import uuid

import streamlit as st


def _get_base_url() -> str:
    dev_config_path = "config/dev_config.json"
    if os.path.exists(dev_config_path):
        with open(dev_config_path, encoding="utf-8") as f:
            dev_config = json.load(f)
        if dev_config.get("DEBUG_MODE", False):
            return "http://localhost:80/"

    with open("config/app_config.json", encoding="utf-8") as f:
        all_configs = json.load(f)
    env = os.environ.get("APP_ENV", "test")
    return all_configs[env]["STREAMLIT_APP_BASE_URL"]


def _generate_card_html(available_buttons: list, base_url: str, trace_id: str) -> str:
    from urllib.parse import quote

    user = st.session_state.get("e2_user_nickname", "unknown")
    email = st.session_state.get("e2_user_email", "")

    cards_html = ""
    trace_param = quote(trace_id or "", safe="")

    for idx, (button_text, path_or_url, is_link) in enumerate(available_buttons):
        if is_link:
            redirect_url = (
                f"{base_url}redirect?target={quote(path_or_url, safe='')}"
                f"&source={quote(button_text, safe='')}"
                f"&user={quote(user, safe='')}"
                f"&email={quote(email, safe='')}"
                f"&trace={trace_param}"
            )
            full_url = redirect_url
        else:
            page_name = path_or_url.replace("pages/", "").replace(".py", "")
            full_url = base_url + page_name
            separator = "&" if "?" in full_url else "?"
            full_url = f"{full_url}{separator}trace={trace_param}"
        data_attrs = (
            f'data-index="{idx}" data-url="{full_url}" data-is-link="{str(is_link).lower()}"'
        )
        cards_html += (
            f'<div class="card-wrapper" {data_attrs}>'
            f'<div class="card">'
            f'<div class="glare"></div>'
            f'<span class="card-text">{button_text}</span>'
            f"</div>"
            f"</div>"
        )
    return cards_html


def _generate_component_html(available_buttons: list, base_url: str, trace_id: str) -> str:
    from pathlib import Path

    from src.utils.ui.ui_utils import load_component_assets

    num_cards = len(available_buttons)
    cards_html = _generate_card_html(available_buttons, base_url, trace_id)

    assets_dir = Path("assets/ui/main_page_button")
    style_css, script_js, template_html = load_component_assets(assets_dir)

    card_width = 160 if num_cards > 4 else 180
    card_height = 220 if num_cards > 4 else 240
    linear_gap = 32 if num_cards > 4 else 40

    full_html = f"""
    <style>
        :root {{
            --card-width: {card_width}px;
            --card-height: {card_height}px;
            --linear-gap: {linear_gap}px;
        }}
        {style_css}
    </style>
    {template_html.replace("{{cards_html}}", cards_html)}
    <script>
        {script_js.replace("{{num_cards}}", str(num_cards))}
    </script>
    """
    return full_html


def render_buttons_grid(available_buttons: list) -> None:
    if not available_buttons:
        return

    base_url = _get_base_url()
    if "session_trace_uuid" not in st.session_state:
        st.session_state.session_trace_uuid = uuid.uuid4().hex
    trace_id = st.session_state.session_trace_uuid

    component_html = _generate_component_html(available_buttons, base_url, trace_id)

    num_cards = len(available_buttons)
    if num_cards <= 4:
        height = 450
    elif num_cards <= 6:
        height = 480
    else:
        height = 480 + (num_cards - 6) * 12

    st.iframe(component_html, height=height)
