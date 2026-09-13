import logging
import os
import uuid
from urllib.parse import urlparse

import streamlit as st

from config.settings import load_app_config_raw

_log = logging.getLogger(__name__)

# RFC 2606 保留域名：出现即说明配置没改，直接退回同源相对路径，
# 否则会把用户跳到 example.com 这种不存在的域名上。
_PLACEHOLDER_HOSTS = ("example.com", "example.org", "example.net")


def _get_base_url() -> str:
    """按钮跳转用的基地址。

    返回以 `/` 结尾的绝对地址或裸 `/`（同源相对路径）。
    """
    raw = str(load_app_config_raw().get("STREAMLIT_APP_BASE_URL", "") or "").strip()
    if not raw:
        return "/"

    parsed = urlparse(raw)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        _log.warning("STREAMLIT_APP_BASE_URL 不是合法绝对地址（%r），改用同源相对路径", raw)
        return "/"

    host = parsed.hostname.lower()
    if any(host == h or host.endswith("." + h) for h in _PLACEHOLDER_HOSTS):
        _log.warning(
            "STREAMLIT_APP_BASE_URL 仍是占位域名（%r），改用同源相对路径。"
            "请在 config/app_config.json 或环境变量 STREAMLIT_APP_BASE_URL 里填真实域名。",
            raw,
        )
        return "/"

    return raw.rstrip("/") + "/"


def _generate_card_html(available_buttons: list, base_url: str, trace_id: str) -> str:
    from urllib.parse import quote

    user = st.session_state.get("demo_user_nickname", "unknown")
    email = st.session_state.get("demo_user_email", "")

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
            page_name = os.path.splitext(os.path.basename(path_or_url))[0]
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
