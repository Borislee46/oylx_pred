import html
import os
from pathlib import Path

import streamlit as st

from src.utils.ui.ui_utils import load_component_assets

LOGO_PATH = "assets/company_logo.png"

_PARALLAX_JS = """
<script>
(function () {
  if (window._parallaxInited) return;
  window._parallaxInited = true;
  const root = window.parent.document.documentElement;
  let ticking = false, mouseX = 0, mouseY = 0;
  const update = () => {
    root.style.setProperty("--bg-pos-x", ((mouseX / window.parent.innerWidth - 0.5) * 25) + "px");
    root.style.setProperty("--bg-pos-y", ((mouseY / window.parent.innerHeight - 0.5) * 25) + "px");
    ticking = false;
  };
  window.parent.addEventListener("mousemove", function (e) {
    mouseX = e.clientX; mouseY = e.clientY;
    if (!ticking) { window.parent.requestAnimationFrame(update); ticking = true; }
  }, { passive: true });
  setTimeout(function () {
    const anchor = window.parent.document.getElementById("main-page-header-anchor");
    if (anchor) anchor.scrollIntoView({ behavior: "smooth", block: "start" });
  }, 50);
})();
</script>
"""


def render_header(
    user_nickname: str,
    *,
    page_title: str | None = None,
    page_subtitle: str | None = None,
) -> None:
    assets_dir = Path("assets/ui/main_page_header")
    style_css, _, template_html = load_component_assets(assets_dir)

    st.html('<div id="main-page-header-anchor"></div>')

    st.html(f"<style>{style_css}</style>")

    if os.path.exists(LOGO_PATH) and os.path.getsize(LOGO_PATH) > 0:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(LOGO_PATH)

    title = page_title or "欧亚数据科学平台"
    subtitle = page_subtitle or "AI驱动的智能决策与个性化数据服务"
    header_html = template_html.replace("{{PAGE_TITLE}}", html.escape(title)).replace(
        "{{PAGE_SUBTITLE}}", html.escape(subtitle)
    )
    st.html(header_html)

    st.iframe(_PARALLAX_JS, height=1)
