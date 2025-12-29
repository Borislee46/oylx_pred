from __future__ import annotations

from pathlib import Path


def load_component_assets(assets_dir: Path) -> tuple[str, str, str]:
    style_css = ""
    script_js = ""
    template_html = ""

    css_path = assets_dir / "style.css"
    if css_path.exists():
        style_css = css_path.read_text(encoding="utf-8")

    js_path = assets_dir / "script.js"
    if js_path.exists():
        script_js = js_path.read_text(encoding="utf-8")

    html_path = assets_dir / "template.html"
    if html_path.exists():
        template_html = html_path.read_text(encoding="utf-8")
    elif (assets_dir / "index.html").exists():
        template_html = (assets_dir / "index.html").read_text(encoding="utf-8")

    return style_css, script_js, template_html
