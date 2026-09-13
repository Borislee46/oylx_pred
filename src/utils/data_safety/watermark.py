import datetime
import html
import os

_cached_css: str = ""
_cached_mtime: float = 0.0


def read_css_file(css_path: str) -> str:
    global _cached_css, _cached_mtime
    try:
        mtime = os.path.getmtime(css_path)
    except OSError:
        return _cached_css or ""
    if mtime == _cached_mtime and _cached_css:
        return _cached_css
    with open(css_path, encoding="utf-8") as f:
        _cached_css = f.read()
    _cached_mtime = mtime
    return _cached_css


def generate_watermark_css(
    user_nickname: str | None = None,
    opacity: float = 0.1,  # 0.1
    color: str = "#000000",
    font_size: str = "14px",
    x_spacing: int = 350,
    y_spacing: int = 150,
) -> str:
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")

    if not user_nickname:
        user_nickname = "访客"

    watermark_text = html.escape(f"{current_date} {user_nickname}")

    css_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "assets", "watermark.css")
    base_css = read_css_file(css_path)

    dynamic_css = f"""
    <style>
    {base_css}

    .watermark-text-layer {{
        opacity: {opacity};
    }}

    .watermark-text-pattern {{
        color: {color};
        font-size: {font_size};
    }}
    </style>"""

    watermark_divs = []
    for row in range(-5, 18):
        for col in range(-5, 20):
            x_pos = col * x_spacing + (row % 2) * (x_spacing // 2)
            y_pos = row * y_spacing
            div_html = f'<div class="watermark-text-pattern" style="left:{x_pos}px;top:{y_pos}px;">{watermark_text}</div>'
            watermark_divs.append(div_html)

    watermark_html = f'<div class="watermark-text-layer">{"".join(watermark_divs)}</div>'

    return dynamic_css + watermark_html
