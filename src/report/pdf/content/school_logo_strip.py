from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.report.pdf.config import pdf_config
from src.report.pdf.tier_labels import tier_display_label
from src.report.pdf.utils import create_responsive_image
from src.utils.numeric import clip_probability_coerce

_MAX_LOGOS = 6
_LOGO_SIZE = 0.52 * inch


def _dedupe_by_university(unified_results: list[dict]) -> list[dict]:
    best: dict[str, dict] = {}
    for r in unified_results:
        if not isinstance(r, dict):
            continue
        uni = str(r.get("university", "") or "").strip()
        if not uni:
            continue
        prob = clip_probability_coerce(r.get("probability"))
        prev = best.get(uni)
        if prev is None or prob > clip_probability_coerce(prev.get("probability")):
            best[uni] = r
    return sorted(
        best.values(),
        key=lambda x: clip_probability_coerce(x.get("probability")),
        reverse=True,
    )


def _logo_cell(school: dict, *, styles, chinese_font_name: str):
    uni = str(school.get("university", "") or "")
    logo_path_obj = pdf_config.get_school_logo_path(uni)
    if logo_path_obj:
        img = create_responsive_image(str(logo_path_obj), _LOGO_SIZE, _LOGO_SIZE)
        img.hAlign = "CENTER"
        top = img
    else:
        top = Paragraph(
            f"<b>{uni[:1] or '?'}</b>",
            ParagraphStyle(
                "_LogoInitial",
                parent=styles["SmallText"],
                fontName=chinese_font_name,
                fontSize=14,
                alignment=TA_CENTER,
                textColor=colors.HexColor(pdf_config.theme_primary),
            ),
        )

    prob = clip_probability_coerce(school.get("probability"))
    tier = tier_display_label(school.get("label"))
    prob_color = pdf_config.get_probability_color(prob)
    badge = Paragraph(
        f"<font color='{prob_color}' size='8'><b>{prob:.0%}</b></font>"
        f"<br/><font color='{pdf_config.text_muted}' size='7'>{tier}</font>",
        ParagraphStyle(
            "_LogoProb",
            parent=styles["SmallText"],
            fontName=chinese_font_name,
            alignment=TA_CENTER,
            leading=10,
            spaceAfter=0,
        ),
    )
    col_w = _LOGO_SIZE + 0.14 * inch
    inner = Table([[top], [badge]], colWidths=[col_w])
    inner.setStyle(
        TableStyle(
            [
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return inner


def build_school_logo_strip(
    unified_results: list[dict],
    *,
    styles,
    chinese_font_name: str,
    max_logos: int = _MAX_LOGOS,
) -> list:
    all_schools = _dedupe_by_university(unified_results)
    schools = all_schools[:max_logos]
    if not schools:
        return []

    overflow = max(0, len(all_schools) - max_logos)
    cells = [_logo_cell(s, styles=styles, chinese_font_name=chinese_font_name) for s in schools]

    col_w = _LOGO_SIZE + 0.14 * inch
    row = Table([cells], colWidths=[col_w] * len(cells))
    row.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(pdf_config.panel_background)),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(pdf_config.border_color)),
            ]
        )
    )

    names = " · ".join(str(s.get("university", "")) for s in schools)
    if overflow:
        names += f" 等 {len(all_schools)} 所"
    caption = Paragraph(
        f"<font color='{pdf_config.text_muted}' size='8'>{names}</font>",
        ParagraphStyle(
            "_LogoCaption",
            parent=styles["SmallText"],
            alignment=TA_CENTER,
            spaceAfter=0,
        ),
    )
    return [row, Spacer(1, 0.06 * inch), caption, Spacer(1, 0.14 * inch)]
