from __future__ import annotations

import html
import re
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.report.pdf.config import pdf_config
from src.report.pdf.generators.pdf_styler import PDFStyler


def _para_text(text: str) -> str:
    if not text:
        return ""
    escaped = html.escape(str(text).strip())
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)


def _bullet_lines(items: list[Any], *, limit: int = 5) -> str:
    lines: list[str] = []
    for item in items[:limit]:
        s = str(item).strip()
        if s:
            lines.append(f"&bull; {_para_text(s)}")
    return "<br/>".join(lines)


def has_rich_ai_narrative(ai_explanation: dict | None) -> bool:
    if not ai_explanation or not isinstance(ai_explanation, dict):
        return False
    overview = str(ai_explanation.get("overview") or "").strip()
    summary = str(ai_explanation.get("summary") or "").strip()
    strengths = ai_explanation.get("strengths") or []
    concerns = ai_explanation.get("concerns") or []
    return bool(overview or summary or strengths or concerns)


def index_school_notes(ai_explanation: dict | None) -> dict[tuple[str, str], str]:
    if not ai_explanation or not isinstance(ai_explanation, dict):
        return {}
    out: dict[tuple[str, str], str] = {}
    for sn in ai_explanation.get("school_notes") or []:
        if not isinstance(sn, dict):
            continue
        uni = str(sn.get("university", "") or "").strip()
        major = str(sn.get("major", "") or "").strip()
        note = str(sn.get("note", "") or "").strip()
        if uni and major and note:
            out[(uni, major)] = note
    return out


def build_ai_narrative_flowables(
    styler: PDFStyler,
    ai_explanation: dict | None,
    page_width: float,
) -> list:
    if not has_rich_ai_narrative(ai_explanation):
        return []

    overview = str(ai_explanation.get("overview") or "").strip()
    summary = str(ai_explanation.get("summary") or "").strip()
    strengths = ai_explanation.get("strengths") or []
    concerns = ai_explanation.get("concerns") or []
    products = ai_explanation.get("products") or []

    body = ParagraphStyle(
        "_PfBody",
        parent=styler.styles["CustomBody"],
        alignment=TA_LEFT,
        leading=16,
        spaceAfter=4,
    )
    story: list = []

    if overview:
        story.append(Paragraph(_para_text(overview), body))
        story.append(Spacer(1, 0.1 * inch))

    if strengths or concerns:
        table = _strengths_concerns_table(styler, strengths, concerns, page_width)
        if table:
            story.append(table)

    if summary:
        story.append(Spacer(1, 0.1 * inch))
        story.append(
            Paragraph(
                f"<b>顾问结论</b><br/>{_para_text(summary)}",
                body,
            )
        )

    prod_lines = []
    for p in products[:3]:
        if not isinstance(p, dict):
            continue
        name = _para_text(str(p.get("name", "")))
        reason = _para_text(str(p.get("reason", "")))
        if name:
            prod_lines.append(f"&bull; <b>{name}</b>" + (f" — {reason}" if reason else ""))
    if prod_lines:
        story.append(Spacer(1, 0.08 * inch))
        story.append(Paragraph("<b>推荐服务</b>", styler.styles["SubTitle"]))
        story.append(Paragraph("<br/>".join(prod_lines), body))

    return story


def _strengths_concerns_table(
    styler: PDFStyler,
    strengths: list,
    concerns: list,
    page_width: float,
) -> Table:
    cell = ParagraphStyle(
        "_PfSC",
        parent=styler.styles["SmallText"],
        alignment=TA_LEFT,
        leading=14,
        spaceAfter=0,
    )
    half = (page_width - 0.12 * inch) / 2
    left = "<b>核心优势</b><br/>" + (_bullet_lines(strengths, limit=4) or "—") if strengths else ""
    right = "<b>需关注点</b><br/>" + (_bullet_lines(concerns, limit=4) or "—") if concerns else ""
    if not left and not right:
        return None

    if not left:
        rows = [[Paragraph(right, cell)]]
        col_widths = [page_width]
    elif not right:
        rows = [[Paragraph(left, cell)]]
        col_widths = [page_width]
    else:
        rows = [[Paragraph(left, cell), Paragraph(right, cell)]]
        col_widths = [half, half]

    table = Table(rows, colWidths=col_widths)
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(pdf_config.panel_background)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return table
