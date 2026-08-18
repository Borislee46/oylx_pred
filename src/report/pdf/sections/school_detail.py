import html
import re

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.pages.prediction.core import get_school_major_details
from src.report.pdf.config import pdf_config
from src.report.pdf.content.school_specific_content import (
    SchoolSpecificContentGenerator,
)
from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.tier_labels import tier_display_label
from src.report.pdf.utils import (
    SchoolDataProcessor,
    create_responsive_image,
)
from src.utils.logger import setup_logger
from src.utils.numeric import clip_probability_coerce

logger = setup_logger("page3", "prediction")

MAX_MAJORS_PER_CARD = 6


class SchoolDetailBuilder:
    def __init__(self, styler: PDFStyler, data_processor: SchoolDataProcessor):
        self.styles = styler.styles
        self.styler = styler
        self.data_processor = data_processor
        self.school_content_generator = SchoolSpecificContentGenerator()

    def create_grouped_school_detail(
        self,
        university: str,
        school_group: list[dict],
        rank: int,
        cases_df: pd.DataFrame,
        *,
        school_notes_index: dict[tuple[str, str], str] | None = None,
    ) -> list:
        if not university or not isinstance(university, str):
            university = "未知大学"
        if not school_group or not isinstance(school_group, list):
            return []

        logo_path_obj = pdf_config.get_school_logo_path(university)
        logo_path = str(logo_path_obj) if logo_path_obj else None
        logo_image = (
            create_responsive_image(logo_path, 0.85 * inch, 0.85 * inch) if logo_path else None
        )
        logo_cell_content = logo_image or Paragraph("", self.styles["SmallText"])

        details_content = []

        school_details = self.school_content_generator.school_db.get(university, {}) or {}
        ranking_info = school_details.get("排名", {}) or {}
        ranking = ranking_info.get("QS世界排名", "N/A") if isinstance(ranking_info, dict) else "N/A"
        feature = school_details.get("特点", "") or ""

        meta_color = pdf_config.text_muted
        title_text = f"<b>{university}</b>"
        if ranking and ranking != "N/A" and str(ranking).strip():
            title_text += f" &nbsp;<font size='-1' color='{meta_color}'>QS {ranking}</font>"
        if feature and feature.strip():
            title_text += f" &nbsp;<font size='-1' color='{meta_color}'>| {feature}</font>"
        details_content.append(
            Paragraph(
                f"<font color='{pdf_config.accent_gold}'>No.{rank}</font>",
                self.styles["LeftSmallText"],
            )
        )
        details_content.append(Paragraph(title_text, self.styles["SubTitle"]))
        details_content.append(Spacer(1, 0.1 * inch))

        valid_group = [s for s in school_group if isinstance(s, dict)]
        valid_group.sort(key=lambda s: clip_probability_coerce(s.get("probability")), reverse=True)
        overflow = max(0, len(valid_group) - MAX_MAJORS_PER_CARD)
        capped_group = valid_group[:MAX_MAJORS_PER_CARD]

        for idx, school_data in enumerate(capped_group):
            if not school_data or not isinstance(school_data, dict):
                continue

            _, major, probability = self.data_processor.extract_school_info(school_data)
            if not major:
                major = "未知专业"
            if not isinstance(probability, (int, float)):
                probability = 0.0

            prob_color = pdf_config.get_probability_color(probability)
            prob_text = tier_display_label(
                school_data.get("label") or _tier_label_from_prob(probability)
            )

            prob_p = Paragraph(
                f"<font color='white'><b>{prob_text} {probability:.0%}</b></font>",
                ParagraphStyle(
                    name="ProbTag",
                    fontName=self.styles["CustomBody"].fontName,
                    fontSize=8.5,
                    textColor=colors.white,
                    backColor=colors.HexColor(prob_color),
                    alignment=TA_CENTER,
                    leading=15,
                    borderPadding=(2, 4, 2, 4),
                    spaceBefore=1,
                    spaceAfter=1,
                ),
            )

            major_p = Paragraph(f"<b>{major}</b>", self.styles["CustomBody"])

            major_table = Table([[major_p, prob_p]], colWidths=[None, "26%"])
            major_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (1, 0), (1, 0), 8),
                        ("RIGHTPADDING", (1, 0), (1, 0), 0),
                    ]
                )
            )
            details_content.append(major_table)

            detailed_info = None
            try:
                detailed_info = get_school_major_details(university, major)
            except Exception as e:
                logger.warning(f"获取专业详情失败: {university} - {major}: {e}")

            if detailed_info and detailed_info != "暂无详细信息" and detailed_info.strip():
                compact_info = self._format_detailed_info_compact(detailed_info)
                if compact_info:
                    details_content.append(Paragraph(compact_info, self.styles["SmallText"]))

            ai_note = (school_notes_index or {}).get((university, major))
            if ai_note:
                safe_note = html.escape(ai_note.strip())
                details_content.append(
                    Paragraph(
                        f"<font color='{pdf_config.text_secondary}' size='9'>"
                        f"<b>AI 解读</b>：{safe_note}</font>",
                        self.styles["SmallText"],
                    )
                )

            if idx < len(capped_group) - 1:
                details_content.append(Spacer(1, 0.08 * inch))

        if overflow > 0:
            details_content.append(Spacer(1, 0.08 * inch))
            details_content.append(
                Paragraph(
                    f"<font color='#666666'>… 另有 {overflow} 个专业，详见系统内完整列表</font>",
                    self.styles["SmallText"],
                )
            )

        application_advice_list = school_details.get("申请建议", [])
        if (
            application_advice_list
            and isinstance(application_advice_list, list)
            and len(application_advice_list) > 0
        ):
            num_to_select = min(len(application_advice_list), 2)
            if num_to_select > 0:
                selected_advice = application_advice_list[:num_to_select]
                details_content.append(Spacer(1, 0.12 * inch))
                advice_points = [
                    f"- {item}" for item in selected_advice if item and str(item).strip()
                ]
                if advice_points:
                    advice_text = "<b>申请要点:</b><br/>" + "<br/>".join(advice_points)
                    details_content.append(Paragraph(advice_text, self.styles["LeftSmallText"]))

        card_table_data = [[logo_cell_content, details_content]]
        card_table = Table(card_table_data, colWidths=[1.05 * inch, None])
        table_style_commands = self.styler.get_table_style_with_font(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(pdf_config.accent_gold)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                ("LINEAFTER", (0, 0), (0, -1), 0.6, colors.HexColor(pdf_config.divider_color)),
                ("LEFTPADDING", (0, 0), (0, -1), 12),
                ("RIGHTPADDING", (0, 0), (0, -1), 10),
                ("LEFTPADDING", (1, 0), (1, -1), 14),
                ("RIGHTPADDING", (1, 0), (1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(pdf_config.panel_background)),
                ("BACKGROUND", (1, 0), (1, -1), colors.HexColor(pdf_config.card_background)),
                ("VALIGN", (0, 0), (0, -1), "MIDDLE"),
            ]
        )
        card_table.setStyle(TableStyle(table_style_commands))

        return [card_table]

    def _format_detailed_info_compact(self, detailed_info: str) -> str:
        if not detailed_info or not isinstance(detailed_info, str):
            return ""

        lines = detailed_info.split("\n")
        key_fields = {
            "GPA": "GPA要求",
            "IELTS": "IELTS",
            "TOEFL": "TOEFL",
            "学费": "学费",
            "截止": "申请截止时间",
        }
        compact_parts = []

        for key, field in key_fields.items():
            found_line = next(
                (line for line in lines if field in line and any(c in line for c in (":", "："))),
                None,
            )
            if found_line:
                value = re.split(r"[：:]", found_line, maxsplit=1)[1].strip()

                if key == "学费":
                    match = re.search(r"(HK\$|RMB|USD|\$|¥)?\s*[\d,]+", value)
                    if match:
                        value = match.group(0).strip()
                    else:
                        continue

                if value and value != "N/A" and value.strip():
                    compact_parts.append(f"<b>{key}</b>: {value}")

        if compact_parts:
            return "<font color='#666666'>" + " | ".join(compact_parts) + "</font>"
        return ""


def _tier_label_from_prob(probability: float) -> str:
    from src.agent.schemas import compute_tiers

    if not isinstance(probability, (int, float)):
        return "冲刺"
    return compute_tiers([clip_probability_coerce(probability)])[0]
