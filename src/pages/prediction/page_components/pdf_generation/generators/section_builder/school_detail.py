import random
import re
from typing import Dict, List, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.content.school_specific_content import (
    SchoolSpecificContentGenerator,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.utils import (
    SchoolDataProcessor,
    create_responsive_image,
)
from src.pages.prediction.prediction_utils import get_school_major_details
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class SchoolDetailBuilder:
    def __init__(self, styler: PDFStyler, data_processor: SchoolDataProcessor):
        self.styles = styler.styles
        self.styler = styler
        self.data_processor = data_processor
        self.school_content_generator = SchoolSpecificContentGenerator()

    def create_grouped_school_detail(
        self,
        university: str,
        school_group: List[Dict],
        rank: int,
        cases_df: pd.DataFrame,
        adaptive_thresholds: Optional[Dict] = None,
    ) -> List:
        if not university or not isinstance(university, str):
            university = "未知大学"
        if not school_group or not isinstance(school_group, list):
            return []
        if adaptive_thresholds is None:
            adaptive_thresholds = {}

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

        title_text = f"<b>{university}</b>"
        if ranking and ranking != "N/A" and str(ranking).strip():
            title_text += f" | <font size='-1' color='#666666'>QS排名: {ranking}</font>"
        if feature and feature.strip():
            title_text += f" | <font size='-1' color='#666666'>{feature}</font>"
        details_content.append(Paragraph(title_text, self.styles["SubTitle"]))

        details_content.append(Spacer(1, 0.12 * inch))

        for idx, school_data in enumerate(school_group):
            if not school_data or not isinstance(school_data, dict):
                continue

            _, major, probability = self.data_processor.extract_school_info(school_data)
            if not major:
                major = "未知专业"
            if not isinstance(probability, (int, float)):
                probability = 0.0

            prob_color = pdf_config.get_probability_color(probability)
            prob_text = self._get_probability_label(probability, adaptive_thresholds)

            prob_p = Paragraph(
                f"<font color='white'>{prob_text}</font>",
                ParagraphStyle(
                    name="ProbTag",
                    fontName=self.styles["CustomBody"].fontName,
                    fontSize=9,
                    textColor=colors.white,
                    backColor=prob_color,
                    alignment=TA_CENTER,
                    leading=14,
                    spaceBefore=2,
                    spaceAfter=2,
                ),
            )

            major_p = Paragraph(str(major), self.styles["CustomBody"])

            major_table = Table([[major_p, prob_p]], colWidths=[None, "16%"])
            major_table.setStyle(
                TableStyle(
                    [
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("LEFTPADDING", (1, 0), (1, 0), 8),
                        ("RIGHTPADDING", (1, 0), (1, 0), 8),
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

            if idx < len(school_group) - 1:
                details_content.append(Spacer(1, 0.08 * inch))

        application_advice_list = school_details.get("申请建议", [])
        if (
            application_advice_list
            and isinstance(application_advice_list, list)
            and len(application_advice_list) > 0
        ):
            num_to_select = min(len(application_advice_list), 2)
            if num_to_select > 0:
                selected_advice = random.sample(application_advice_list, num_to_select)
                details_content.append(Spacer(1, 0.12 * inch))
                advice_points = [
                    f"- {item}" for item in selected_advice if item and str(item).strip()
                ]
                if advice_points:
                    advice_text = "<b>申请要点:</b><br/>" + "<br/>".join(advice_points)
                    details_content.append(Paragraph(advice_text, self.styles["LeftSmallText"]))

        card_table_data = [[logo_cell_content, details_content]]
        card_table = Table(card_table_data, colWidths=[1.1 * inch, None])
        table_style_commands = self.styler.get_table_style_with_font(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#FAFAFA")),
            ]
        )
        card_table.setStyle(TableStyle(table_style_commands))

        return [card_table]

    def _get_probability_label(
        self, probability: float, adaptive_thresholds: Optional[Dict] = None
    ) -> str:
        if not isinstance(probability, (int, float)):
            probability = 0.0
        probability = max(0.0, min(1.0, float(probability)))

        if adaptive_thresholds and isinstance(adaptive_thresholds, dict):
            safety_threshold = adaptive_thresholds.get(
                "safety", SCHOOL_CATEGORY_THRESHOLDS.get("safety", 0.7)
            )
            target_threshold = adaptive_thresholds.get(
                "target_lower", SCHOOL_CATEGORY_THRESHOLDS.get("target_lower", 0.4)
            )
        else:
            safety_threshold = SCHOOL_CATEGORY_THRESHOLDS.get("safety", 0.7)
            target_threshold = SCHOOL_CATEGORY_THRESHOLDS.get("target_lower", 0.4)

        if not isinstance(safety_threshold, (int, float)):
            safety_threshold = 0.7
        if not isinstance(target_threshold, (int, float)):
            target_threshold = 0.4

        if probability >= safety_threshold:
            return "保底"
        elif probability >= target_threshold:
            return "目标"
        else:
            return "冲刺"

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
            found_line = next((line for line in lines if field in line and ":" in line), None)
            if found_line:
                value = found_line.split(":", 1)[1].strip()

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

