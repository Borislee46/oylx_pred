import io
from typing import Dict, List

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.content.radar import (
    create_student_background_radar_chart,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.analyst_notes import (
    AnalystNotesGenerator,
)
from src.pages.prediction.page_components.pdf_generation.utils import SchoolDataProcessor
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class BackgroundSectionBuilder:
    def __init__(
        self,
        styler: PDFStyler,
        data_processor: SchoolDataProcessor,
        analyst_notes_generator: AnalystNotesGenerator,
    ):
        self.styles = styler.styles
        self.styler = styler
        self.data_processor = data_processor
        self.analyst_notes_generator = analyst_notes_generator

    def create_background_section(self, user_data: Dict) -> List:
        story = []
        if not user_data or not isinstance(user_data, dict):
            user_data = {}

        story.append(Paragraph("个人背景竞争力分析", self.styles["SectionTitle"]))
        story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("核心维度分析", self.styles["SubTitle"]))
        radar_fig = None
        try:
            from src.utils.school_level_service import SCHOOL_LEVEL_PRIORITY

            radar_fig = create_student_background_radar_chart(user_data, SCHOOL_LEVEL_PRIORITY)
            if radar_fig:
                buf = io.BytesIO()
                radar_fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.15)
                buf.seek(0)

                page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm
                radar_image = Image(buf)
                max_radar_width = page_width * 0.55
                if radar_image.imageWidth > 0:
                    scale_ratio = max_radar_width / radar_image.imageWidth
                    radar_image.drawWidth = max_radar_width
                    radar_image.drawHeight = radar_image.drawHeight * scale_ratio
                radar_image.hAlign = "CENTER"

                story.append(radar_image)
                story.append(Spacer(1, 0.25 * inch))
        except Exception as e:
            logger.error(f"无法生成雷达图: {e}", exc_info=True)
            error_style = ParagraphStyle(
                name="ErrorStyle",
                parent=self.styles["SmallText"],
                textColor=colors.HexColor("#999999"),
                alignment=TA_LEFT,
            )
            story.append(Paragraph("核心维度分析图暂时无法生成", error_style))
            story.append(Spacer(1, 0.25 * inch))
        finally:
            if radar_fig is not None:
                plt.close(radar_fig)

        story.append(Paragraph("背景详情", self.styles["SubTitle"]))
        story.append(Spacer(1, 0.12 * inch))

        gpa_display = user_data.get("gpa_raw") or user_data.get("gpa_score", "N/A")
        gpa_scale = user_data.get("gpa_scale", "N/A")
        language_type = user_data.get("language_type", "语言")
        language_score = user_data.get("language_score_raw") or user_data.get(
            "language_score", "N/A"
        )

        if isinstance(gpa_display, (int, float)):
            gpa_display = f"{gpa_display:.2f}".rstrip("0").rstrip(".")
        else:
            gpa_display = str(gpa_display) if gpa_display else "N/A"

        if gpa_scale and gpa_scale not in ("N/A", "未知", "", None):
            gpa_text = f"{gpa_display} (<b>{gpa_scale}</b>制)"
        else:
            gpa_text = str(gpa_display)

        background_university = user_data.get("background_university", "N/A") or "N/A"
        background_major = user_data.get("background_major", "N/A") or "N/A"

        academic_data = [
            [
                Paragraph("<b>本科院校</b>", self.styles["CustomBody"]),
                Paragraph(str(background_university), self.styles["CustomBody"]),
            ],
            [
                Paragraph("<b>本科专业</b>", self.styles["CustomBody"]),
                Paragraph(str(background_major), self.styles["CustomBody"]),
            ],
            [
                Paragraph("<b>GPA成绩</b>", self.styles["CustomBody"]),
                Paragraph(gpa_text, self.styles["CustomBody"]),
            ],
            [
                Paragraph(f"<b>{language_type}成绩</b>", self.styles["CustomBody"]),
                Paragraph(str(language_score), self.styles["CustomBody"]),
            ],
        ]
        academic_table = Table(academic_data, colWidths=[1.5 * inch, None])
        table_style_commands = self.styler.get_table_style_with_font(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8F9FA")),
            ]
        )
        academic_table.setStyle(TableStyle(table_style_commands))

        soft_skills = {
            "研究经历": user_data.get("research_count", 0) or 0,
            "实习经历": user_data.get("internship_count", 0) or 0,
            "获奖经历": user_data.get("award_count", 0) or 0,
            "论文发表": user_data.get("paper_count", 0) or 0,
        }

        for key in soft_skills:
            if not isinstance(soft_skills[key], (int, float)):
                soft_skills[key] = 0

        soft_skills_data = []
        for name, count in soft_skills.items():
            count_str = f"{int(count)} 项" if isinstance(count, (int, float)) else "0 项"
            soft_skills_data.append(
                [
                    Paragraph(name, self.styles["CustomBody"]),
                    Paragraph(count_str, self.styles["CustomBody"]),
                ]
            )

        soft_skills_table = Table(soft_skills_data, colWidths=[1.5 * inch, None])
        table_style_commands = self.styler.get_table_style_with_font(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("GRID", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F8F9FA")),
            ]
        )
        soft_skills_table.setStyle(TableStyle(table_style_commands))

        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm
        column_width = (page_width - 0.3 * inch) / 2

        two_column_table = Table(
            [[academic_table, soft_skills_table]],
            colWidths=[column_width, column_width],
        )
        two_column_table.setStyle(
            TableStyle(
                [
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (0, -1), 0),
                    ("RIGHTPADDING", (0, 0), (0, -1), 8),
                    ("LEFTPADDING", (1, 0), (1, -1), 8),
                    ("RIGHTPADDING", (1, 0), (1, -1), 0),
                ]
            )
        )

        analyst_notes = self.analyst_notes_generator.generate_analyst_notes(user_data, soft_skills)

        left_aligned_body_style = ParagraphStyle(
            name="LeftBody",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=0,
        )
        notes_paragraph = Paragraph(analyst_notes, left_aligned_body_style)

        notes_table = Table(
            [[notes_paragraph]],
            style=[
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor(pdf_config.theme_background),
                ),
                ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 14),
                ("RIGHTPADDING", (0, 0), (-1, -1), 14),
                ("TOPPADDING", (0, 0), (-1, -1), 14),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 14),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ],
        )

        background_section_content = [two_column_table, Spacer(1, 0.2 * inch), notes_table]
        story.append(KeepTogether(background_section_content))

        return story
