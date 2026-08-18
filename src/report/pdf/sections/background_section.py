import io

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import Image, KeepTogether, Paragraph, Spacer, Table, TableStyle

from src.report.pdf.config import pdf_config
from src.report.pdf.content.radar import (
    create_student_background_radar_chart,
)
from src.report.pdf.content.social_proof import build_background_social_proof
from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.sections.analyst_notes import (
    AnalystNotesGenerator,
)
from src.report.pdf.utils import SchoolDataProcessor, resolve_language_score
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

    def create_background_section(
        self,
        user_data: dict,
        *,
        skip_analyst_notes: bool = False,
        unified_results: list[dict] | None = None,
    ) -> list:
        story = []
        if not user_data or not isinstance(user_data, dict):
            user_data = {}

        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm

        story.extend(self.styler.make_section_header("个人背景竞争力分析", page_width))
        story.append(Spacer(1, 0.05 * inch))
        story.append(
            Paragraph(
                "以下雷达图与背景表为方案的数据依据；选校概率与 AI 解读见后文各院校卡片。",
                self.styles["SmallText"],
            )
        )
        story.append(Spacer(1, 0.08 * inch))

        story.append(Paragraph("核心维度分析", self.styles["SubTitle"]))
        proof_html = build_background_social_proof(unified_results)
        proof_style = ParagraphStyle(
            name="SocialProof",
            parent=self.styles["SmallText"],
            alignment=TA_LEFT,
            leading=14,
            spaceAfter=0,
            textColor=colors.HexColor(pdf_config.text_secondary),
        )
        radar_fig = None
        try:
            radar_fig = create_student_background_radar_chart(user_data)
            if radar_fig:
                buf = io.BytesIO()
                radar_fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.15)
                buf.seek(0)

                radar_image = Image(buf)
                radar_w = page_width * 0.52
                if radar_image.imageWidth > 0:
                    scale_ratio = radar_w / radar_image.imageWidth
                    radar_image.drawWidth = radar_w
                    radar_image.drawHeight = radar_image.imageHeight * scale_ratio
                radar_image.hAlign = "CENTER"

                if proof_html:
                    proof_para = Paragraph(proof_html, proof_style)
                    proof_w = page_width - radar_w - 0.12 * inch
                    proof_box = Table([[proof_para]], colWidths=[proof_w])
                    proof_box.setStyle(
                        TableStyle(
                            [
                                (
                                    "BACKGROUND",
                                    (0, 0),
                                    (-1, -1),
                                    colors.HexColor(pdf_config.panel_background),
                                ),
                                (
                                    "LINEBEFORE",
                                    (0, 0),
                                    (0, -1),
                                    3,
                                    colors.HexColor(pdf_config.accent_gold),
                                ),
                                (
                                    "BOX",
                                    (0, 0),
                                    (-1, -1),
                                    0.8,
                                    colors.HexColor(pdf_config.border_color),
                                ),
                                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                                ("TOPPADDING", (0, 0), (-1, -1), 10),
                                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                            ]
                        )
                    )
                    layout = Table(
                        [[radar_image, proof_box]],
                        colWidths=[radar_w, proof_w],
                    )
                    layout.setStyle(
                        TableStyle(
                            [
                                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("LEFTPADDING", (0, 0), (0, 0), 0),
                                ("RIGHTPADDING", (0, 0), (0, 0), 8),
                                ("LEFTPADDING", (1, 0), (1, 0), 4),
                            ]
                        )
                    )
                    story.append(layout)
                else:
                    story.append(radar_image)
                story.append(Spacer(1, 0.25 * inch))
            elif proof_html:
                story.append(Paragraph(proof_html, proof_style))
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

        gpa_display = (
            user_data.get("gpa_raw") or user_data.get("gpa") or user_data.get("gpa_score", "N/A")
        )
        gpa_scale = user_data.get("gpa_scale", "N/A")
        language_type = user_data.get("language_type", "语言")
        lang_raw = user_data.get("language_score_raw")
        lang_score_val = resolve_language_score(lang_raw, user_data.get("language_score"))
        language_score = (
            lang_raw
            if lang_raw
            else (f"{lang_score_val:g}" if lang_score_val is not None else "N/A")
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
        background_major_2 = user_data.get("background_major_2", "") or ""
        is_dual_degree = user_data.get("is_dual_degree", False)
        degree_type = user_data.get("degree_type", "") or ""

        academic_rows = [
            ("本科院校", str(background_university)),
            ("本科专业", str(background_major)),
            ("GPA 成绩", gpa_text),
            (f"{language_type} 成绩", str(language_score)),
        ]

        if background_major_2:
            if degree_type:
                label = f"第二专业（{degree_type}）"
            elif is_dual_degree:
                label = "第二专业（双学位）"
            else:
                label = "第二专业（辅修）"
            academic_rows.insert(2, (label, str(background_major_2)))
        academic_table = self._build_kv_table("学术背景", academic_rows)

        soft_skills = {
            "研究经历": user_data.get("research_count", 0) or 0,
            "实习经历": user_data.get("internship_count", 0) or 0,
            "获奖经历": user_data.get("award_count", 0) or 0,
            "论文发表": user_data.get("paper_count", 0) or 0,
        }

        for key in soft_skills:
            if not isinstance(soft_skills[key], (int, float)):
                soft_skills[key] = 0

        soft_rows = [
            (name, f"{int(count)} 项" if isinstance(count, (int, float)) else "0 项")
            for name, count in soft_skills.items()
        ]
        soft_skills_table = self._build_kv_table("软实力背景", soft_rows)

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

        analyst_notes = ""
        if not skip_analyst_notes:
            analyst_notes = self.analyst_notes_generator.generate_analyst_notes(
                user_data, soft_skills
            )

        left_aligned_body_style = ParagraphStyle(
            name="LeftBody",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=0,
        )

        background_section_content: list = [two_column_table, Spacer(1, 0.22 * inch)]
        if analyst_notes:
            notes_paragraph = Paragraph(analyst_notes, left_aligned_body_style)
            notes_table = Table(
                [[notes_paragraph]],
                style=[
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        colors.HexColor(pdf_config.panel_background),
                    ),
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(pdf_config.accent_gold)),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 16),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 16),
                    ("TOPPADDING", (0, 0), (-1, -1), 15),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 15),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ],
            )
            background_section_content.append(notes_table)

        story.append(KeepTogether(background_section_content))

        return story

    def _build_kv_table(self, title: str, rows: list[tuple[str, str]]) -> Table:
        header_style = ParagraphStyle(
            name="KVHeader",
            parent=self.styles["CustomBody"],
            textColor=colors.white,
            fontSize=10.5,
            leading=14,
            spaceAfter=0,
            alignment=TA_LEFT,
        )
        label_style = ParagraphStyle(
            name="KVLabel",
            parent=self.styles["CustomBody"],
            textColor=colors.HexColor(pdf_config.text_secondary),
            spaceAfter=0,
        )
        value_style = ParagraphStyle(
            name="KVValue",
            parent=self.styles["CustomBody"],
            textColor=colors.HexColor(pdf_config.text_primary),
            spaceAfter=0,
        )

        data = [[Paragraph(f"<b>{title}</b>", header_style), ""]]
        for label, value in rows:
            data.append(
                [
                    Paragraph(f"<b>{label}</b>", label_style),
                    Paragraph(str(value), value_style),
                ]
            )

        table = Table(data, colWidths=[1.4 * inch, None])
        commands = self.styler.get_table_style_with_font(
            [
                ("SPAN", (0, 0), (1, 0)),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(pdf_config.theme_primary)),
                ("TOPPADDING", (0, 0), (-1, 0), 7),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 7),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BACKGROUND", (0, 1), (0, -1), colors.HexColor(pdf_config.panel_background)),
                ("BACKGROUND", (1, 1), (1, -1), colors.HexColor(pdf_config.card_background)),
                ("LINEBELOW", (0, 1), (-1, -1), 0.6, colors.HexColor(pdf_config.divider_color)),
                ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                ("LEFTPADDING", (0, 0), (-1, -1), 9),
                ("RIGHTPADDING", (0, 0), (-1, -1), 9),
                ("TOPPADDING", (0, 1), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ]
        )
        table.setStyle(TableStyle(commands))
        return table
