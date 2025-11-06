import io
import random
import re
from datetime import datetime
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.content.radar import (
    create_student_background_radar_chart,
)
from src.pages.prediction.page_components.pdf_generation.content.school_specific_content import (
    SchoolSpecificContentGenerator,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_ai_agent import (
    PDFAIAgent,
)
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.utils import (
    ContentFormatter,
    SchoolDataProcessor,
    create_responsive_image,
)
from src.pages.prediction.prediction_utils import get_school_major_details
from src.pages.prediction.school_combination_optimizer_algorithm.config import (
    SCHOOL_CATEGORY_THRESHOLDS,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


class PDFSectionBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler
        self.school_content_generator = SchoolSpecificContentGenerator()
        self.data_processor = SchoolDataProcessor()
        self.formatter = ContentFormatter()
        self.ai_agent = PDFAIAgent()

    def create_cover_page(self, user_nickname: str, report_title: str) -> List:
        story = []

        if not user_nickname or not isinstance(user_nickname, str):
            user_nickname = "用户"
        if not report_title or not isinstance(report_title, str):
            report_title = "留学申请方案报告"

        user_nickname = user_nickname.strip()
        report_title = report_title.strip()

        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm
        page_height = A4[1] - (pdf_config.top_margin + pdf_config.bottom_margin) * cm

        story.append(Spacer(1, 0.4 * inch))

        logo_path = pdf_config.get_product_logo_path()
        if logo_path:
            logo = create_responsive_image(str(logo_path), page_width * 0.28, page_height * 0.22)
            if logo:
                logo.hAlign = "CENTER"
                story.append(logo)
                story.append(Spacer(1, 0.5 * inch))
            else:
                story.append(Spacer(1, 0.3 * inch))
        else:
            story.append(Spacer(1, 0.3 * inch))

        story.append(Paragraph("EasyApply 留学择校系统", self.styles["CustomTitle"]))
        story.append(Spacer(1, 0.2 * inch))
        story.append(Paragraph(report_title, self.styles["SectionTitle"]))
        story.append(Spacer(1, 0.5 * inch))
        story.append(
            Paragraph(f"尊敬的 <b>{user_nickname}</b> 老师，您好：", self.styles["SubTitle"])
        )
        story.append(Spacer(1, 0.3 * inch))

        intro_text = (
            "本报告针对学生的申请情况生成，供您参考使用。<br/><br/>"
            "报告基于学生提供的个人背景信息，通过机器学习模型深度分析，生成了一套个性化的留学申请方案。"
            "报告将包含以下核心内容：<br/>"
            "- <b>个人背景竞争力分析</b>：全面评估学生的学术、语言及软实力背景。<br/>"
            "- <b>智能择校策略解读</b>：系统已为学生生成多种不同风险偏好的申请组合，"
            "本报告将详细解读所有策略方案，包括冲刺、目标和保底院校的分布情况。<br/><br/>"
            "建议您结合报告内容，为学生提供专业的申请指导。"
        )

        left_aligned_body_style = ParagraphStyle(
            name="LeftBodyOnCover",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=16,
            spaceAfter=12,
        )

        intro_para = Paragraph(intro_text, left_aligned_body_style)
        intro_table = Table([[intro_para]], colWidths=[page_width])
        intro_table.setStyle(
            TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 2, colors.HexColor(pdf_config.theme_primary)),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
                    ("LEFTPADDING", (0, 0), (-1, -1), 50),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 50),
                    ("TOPPADDING", (0, 0), (-1, -1), 40),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 40),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(intro_table)
        story.append(Spacer(1, 0.4 * inch))

        current_time = datetime.now().strftime("%Y年%m月%d日")
        time_style = ParagraphStyle(
            name="TimeStyle",
            parent=self.styles["SmallText"],
            alignment=TA_CENTER,
            textColor=colors.HexColor("#888888"),
        )
        story.append(Paragraph(f"报告生成时间: {current_time}", time_style))

        return story

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
                alignment=TA_CENTER,
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

        analyst_notes = self._generate_analyst_notes(user_data, soft_skills)

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

    def create_optimization_strategies_section(
        self, optimization_results: Dict, cases_df: pd.DataFrame
    ) -> List:
        story = []

        if not optimization_results or not isinstance(optimization_results, dict):
            optimization_results = {}
        if cases_df is None or not isinstance(cases_df, pd.DataFrame):
            cases_df = pd.DataFrame()

        recommendations = optimization_results.get("recommendations", []) or []
        adaptive_thresholds = optimization_results.get("adaptive_thresholds", {}) or {}

        if not recommendations:
            story.append(Paragraph("申请策略解读", self.styles["SectionTitle"]))
            empty_style = ParagraphStyle(
                name="EmptyStyle",
                parent=self.styles["SmallText"],
                textColor=colors.HexColor("#999999"),
                alignment=TA_LEFT,
            )
            story.append(Paragraph("未能生成申请策略的详细建议。", empty_style))
            return story

        for strategy_idx, strategy in enumerate(recommendations):
            if not strategy or not isinstance(strategy, dict):
                continue

            strategy_name = strategy.get("strategy_name") or f"申请策略{strategy_idx + 1}"
            schools = strategy.get("schools", []) or []

            if not schools:
                continue

            if strategy_idx > 0:
                story.append(PageBreak())

            strategy_header = [
                Paragraph(f"{strategy_name}解读", self.styles["SectionTitle"]),
                Spacer(1, 0.15 * inch),
            ]
            strategy_desc = pdf_config.get_strategy_description(strategy_name, len(schools))
            strategy_header.append(Paragraph(strategy_desc, self.styles["CustomBody"]))
            story.append(KeepTogether(strategy_header))
            story.append(Spacer(1, 0.2 * inch))

            grouped_schools = self.data_processor.group_schools_by_university(schools)
            if not grouped_schools:
                continue

            rank = 1
            for university, school_group in grouped_schools.items():
                if not university or not school_group:
                    continue

                school_detail = self._create_grouped_school_detail(
                    university, school_group, rank, cases_df, adaptive_thresholds
                )
                if school_detail:
                    story.append(KeepTogether(school_detail))

                rank += 1
                if rank <= len(grouped_schools):
                    story.append(Spacer(1, 0.2 * inch))

            if strategy_idx < len(recommendations) - 1:
                story.append(Spacer(1, 0.3 * inch))

        story.append(Spacer(1, 0.4 * inch))
        closing_style = ParagraphStyle(
            name="ClosingStyle",
            parent=self.styles["CustomBody"],
            fontSize=13,
            textColor=colors.HexColor("#1E90FF"),
            alignment=TA_CENTER,
            spaceBefore=10,
        )
        story.append(Paragraph("祝同学申请顺利！", closing_style))

        return story

    def _generate_analyst_notes(self, user_data: Dict, soft_skills: Dict) -> str:
        if not user_data:
            user_data = {}
        if not soft_skills:
            soft_skills = {}

        ai_notes = self.ai_agent.generate_analyst_notes(user_data, soft_skills)
        if ai_notes and ai_notes.strip():
            if not ai_notes.startswith("<b>分析师建议:</b>"):
                ai_notes = "<b>分析师建议:</b><br/>" + ai_notes
            return ai_notes

        return self._generate_fallback_analyst_notes(user_data, soft_skills)

    def _generate_fallback_analyst_notes(self, user_data: Dict, soft_skills: Dict) -> str:
        notes = ["<b>分析师建议:</b>"]

        gpa_value = user_data.get("gpa_score", 0)
        if isinstance(gpa_value, str) and (gpa_value == "未填写" or not gpa_value.strip()):
            gpa = 0.0
        else:
            try:
                gpa = float(gpa_value) if gpa_value else 0.0
            except (ValueError, TypeError):
                gpa = 0.0

        if gpa >= 3.8:
            notes.append(
                "- <b>学术表现卓越</b>：该学生拥有极具竞争力的GPA成绩，这将是其申请中的核心优势，"
                "建议顾问在指导文书写作时突出学生的学术能力。"
            )
        elif gpa >= 3.5:
            notes.append(
                "- <b>学术背景良好</b>：学生的GPA成绩达到了大部分名校的要求，具备扎实的学术基础。"
            )
        elif gpa > 0:
            notes.append(
                "- <b>学术背景提示</b>：学生的GPA成绩可能在申请顶尖项目时面临挑战，"
                "建议顾问通过高质量的文书、研究经历或GMAT/GRE成绩来帮助学生弥补短板。"
            )

        strong_points = [
            k for k, v in soft_skills.items() if isinstance(v, (int, float)) and v >= 3
        ]
        if strong_points:
            points_str = "、".join(strong_points)
            notes.append(
                f"- <b>软实力突出</b>：该学生在 <b>{points_str}</b> 方面积累了丰富的经验，"
                "这极大地增强了其申请竞争力。建议顾问在指导学生撰写简历和文书时，详细阐述这些经历的深度和成果。"
            )
        else:
            weak_points = [
                k for k, v in soft_skills.items() if isinstance(v, (int, float)) and v == 0
            ]
            if len(weak_points) >= 2:
                points_str = "、".join(weak_points)
                notes.append(
                    f"- <b>软实力建议</b>：我们注意到该学生在 <b>{points_str}</b> 等方面的经历尚有提升空间。"
                    "建议顾问在后续指导过程中，帮助学生强化相关背景。"
                )

        notes.append(
            "- <b>综合建议</b>：建议顾问结合三种申请策略，为学生选择最符合其个人目标和风险偏好的方案。"
            "报告仅供参考，具体申请方案需结合实际情况进行调整。"
        )

        return "<br/>".join(notes)

    def _create_grouped_school_detail(
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
                logger.warning(f"获取专业详情失败 {university} - {major}: {e}")

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
