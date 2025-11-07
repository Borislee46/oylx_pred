from datetime import datetime
from typing import List

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.utils import create_responsive_image


class CoverPageBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler

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

