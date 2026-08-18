from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm, inch
from reportlab.platypus import HRFlowable, Paragraph, Spacer, Table, TableStyle

from src.report.pdf.config import pdf_config
from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.utils import create_trimmed_logo_image


class CoverPageBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler

    def create_cover_page(self, user_nickname: str, report_title: str) -> list:
        story = []

        if not user_nickname or not isinstance(user_nickname, str):
            user_nickname = "用户"
        if not report_title or not isinstance(report_title, str):
            report_title = "留学申请方案报告"

        user_nickname = user_nickname.strip()
        report_title = report_title.strip()

        page_width = A4[0] - (pdf_config.left_margin + pdf_config.right_margin) * cm
        page_height = A4[1] - (pdf_config.top_margin + pdf_config.bottom_margin) * cm

        story.append(Spacer(1, 0.7 * inch))

        logo_path = pdf_config.get_product_logo_path()
        if logo_path:
            logo = create_trimmed_logo_image(str(logo_path), page_width * 0.22, page_height * 0.16)
            if logo:
                logo.hAlign = "CENTER"
                story.append(logo)
                story.append(Spacer(1, 0.5 * inch))
            else:
                story.append(Spacer(1, 0.2 * inch))
        else:
            story.append(Spacer(1, 0.2 * inch))

        story.append(Paragraph("Signals | 留学择校系统", self.styles["Eyebrow"]))
        story.append(Paragraph(report_title, self.styles["CustomTitle"]))
        story.append(Spacer(1, 0.14 * inch))
        story.append(
            HRFlowable(
                width=78,
                thickness=2,
                color=colors.HexColor(pdf_config.accent_gold),
                spaceBefore=0,
                spaceAfter=12,
                hAlign="CENTER",
            )
        )
        story.append(
            Paragraph("PERSONALIZED ADMISSION STRATEGY REPORT", self.styles["CoverSubtitle"])
        )
        story.append(Spacer(1, 0.55 * inch))

        story.append(Paragraph("尊敬的老师，您好：", self.styles["SubTitle"]))
        story.append(Spacer(1, 0.22 * inch))

        intro_text = (
            "本报告针对学生的申请情况量身定制，供您参考使用。<br/><br/>"
            "报告基于学生提供的个人背景信息，通过机器学习模型深度分析，生成了一套个性化的留学申请方案。"
            "核心内容包括：<br/><br/>"
            "<b>- 个人背景竞争力分析</b> &nbsp;&nbsp;全面评估学生的学术、语言及软实力背景。<br/><br/>"
            "<b>- 智能择校策略解读</b> &nbsp;&nbsp;系统已生成多种不同风险偏好的申请组合，本报告将逐一解读冲刺、目标与保底院校的分布。<br/><br/>"
            "建议您结合报告内容，为学生提供专业的申请指导。"
        )

        left_aligned_body_style = ParagraphStyle(
            name="LeftBodyOnCover",
            parent=self.styles["CustomBody"],
            alignment=TA_LEFT,
            leading=17,
            spaceAfter=0,
        )

        intro_para = Paragraph(intro_text, left_aligned_body_style)
        intro_table = Table([[intro_para]], colWidths=[page_width])
        intro_table.setStyle(
            TableStyle(
                [
                    ("LINEBEFORE", (0, 0), (0, -1), 3, colors.HexColor(pdf_config.accent_gold)),
                    ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor(pdf_config.border_color)),
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor(pdf_config.panel_background)),
                    ("LEFTPADDING", (0, 0), (-1, -1), 34),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 34),
                    ("TOPPADDING", (0, 0), (-1, -1), 30),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 30),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        story.append(intro_table)
        story.append(Spacer(1, 0.45 * inch))

        story.append(
            HRFlowable(
                width="40%",
                thickness=0.6,
                color=colors.HexColor(pdf_config.divider_color),
                spaceBefore=0,
                spaceAfter=8,
                hAlign="CENTER",
            )
        )
        current_time = datetime.now().strftime("%Y年%m月%d日")
        time_style = ParagraphStyle(
            name="TimeStyle",
            parent=self.styles["SmallText"],
            alignment=TA_CENTER,
            textColor=colors.HexColor(pdf_config.text_muted),
        )
        story.append(Paragraph(f"报告生成时间 {current_time} &nbsp;|&nbsp; 仅供参考", time_style))

        return story
