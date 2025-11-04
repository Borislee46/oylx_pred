import io
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import BaseDocTemplate, Frame, NextPageTemplate, PageBreak, PageTemplate

from src.utils.logger import setup_logger

from .pdf_section_builder import PDFSectionBuilder
from .pdf_styler import PDFStyler

logger = setup_logger("page3", "prediction")


class WatermarkedDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename,
        user_nickname: Optional[str] = None,
        chinese_font_name: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(filename, **kwargs)
        from ..config import pdf_config

        self.user_nickname = user_nickname or "用户"
        self.chinese_font_name = chinese_font_name or "Helvetica"
        self.theme_primary = colors.HexColor(pdf_config.theme_primary)
        self.watermark_opacity = pdf_config.watermark_opacity
        self.watermark_font_size = pdf_config.watermark_font_size
        self.watermark_rotation = pdf_config.watermark_rotation
        self.watermark_x_spacing = pdf_config.watermark_x_spacing
        self.watermark_y_spacing = pdf_config.watermark_y_spacing

        frame_cover = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="cover")
        frame_normal = Frame(
            self.leftMargin, self.bottomMargin, self.width, self.height, id="normal"
        )

        cover_template = PageTemplate(id="CoverPage", frames=[frame_cover])
        later_template = PageTemplate(
            id="LaterPages", frames=[frame_normal], onPage=self._draw_header_footer_and_watermark
        )

        self.addPageTemplates([cover_template, later_template])

    def _draw_header_footer_and_watermark(self, canvas, doc):
        self.canv = canvas
        self.page = doc.page
        canvas.saveState()
        self._draw_header_footer()
        self._draw_watermark()
        canvas.restoreState()

    def _draw_header_footer(self):
        page_width, page_height = A4
        margin = self.leftMargin

        self.canv.setStrokeColor(self.theme_primary)
        self.canv.setLineWidth(1)
        self.canv.line(
            margin, page_height - 1.5 * cm, page_width - self.rightMargin, page_height - 1.5 * cm
        )
        self.canv.line(margin, 1.5 * cm, page_width - self.rightMargin, 1.5 * cm)

        page_num_text = f"Page {self.page}"
        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(colors.darkgrey)
        self.canv.drawRightString(page_width - self.rightMargin, 1.0 * cm, page_num_text)

    def _draw_watermark(self):
        page_width, page_height = A4
        current_date = datetime.now().strftime("%Y-%m-%d")
        watermark_text = f"EasyApply {current_date} {self.user_nickname}"

        self.canv.setFillColor(colors.black, alpha=self.watermark_opacity)
        try:
            pdfmetrics.getFont(self.chinese_font_name)
            self.canv.setFont(self.chinese_font_name, self.watermark_font_size)
        except (KeyError, AttributeError):
            self.canv.setFont("Helvetica", self.watermark_font_size)

        rows = int(page_height / self.watermark_y_spacing) + 1
        cols = int(page_width / self.watermark_x_spacing) + 1

        for row in range(rows):
            for col in range(cols):
                x_offset = (row % 2) * (self.watermark_x_spacing // 2)
                x_pos = col * self.watermark_x_spacing - x_offset
                y_pos = row * self.watermark_y_spacing

                margin = 40
                if (
                    margin <= x_pos <= page_width - margin
                    and margin <= y_pos <= page_height - margin
                ):
                    self.canv.saveState()
                    self.canv.translate(x_pos, y_pos)
                    self.canv.rotate(self.watermark_rotation)
                    self.canv.drawCentredString(0, 0, watermark_text)
                    self.canv.restoreState()


class PDFReportGenerator:
    def __init__(self):
        self.styler = PDFStyler()
        self.section_builder = PDFSectionBuilder(self.styler)

    def generate_report(
        self,
        user_data: Dict,
        prediction_results: Any,
        cases_df: pd.DataFrame,
        user_nickname: str,
        optimization_results: Optional[Dict] = None,
    ) -> bytes:
        try:
            buffer = io.BytesIO()
            from ..config import pdf_config

            final_nickname = user_nickname or user_data.get("user_nickname") or "用户"

            doc = WatermarkedDocTemplate(
                buffer,
                pagesize=A4,
                rightMargin=pdf_config.right_margin * cm,
                leftMargin=pdf_config.left_margin * cm,
                topMargin=pdf_config.top_margin * cm,
                bottomMargin=pdf_config.bottom_margin * cm,
                user_nickname=final_nickname,
                chinese_font_name=self.styler.chinese_font_name,
            )

            story = []

            story.extend(
                self.section_builder.create_cover_page(final_nickname, "个性化申请方案报告")
            )
            story.append(NextPageTemplate("LaterPages"))
            story.append(PageBreak())

            story.extend(self.section_builder.create_background_section(user_data))
            story.append(PageBreak())

            if optimization_results:
                story.extend(
                    self.section_builder.create_optimization_strategies_section(
                        optimization_results, cases_df
                    )
                )

            doc.build(story)
            buffer.seek(0)

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"PDF报告生成失败: {str(e)}", exc_info=True)
            raise e
