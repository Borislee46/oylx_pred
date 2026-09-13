import io
from datetime import datetime

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    CondPageBreak,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
)

from src.utils.logger import setup_logger

from ..sections import PDFSectionBuilder
from ..tier_labels import attach_canonical_tier_labels
from .pdf_styler import PDFStyler

logger = setup_logger("page3", "prediction")


class WatermarkedDocTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename,
        user_nickname: str | None = None,
        chinese_font_name: str | None = None,
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
        from ..config import pdf_config

        page_width, page_height = A4
        margin = self.leftMargin
        right_edge = page_width - self.rightMargin
        gold = colors.HexColor(pdf_config.accent_gold)
        muted = colors.HexColor(pdf_config.text_muted)

        top_y = page_height - 1.45 * cm
        self.canv.setStrokeColor(self.theme_primary)
        self.canv.setLineWidth(1.1)
        self.canv.line(margin, top_y, right_edge, top_y)
        self.canv.setStrokeColor(gold)
        self.canv.setLineWidth(1.1)
        self.canv.line(margin, top_y - 2.4, margin + 46, top_y - 2.4)

        try:
            self.canv.setFont(self.chinese_font_name, 7.5)
        except (KeyError, AttributeError):
            self.canv.setFont("Helvetica", 7.5)
        self.canv.setFillColor(muted)
        self.canv.drawString(margin, page_height - 1.18 * cm, "Signals 留学择校系统")

        self.canv.setStrokeColor(colors.HexColor(pdf_config.divider_color))
        self.canv.setLineWidth(0.8)
        self.canv.line(margin, 1.45 * cm, right_edge, 1.45 * cm)

        self.canv.setFont("Helvetica", 8)
        self.canv.setFillColor(muted)
        self.canv.drawString(margin, 1.0 * cm, "Signals | Confidential")
        self.canv.drawRightString(right_edge, 1.0 * cm, f"— {self.page} —")

    def _draw_watermark(self):
        page_width, page_height = A4
        current_date = datetime.now().strftime("%Y-%m-%d")
        watermark_text = f"Signals {current_date} {self.user_nickname}"

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
        user_data: dict,
        cases_df: pd.DataFrame,
        user_nickname: str,
        optimization_results: dict | None = None,
        unified_results: list[dict] | None = None,
        ai_explanation: dict | None = None,
    ) -> bytes:
        try:
            buffer = io.BytesIO()
            from ..config import pdf_config

            final_nickname = user_nickname or user_data.get("user_nickname") or "用户"
            labeled_results = attach_canonical_tier_labels(unified_results or [])
            if labeled_results:
                unified_results = labeled_results
                if optimization_results:
                    optimization_results = _sync_optimization_labels(
                        optimization_results, labeled_results
                    )

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

            story.extend(
                self.section_builder.create_executive_summary(
                    user_data, unified_results, ai_explanation
                )
            )
            story.append(PageBreak())

            skip_analyst = bool(
                ai_explanation
                and isinstance(ai_explanation, dict)
                and str(ai_explanation.get("overview") or "").strip()
            )
            story.extend(
                self.section_builder.create_background_section(
                    user_data,
                    skip_analyst_notes=skip_analyst,
                    unified_results=unified_results,
                )
            )
            story.append(CondPageBreak(3.0 * inch))

            improvement = self.section_builder.create_improvement_section(
                user_data, unified_results, ai_explanation
            )
            if improvement:
                story.extend(improvement)
                story.append(CondPageBreak(3.0 * inch))

            if optimization_results:
                story.extend(
                    self.section_builder.create_optimization_strategies_section(
                        optimization_results,
                        cases_df,
                        ai_explanation=ai_explanation,
                    )
                )

            doc.build(story)
            buffer.seek(0)

            return buffer.getvalue()

        except Exception as e:
            logger.error(f"PDF报告生成失败: {str(e)}", exc_info=True)
            raise

    def generate_report_from_pathfinder(
        self,
        unified_results: list[dict],
        input_data: dict,
        user_nickname: str,
    ) -> bytes:
        if not unified_results:
            raise ValueError("unified_results 为空，无法生成报告")

        labeled = attach_canonical_tier_labels(unified_results)

        return self.generate_report(
            user_data=input_data,
            optimization_results=_adapt_labeled_to_optimization(labeled),
            cases_df=None,
            user_nickname=user_nickname,
            unified_results=labeled,
        )

    def generate_report_from_pathfinder_v2(
        self,
        portfolio_combo: list[dict],
        ai_explanation: dict | None,
        input_data: dict,
        user_nickname: str,
    ) -> bytes:
        if not portfolio_combo:
            raise ValueError("portfolio_combo 为空，无法生成报告")

        labeled = attach_canonical_tier_labels(portfolio_combo)

        return self.generate_report(
            user_data=input_data,
            optimization_results=_adapt_labeled_to_optimization(
                labeled, strategy_name="Pathfinder 智能选校方案"
            ),
            cases_df=None,
            user_nickname=user_nickname,
            unified_results=labeled,
            ai_explanation=ai_explanation,
        )


def _adapt_labeled_to_optimization(
    labeled_results: list[dict],
    *,
    strategy_name: str = "Pathfinder 推荐方案",
) -> dict:
    return {
        "recommendations": [
            {
                "strategy_name": strategy_name,
                "schools": labeled_results,
            }
        ],
    }


def _sync_optimization_labels(
    optimization_results: dict,
    labeled_results: list[dict],
) -> dict:
    label_by_key = {
        (
            str(r.get("university", "") or "").strip(),
            str(r.get("major", "") or "").strip(),
        ): r.get("label")
        for r in labeled_results
        if r.get("label")
    }
    synced = dict(optimization_results)
    recs = []
    for rec in optimization_results.get("recommendations", []) or []:
        if not isinstance(rec, dict):
            recs.append(rec)
            continue
        schools = []
        for s in rec.get("schools", []) or []:
            if not isinstance(s, dict):
                schools.append(s)
                continue
            row = dict(s)
            key = (
                str(row.get("university", "") or "").strip(),
                str(row.get("major", "") or "").strip(),
            )
            if key in label_by_key:
                row["label"] = label_by_key[key]
            schools.append(row)
        recs.append({**rec, "schools": schools})
    synced["recommendations"] = recs
    return synced
