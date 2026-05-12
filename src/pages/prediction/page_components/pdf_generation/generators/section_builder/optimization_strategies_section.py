# !!EXPERIMENTAL: recovered from deleted commit. Grep this line to find/remove all experimental files.

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import KeepTogether, PageBreak, Paragraph, Spacer

from src.pages.prediction.page_components.pdf_generation.config import pdf_config
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.school_detail import (
    SchoolDetailBuilder,
)
from src.pages.prediction.page_components.pdf_generation.utils import SchoolDataProcessor


class OptimizationStrategiesSectionBuilder:
    def __init__(
        self,
        styler: PDFStyler,
        data_processor: SchoolDataProcessor,
        school_detail_builder: SchoolDetailBuilder,
    ):
        self.styles = styler.styles
        self.styler = styler
        self.data_processor = data_processor
        self.school_detail_builder = school_detail_builder

    def create_optimization_strategies_section(
        self, optimization_results: dict, cases_df: pd.DataFrame
    ) -> list:
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

                school_detail = self.school_detail_builder.create_grouped_school_detail(
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
