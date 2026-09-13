import pandas as pd

from src.report.pdf.generators.pdf_styler import PDFStyler
from src.report.pdf.sections.analyst_notes import (
    AnalystNotesGenerator,
)
from src.report.pdf.sections.background_section import (
    BackgroundSectionBuilder,
)
from src.report.pdf.sections.cover_page import (
    CoverPageBuilder,
)
from src.report.pdf.sections.executive_summary import (
    ExecutiveSummaryBuilder,
)
from src.report.pdf.sections.improvement_suggestions import (
    ImprovementSectionBuilder,
)
from src.report.pdf.sections.optimization_strategies_section import (
    OptimizationStrategiesSectionBuilder,
)
from src.report.pdf.sections.school_detail import (
    SchoolDetailBuilder,
)
from src.report.pdf.utils import (
    SchoolDataProcessor,
)


class PDFSectionBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.data_processor = SchoolDataProcessor()

        self.analyst_notes_generator = AnalystNotesGenerator()
        self.cover_page_builder = CoverPageBuilder(styler)
        self.executive_summary_builder = ExecutiveSummaryBuilder(styler)
        self.background_section_builder = BackgroundSectionBuilder(
            styler, self.data_processor, self.analyst_notes_generator
        )
        self.school_detail_builder = SchoolDetailBuilder(styler, self.data_processor)
        self.optimization_strategies_section_builder = OptimizationStrategiesSectionBuilder(
            styler, self.data_processor, self.school_detail_builder
        )
        self.improvement_section_builder = ImprovementSectionBuilder(styler)

    def create_cover_page(self, user_nickname: str, report_title: str) -> list:
        return self.cover_page_builder.create_cover_page(user_nickname, report_title)

    def create_executive_summary(
        self,
        user_data: dict,
        unified_results: list[dict] | None,
        ai_explanation: dict | None = None,
    ) -> list:
        return self.executive_summary_builder.create_executive_summary(
            user_data, unified_results, ai_explanation
        )

    def create_background_section(
        self, user_data: dict, *, skip_analyst_notes: bool = False, unified_results=None
    ) -> list:
        return self.background_section_builder.create_background_section(
            user_data,
            skip_analyst_notes=skip_analyst_notes,
            unified_results=unified_results,
        )

    def create_improvement_section(
        self,
        user_data: dict,
        unified_results: list[dict] | None,
        ai_explanation: dict | None = None,
    ) -> list:
        return self.improvement_section_builder.create_improvement_section(
            user_data, unified_results, ai_explanation
        )

    def create_optimization_strategies_section(
        self,
        optimization_results: dict,
        cases_df: pd.DataFrame,
        ai_explanation: dict | None = None,
    ) -> list:
        return self.optimization_strategies_section_builder.create_optimization_strategies_section(
            optimization_results, cases_df, ai_explanation
        )
