from typing import Dict, List

import pandas as pd

from src.agent.pdf_agent import PDFAgent
from src.pages.prediction.page_components.pdf_generation.generators.pdf_styler import PDFStyler
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.analyst_notes import (
    AnalystNotesGenerator,
)
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.background_section import (
    BackgroundSectionBuilder,
)
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.cover_page import (
    CoverPageBuilder,
)
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.optimization_strategies_section import (
    OptimizationStrategiesSectionBuilder,
)
from src.pages.prediction.page_components.pdf_generation.generators.section_builder.school_detail import (
    SchoolDetailBuilder,
)
from src.pages.prediction.page_components.pdf_generation.utils import (
    ContentFormatter,
    SchoolDataProcessor,
)


class PDFSectionBuilder:
    def __init__(self, styler: PDFStyler):
        self.styles = styler.styles
        self.styler = styler
        self.data_processor = SchoolDataProcessor()
        self.formatter = ContentFormatter()
        self.ai_agent = PDFAgent()

        self.analyst_notes_generator = AnalystNotesGenerator(self.ai_agent)
        self.cover_page_builder = CoverPageBuilder(styler)
        self.background_section_builder = BackgroundSectionBuilder(
            styler, self.data_processor, self.analyst_notes_generator
        )
        self.school_detail_builder = SchoolDetailBuilder(styler, self.data_processor)
        self.optimization_strategies_section_builder = OptimizationStrategiesSectionBuilder(
            styler, self.data_processor, self.school_detail_builder
        )

    def create_cover_page(self, user_nickname: str, report_title: str) -> List:
        return self.cover_page_builder.create_cover_page(user_nickname, report_title)

    def create_background_section(self, user_data: Dict) -> List:
        return self.background_section_builder.create_background_section(user_data)

    def create_optimization_strategies_section(
        self, optimization_results: Dict, cases_df: pd.DataFrame
    ) -> List:
        return self.optimization_strategies_section_builder.create_optimization_strategies_section(
            optimization_results, cases_df
        )
