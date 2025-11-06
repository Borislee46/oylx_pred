from src.pages.prediction.page_components.back_to_homepage_section import (
    display_back_to_homepage,
)
from src.pages.prediction.page_components.combination_analysis_section import (
    display_combination_analysis_section,
)
from src.pages.prediction.page_components.content_display import display_content
from src.pages.prediction.page_components.feedback_section import (
    display_feedback_section,
)
from src.pages.prediction.page_components.header_section import (
    get_product_logo_image_as_base64,
    render_header,
)
from src.pages.prediction.page_components.result_section import (
    display_results_section,
)
from src.pages.prediction.page_components.submission_logger import (
    build_user_form_log,
    log_first_submission_if_needed,
)

__all__ = [
    "render_header",
    "get_product_logo_image_as_base64",
    "display_results_section",
    "display_content",
    "display_combination_analysis_section",
    "display_feedback_section",
    "display_back_to_homepage",
    "build_user_form_log",
    "log_first_submission_if_needed",
]
