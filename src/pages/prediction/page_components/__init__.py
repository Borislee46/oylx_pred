"""页面组件模块

提供预测页面的各种UI组件和显示功能。
"""

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
    # 头部组件
    "render_header",
    "get_product_logo_image_as_base64",
    # 结果展示
    "display_results_section",
    # 内容显示
    "display_content",
    # 组合分析
    "display_combination_analysis_section",
    # 反馈
    "display_feedback_section",
    # 导航
    "display_back_to_homepage",
    # 日志
    "build_user_form_log",
    "log_first_submission_if_needed",
]

