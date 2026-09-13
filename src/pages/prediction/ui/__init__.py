from .background_summary import build_background_summary
from .content_display import display_content
from .fragments import form_fragment, results_fragment
from .handler import handle_form_submission
from .lead_in_ui import (
    render_lead_in_actions,
    render_lead_in_ghost,
)
from .page_render import (
    render_background_summary_bar,
    render_lead_in_section,
    render_page_footer,
    render_timeline_rail,
)
from .page_state_machine import HKPagePhase, PageStateMachine
from .progress_text import soften_progress_text, status_label
from .scroll_utils import scroll_to_anchor
from .submission_logger import build_user_form_log, log_first_submission_if_needed
from .timeline import (
    TIMELINE_PHASES,
    TIMELINE_STATE_INDEX,
    timeline_current_index,
    timeline_phases,
)
from .ui_elements import (
    display_back_to_homepage,
    display_feedback_section,
    get_product_logo_image_as_base64,
    render_header,
)

__all__ = [
    "handle_form_submission",
    "soften_progress_text",
    "status_label",
    "build_background_summary",
    "build_user_form_log",
    "log_first_submission_if_needed",
    "display_back_to_homepage",
    "display_content",
    "display_feedback_section",
    "get_product_logo_image_as_base64",
    "render_header",
    "render_lead_in_actions",
    "render_lead_in_ghost",
    "timeline_phases",
    "timeline_current_index",
    "TIMELINE_PHASES",
    "TIMELINE_STATE_INDEX",
    "HKPagePhase",
    "PageStateMachine",
    "form_fragment",
    "results_fragment",
    "render_background_summary_bar",
    "render_lead_in_section",
    "render_page_footer",
    "render_timeline_rail",
    "scroll_to_anchor",
]
