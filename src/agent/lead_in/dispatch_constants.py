PENDING_KEY = "_lead_in_pending_text"
IN_PROGRESS_KEY = "_lead_in_in_progress"
TOOLS_FAILED_KEY = "_lead_in_tools_failed"
RUNNING_HASH_KEY = "_lead_in_running_hash"
RUNNING_TS_KEY = "_lead_in_running_ts"
RETRY_COUNT_KEY = "_lead_in_retry_count"
PROGRESS_STEPS_KEY = "_lead_in_progress_steps"
PROGRESS_TEXT_KEY = "_lead_in_progress_text"
PROGRESS_VARIANT_KEY = "_lead_in_progress_variant"
INTENT_BLOCKED_KEY = "_lead_in_intent_blocked"
INTENT_STEP_LABEL = "意图识别"
MSGS_KEY = "_lead_in_msgs"
TURN_KEY = "_lead_in_turn"
DISMISS_KEY = "_lead_in_feedback_dismissed"

RETRY_COOLDOWN = 1.5
MAX_RETRIES = 3
MAX_RETRY_ELAPSED = 120


class LeadInCancelled(Exception):
    """Raised when the user cancels an in-flight lead-in extraction."""
