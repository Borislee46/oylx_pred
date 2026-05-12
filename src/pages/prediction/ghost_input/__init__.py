"""Ghost text autocomplete — browser-side DeepSeek prefix continuation.

Frontend calls DeepSeek directly, no Streamlit round-trip.
API key source: app_config.OPEN_AI_API_KEY.
Analyze action → st.session_state._ghost_analyze_text.
Return "" means no user action to process.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

_FRONTEND_DIR = Path(__file__).parent / "frontend"


# ── Allowed universities (single source: prediction_rules.json) ──────────
def _load_allowed_universities() -> list[str]:
    """Return the canonical university list from config, with aliases appended."""
    try:
        rules_path = Path(__file__).parent.parent.parent.parent / "config" / "prediction_rules.json"
        if rules_path.exists():
            import json

            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            unis = rules.get("UNIVERSITY_DISPLAY_ORDER", [])
            # Also add common abbreviations that appear in completions
            aliases = [
                "港大",
                "港中文",
                "港科大",
                "港理工",
                "港城市",
                "港浸会",
                "港岭南",
                "港教育",
                "港都会",
                "港恒生",
                "港珠海",
                "新国立",
                "南洋理工",
                "新管理",
                "NUS",
                "NTU",
                "HKU",
                "CUHK",
                "HKUST",
                "CityU",
                "PolyU",
                "SMU",
            ]
            return unis + aliases
    except Exception:
        pass
    return []


_component = st.components.v1.declare_component(
    "ghost_text_area",
    path=str(_FRONTEND_DIR),
)
_log = logging.getLogger("ghost_input")


def ghost_text_area(
    api_key: str = "",
    api_base_url: str = "https://api.deepseek.com/beta",
    api_model: str = "deepseek-v4-flash",
    placeholder: str = "",
    initial_text: str = "",
    height: int = 100,
    rate_limit_max: int = 30,
    rate_limit_window_seconds: int = 60,
    rate_limit_cooldown_seconds: int = 15,
    key: str | None = None,
) -> str:
    """Render ghost-autocomplete textarea. Returns text on blur/accept/analyze, "" otherwise."""
    config = {
        "api_key": api_key,
        "api_base_url": api_base_url.rstrip("/"),
        "api_model": api_model,
        "initial_text": initial_text,
        "placeholder": placeholder,
        "rows": max(3, height // 24),
        "height": height + 40,
        "rate_max": rate_limit_max,
        "rate_window_ms": rate_limit_window_seconds * 1000,
        "rate_cooldown_ms": rate_limit_cooldown_seconds * 1000,
        "allowed_universities": _load_allowed_universities(),
        "allowed_regions": ["香港", "新加坡", "澳门", "马来西亚"],
    }

    result = _component(
        config=config,
        default=None,
        key=key,
    )

    _log.debug(
        "COMPONENT_RAW | result_type=%s result=%s",
        type(result).__name__,
        repr(result)[:200] if result is not None else "None",
    )

    # Return text on user actions: blur / accept / sync / analyze.
    if isinstance(result, dict) and result.get("action") in ("blur", "accept", "sync", "analyze"):
        event_id = str(result.get("event_id") or "")
        if event_id:
            last_event_key = f"_ghost_last_event_id:{key or 'default'}"
            if st.session_state.get(last_event_key) == event_id:
                return ""
            st.session_state[last_event_key] = event_id

        if result.get("action") not in ("sync",):
            _log_telemetry(result)
        text = str(result.get("text") or "")
        # "analyze" action: signal that the AI button inside the iframe
        # was clicked.  Store the text so render_lead_in_actions can pick
        # it up and run the AI agent.
        if result.get("action") == "analyze" and text:
            st.session_state["_ghost_analyze_text"] = text
            _log.info("GHOST_ANALYZE | text_len=%d text=%s", len(text), text[:120])
        return text
    return ""


def _log_telemetry(result: dict) -> None:
    """Record ghost_input usage stats from the frontend."""
    telemetry = result.get("telemetry")
    if not telemetry:
        return
    counters = telemetry.get("counters", {})
    _log.info(
        "TELEMETRY | session=%s duration=%ss "
        "attempts=%d ok=%d fail=%d retry=%d "
        "cache_hit=%d cache_set=%d "
        "rule_hit=%d rule_miss=%d "
        "shown=%d accepted=%d dismissed=%d "
        "rate_limited=%d dedup=%d",
        telemetry.get("session", "?"),
        telemetry.get("duration_s", "?"),
        counters.get("fetch_attempt", 0),
        counters.get("fetch_ok", 0),
        counters.get("fetch_fail", 0),
        counters.get("fetch_retry", 0),
        counters.get("cache_hit", 0),
        counters.get("cache_set", 0),
        counters.get("rule_hit", 0),
        counters.get("rule_miss", 0),
        counters.get("suggestion_shown", 0),
        counters.get("suggestion_accepted", 0),
        counters.get("suggestion_dismissed", 0),
        counters.get("rate_limited", 0),
        counters.get("dedup_blocked", 0),
    )
    # Warn on elevated error rate
    ok = counters.get("fetch_ok", 0)
    fail = counters.get("fetch_fail", 0)
    total = ok + fail
    if total > 0 and fail / total > 0.5:
        _log.warning(
            "TELEMETRY | high error rate session=%s ok=%d fail=%d",
            telemetry.get("session", "?"),
            ok,
            fail,
        )
