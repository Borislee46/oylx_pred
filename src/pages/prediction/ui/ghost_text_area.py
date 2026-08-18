from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.utils.logger import setup_logger

_FRONTEND_DIR = Path(__file__).parent / "frontend"


def _load_allowed_universities() -> list[str]:
    from src.utils.schools.config_loader import UNIVERSITY_DISPLAY_ORDER

    unis = list(UNIVERSITY_DISPLAY_ORDER)
    aliases = [
        "港大",
        "港中文",
        "港中深",
        "港科大",
        "港理工",
        "港城市",
        "港浸会",
        "港岭南",
        "港教育",
        "港都会",
        "港恒生",
        "港珠海",
        "HKU",
        "CUHK",
        "CUHKSZ",
        "HKUST",
        "PolyU",
        "CityU",
        "HKBU",
        "LingnanU",
        "EdUHK",
        "HKMU",
        "HSUHK",
        "Chu Hai",
        "新国立",
        "南洋理工",
        "新管理",
        "NUS",
        "NTU",
        "SMU",
        "澳大",
        "澳科大",
        "澳理工",
        "澳城大",
        "UM",
        "MUST",
        "MPU",
        "CityU Macau",
        "University of Malaya",
        "UPM",
        "USM",
        "UKM",
    ]
    return unis + aliases


_component = st.components.v1.declare_component(
    "ghost_text_area",
    path=str(_FRONTEND_DIR),
)
_log = setup_logger("page3", "prediction")
_LEAD_IN_BUSY_KEY = "_lead_in_in_progress"
_GHOST_ERROR_TRACKER_KEY = "_ghost_error_tracker"
_GHOST_TOKEN_KEY = "_ghost_proxy_token"


def _resolve_ghost_proxy() -> tuple[int, str, bool]:
    from src.pages.prediction.ui.lead_in_wait_sse import (
        ensure_sse_server,
        issue_ghost_token,
        same_origin_sse_enabled,
    )

    port = ensure_sse_server()
    token = st.session_state.get(_GHOST_TOKEN_KEY)
    if not token:
        token = issue_ghost_token()
        st.session_state[_GHOST_TOKEN_KEY] = token
    return port, token, same_origin_sse_enabled()


def ghost_text_area(
    *,
    enabled: bool = False,
    api_model: str = "deepseek-v4-flash",
    placeholder: str = "",
    initial_text: str = "",
    height: int = 100,
    rate_limit_max: int = 30,
    rate_limit_window_seconds: int = 60,
    rate_limit_cooldown_seconds: int = 15,
    key: str | None = None,
    lead_in_busy: bool = False,
) -> str:
    error_tracker = st.session_state.get(_GHOST_ERROR_TRACKER_KEY, {"ok": 0, "fail": 0})
    total = error_tracker.get("ok", 0) + error_tracker.get("fail", 0)
    error_rate = error_tracker.get("fail", 0) / max(total, 1)
    if error_rate > 0.5 and total >= 6:
        throttle_hint = "high"
    elif error_rate > 0.3 and total >= 6:
        throttle_hint = "normal"
    else:
        throttle_hint = "none"

    proxy_token = ""
    proxy_port = 0
    proxy_same_origin = False
    if enabled:
        proxy_port, proxy_token, proxy_same_origin = _resolve_ghost_proxy()

    config = {
        "api_enabled": bool(enabled),
        "ghost_token": proxy_token,
        "ghost_port": int(proxy_port or 0),
        "ghost_same_origin": bool(proxy_same_origin),
        "api_model": api_model,
        "initial_text": initial_text,
        "text_revision": int(st.session_state.get("_ghost_text_revision", 0)),
        "placeholder": placeholder,
        "rows": max(3, height // 24),
        "height": height + 40,
        "rate_max": rate_limit_max,
        "rate_window_ms": rate_limit_window_seconds * 1000,
        "rate_cooldown_ms": rate_limit_cooldown_seconds * 1000,
        "allowed_universities": _load_allowed_universities(),
        "allowed_regions": ["香港", "新加坡", "澳门", "马来西亚"],
        "lead_in_busy": lead_in_busy or bool(st.session_state.get(_LEAD_IN_BUSY_KEY, False)),
        "throttle_hint": throttle_hint,
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

    if isinstance(result, dict) and result.get("action") in ("blur", "accept", "analyze"):
        action = result.get("action")
        event_id = str(result.get("event_id") or "")
        lead_in_busy_now = lead_in_busy or bool(st.session_state.get(_LEAD_IN_BUSY_KEY, False))

        if event_id:
            last_event_key = f"_ghost_last_event_id:{key or 'default'}"
            prev = st.session_state.get(last_event_key)
            if prev == event_id:
                _log.debug("GHOST_DEDUP_BLOCK | action=%s event_id=%s", action, event_id)
                return ""
            _log.info(
                "GHOST_EVENT | action=%s event_id=%s prev_event_id=%s",
                action,
                event_id,
                prev,
            )
            st.session_state[last_event_key] = event_id
        else:
            _log.warning("GHOST_EVENT | action=%s 无 event_id（去重失效风险）", action)

        _log_telemetry(result)
        text = str(result.get("text") or "")
        if action == "analyze" and text:
            if lead_in_busy_now:
                # 当前轮次仍在处理：不丢弃，排队等本轮结束自动续跑
                st.session_state["_ghost_pending_analyze"] = text
                _log.info(
                    "GHOST_QUEUE | analyze 排队等待本轮结束 event_id=%s text=%s",
                    event_id,
                    text[:80],
                )
            else:
                st.session_state["_ghost_analyze_text"] = text
                _log.info(
                    "GHOST_ANALYZE | text_len=%d event_id=%s text=%s",
                    len(text),
                    event_id,
                    text[:120],
                )
        return text
    return ""


def _log_telemetry(result: dict) -> None:
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
    if total > 0:
        st.session_state[_GHOST_ERROR_TRACKER_KEY] = {"ok": ok, "fail": fail}
