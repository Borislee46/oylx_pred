import logging
from typing import Any, Protocol

import streamlit as st

from src.agent.bridges.fuzzy_match import _load_target_major_options
from src.agent.context import StudentContext
from src.pages.prediction.handler_config import DEFAULT_FORM_KEYS

_log = logging.getLogger(__name__)


REQUIRED_FIELDS: dict[str, str] = {
    "university": "院校",
    "major": "专业",
    "gpa": "GPA",
    "language_score": "语言成绩",
}


class FormGateway(Protocol):
    def read(self) -> dict: ...

    def options(self, field: str) -> list[str]: ...

    def write(self, fields: dict) -> dict: ...

    def submit(self) -> dict: ...

    def expand(self) -> None: ...


def compute_missing_required(bg: dict) -> list[str]:
    missing: list[str] = []
    if not bg.get("university"):
        missing.append(REQUIRED_FIELDS["university"])
    if not bg.get("major"):
        missing.append(REQUIRED_FIELDS["major"])
    gpa_val = bg.get("gpa")
    lang_val = bg.get("language_score")
    if not (isinstance(gpa_val, (int, float)) and gpa_val > 0):
        missing.append(REQUIRED_FIELDS["gpa"])
    if not (isinstance(lang_val, (int, float)) and lang_val > 0):
        missing.append(REQUIRED_FIELDS["language_score"])
    _log.debug("compute_missing_required | missing=%s", missing)
    return missing


class StreamlitFormGateway:
    def __init__(self, session_manager: Any, ctx: StudentContext) -> None:
        self._sm = session_manager
        self._ctx = ctx
        self._uni_opts = self._snapshot_options("university")
        self._major_opts = self._snapshot_options("major")
        self._target_major_opts = _load_target_major_options()
        _log.debug(
            "StreamlitFormGateway init | uni_opts=%d major_opts=%d target_major_opts=%d",
            len(self._uni_opts),
            len(self._major_opts),
            len(self._target_major_opts),
        )
        self._submit_requested = False
        self._expand_requested = False
        self._applied: dict[str, Any] = {}
        self._dropped_schools: list = []
        self._low_confidence: list = []
        self._university_alias: str | None = None
        self._university_display_label: str | None = None

    def _snapshot_options(self, field: str) -> list[str]:
        if field == "university":
            cache = self._sm.get(DEFAULT_FORM_KEYS.background_universities_cache) or {}
            cached = list(cache.keys()) if isinstance(cache, dict) else list(cache)
            return cached
        cache = self._sm.get(DEFAULT_FORM_KEYS.background_majors_cache, {}) or {}
        cached = list(cache.get("majors_display", []))
        return cached

    def read(self) -> dict:
        bg = self._ctx.extracted_background or {}
        present = {k: v for k, v in bg.items() if v not in (None, "", [])}
        result = {"present": present, "missing": compute_missing_required(bg)}
        _log.debug("gateway.read | present=%s missing=%s", list(present.keys()), result["missing"])
        return result

    def options(self, field: str) -> list[str]:
        f = (field or "").strip().lower()
        if f in ("university", "background_university", "院校"):
            return self._uni_opts[:50]
        if f in ("major", "background_major", "专业"):
            return self._major_opts[:50]
        if f in ("target_major", "target_majors", "目标专业"):
            return self._target_major_opts[:50]
        return []

    def write(self, fields: dict) -> dict:
        clean = {
            k: v
            for k, v in (fields or {}).items()
            if v not in (None, "", [], 0, 0.0) and not (isinstance(v, str) and v.strip() == "")
        }
        if clean:
            _log.debug("gateway.write | fields=%s", list(clean.keys()))
            self._ctx.extracted_background = {**(self._ctx.extracted_background or {}), **clean}
        return {
            "applied": list(clean.keys()),
            "missing": compute_missing_required(self._ctx.extracted_background or {}),
        }

    def submit(self) -> dict:
        missing = compute_missing_required(self._ctx.extracted_background or {})
        if missing:
            _log.info("gateway.submit | blocked: missing=%s", missing)
            return {"ok": False, "missing": missing}
        self._submit_requested = True
        _log.info("gateway.submit | ready, deferring to flush()")
        return {"ok": True, "missing": []}

    def expand(self) -> None:
        _log.debug("gateway.expand | requested")
        self._expand_requested = True

    def flush(self) -> None:
        if self._ctx.extracted_background:
            from src.pages.prediction.form_bridge import (  # lazy — circular import guard
                apply_lead_in_to_form,
            )

            applied = apply_lead_in_to_form(self._ctx, self._sm)
            self._applied = {k: v for k, v in applied.items() if not k.startswith("_")}
            self._dropped_schools = applied.get("_dropped_schools", [])
            self._low_confidence = list(
                (self._sm.get("lead_in_low_confidence_fields", {}) or {}).keys()
            )
            self._university_alias = self._sm.get("background_university_alias")
            self._university_display_label = self._sm.get("background_university_display_label")
            _log.info(
                "gateway.flush | applied=%d fields dropped_schools=%s low_conf=%s submit=%s expand=%s",
                len(self._applied),
                self._dropped_schools,
                self._low_confidence,
                self._submit_requested,
                self._expand_requested,
            )
        if self._submit_requested:
            self._sm.set(_lead_in_processed=True)
        if self._expand_requested:
            st.session_state["form_expander"] = True

    def applied_fields(self) -> dict:
        result = dict(self._applied)
        if self._university_display_label:
            result["_university_display_label"] = self._university_display_label
        if self._university_alias:
            result["_university_alias"] = self._university_alias
        return result
