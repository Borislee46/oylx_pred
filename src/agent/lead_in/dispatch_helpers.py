import logging
import re
import time
from typing import Any

from src.agent.context import StudentContext
from src.agent.field_labels import FIELD_LABEL
from src.agent.lead_in.session_continuity import build_form_snapshot
from src.agent.tools.form_gateway import compute_missing_required

_log = logging.getLogger("LeadInDispatcher")


def _build_router_history(ctx: StudentContext, session_manager: Any | None = None) -> str:
    if getattr(ctx, "session_continuity", "continue") == "fresh":
        return ""
    parts: list[str] = []
    snapshot = build_form_snapshot(session_manager)
    if snapshot and snapshot != "（空）":
        parts.append(f"## 当前表单快照\n{snapshot}")
    turns = ctx.conversation_turns or []
    if turns:
        parts.append("## 对话历史")
        for t in turns[-6:]:
            role = "顾问" if t.get("role") == "user" else "AI"
            content = t.get("content") or t.get("summary", "")
            if content and str(content).strip():
                parts.append(f"- {role}：{str(content)[:200]}")
    if not parts:
        return ""
    parts.append(
        "## 会话说明\n"
        "上文为同一学生的已有表单与对话。仅补充/调整当前输入中的新信息；"
        "不要从对话历史回填已被用户新输入取代的背景字段。"
    )
    return "\n\n".join(parts)


def _serialize_trace(deps) -> list[dict[str, Any]]:
    trace = getattr(deps, "trace", None)
    if trace is None:
        return []
    return [
        {
            "seq": e.seq,
            "tool": e.tool,
            "args_preview": e.args_preview,
            "result_preview": e.result_preview,
            "ts": e.ts,
            "ok": e.ok,
        }
        for e in getattr(trace, "entries", [])
    ]


def _detail_lines_as_trace(details: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "seq": i,
            "tool": "router_stage",
            "args_preview": {},
            "result_preview": line,
            "ts": 0.0,
            "ok": True,
        }
        for i, line in enumerate(details)
    ]


_SUMMARY_KEYS = ("university", "major", "gpa", "language_score", "country", "target_schools")


def _summarize_extracted_fields(extracted: dict[str, Any]) -> str:
    shown = [FIELD_LABEL[k] for k in _SUMMARY_KEYS if extracted.get(k) not in (None, "", [])]
    if shown:
        return "识别到：" + "、".join(shown[:8])
    return f"提取到 {len(extracted)} 个字段"


def _append_conversation_turn(
    ctx: StudentContext,
    role: str,
    content: str,
    *,
    sm: Any = None,
) -> None:
    if not content or not content.strip():
        return
    text = content.strip()
    turns = ctx.conversation_turns or []
    if turns:
        last = turns[-1]
        if last.get("role") == role and str(last.get("content", "")).strip() == text:
            return
    turns.append(
        {
            "role": role,
            "content": text,
            "ts": time.time(),
        }
    )
    if len(turns) > 20:
        ctx.conversation_turns = turns[-20:]
    else:
        ctx.conversation_turns = turns

    if sm is not None:
        try:
            sm.update(conversation_turns=list(ctx.conversation_turns))
        except Exception:
            _log.debug(
                "DISPATCH | sm.update(conversation_turns) failed (non-critical)",
                exc_info=True,
            )


def _looks_structured(text: str) -> bool:
    t = text.strip()
    if len(t) > 150:
        return False
    separators = t.count("，") + t.count(",") + t.count("；") + t.count(";")
    if separators < 2:
        return False
    if not re.search(r"\d+\.?\d*", t):
        return False
    return True


_PROFILE_HINT = (
    r"(大学|学院|学校|专业|GPA|gpa|均分|绩点|雅思|托福|IELTS|TOEFL|"
    r"University|College|本科|硕士|读研|申请)"
)


def _looks_extractable_profile(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 8:
        return False
    if not re.search(_PROFILE_HINT, t, re.IGNORECASE):
        return False
    if re.search(r"\d+\.?\d*", t):
        return True
    has_school = bool(re.search(r"(大学|学院|University|College)", t, re.IGNORECASE))
    has_major = "专业" in t or bool(re.search(r"major", t, re.IGNORECASE))
    return has_school and has_major


def should_auto_predict(ctx: StudentContext, *, had_api_error: bool = False) -> bool:
    if had_api_error:
        return False
    bg = ctx.extracted_background or {}
    return not compute_missing_required(bg)
