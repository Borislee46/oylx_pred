import logging
import time
from typing import Any

from src.agent.context import StudentContext
from src.agent.tools.form_gateway import compute_missing_required

_log = logging.getLogger("LeadInDispatcher")


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
            "duration_ms": e.duration_ms,
        }
        for e in getattr(trace, "entries", [])
    ]


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


def should_auto_predict(ctx: StudentContext, *, had_api_error: bool = False) -> bool:
    if had_api_error:
        return False
    bg = ctx.extracted_background or {}
    return not compute_missing_required(bg)
