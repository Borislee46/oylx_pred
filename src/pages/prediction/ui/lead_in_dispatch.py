from __future__ import annotations

import sys
import time
from collections.abc import Callable
from typing import Any

import streamlit as st

from src.agent.context import StudentContext
from src.agent.lead_in.dispatch_constants import MAX_RETRIES, LeadInCancelled
from src.agent.lead_in.dispatcher import (
    DISMISS_KEY,
    IN_PROGRESS_KEY,
    PENDING_KEY,
    PROGRESS_STEPS_KEY,
    PROGRESS_TEXT_KEY,
    PROGRESS_VARIANT_KEY,
    RETRY_COUNT_KEY,
    RUNNING_HASH_KEY,
    RUNNING_TS_KEY,
    LeadInDispatcher,
)
from src.agent.lead_in.state_machine import LeadInTurnStateMachine
from src.pages.prediction.ui.lead_in_echo import (
    path_narrative,
    sanitize_feedback,
    strip_emoji,
)
from src.pages.prediction.ui.lead_in_progress_copy import (
    PIPELINE,
    humanize_detail,
)
from src.pages.prediction.ui.lead_in_wait_sse import (
    close_run,
    ensure_sse_server,
    new_run_id,
    open_run,
    same_origin_sse_enabled,
)
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_LAST_APPLIED_KEY = "_lead_in_last_applied"
_LEAD_IN_RUNNING_HASH = RUNNING_HASH_KEY
_LEAD_IN_RUNNING_TS = RUNNING_TS_KEY
_LEAD_IN_RETRY_COUNT = RETRY_COUNT_KEY
_LEAD_IN_PROGRESS_STEPS = PROGRESS_STEPS_KEY
_LEAD_IN_PROGRESS_TEXT = PROGRESS_TEXT_KEY
_PENDING_ANALYZE_KEY = PENDING_KEY
_IN_PROGRESS_KEY = IN_PROGRESS_KEY
_DISMISS_KEY = DISMISS_KEY
_SSE_PORT_KEY = "_lead_in_sse_port"
_SSE_RUN_ID_KEY = "_lead_in_sse_run_id"
_SSE_URL_KEY = "_lead_in_sse_url"
_PENDING_ANALYZE_KEY = "_ghost_pending_analyze"

_STAGE_SHORT = {
    "解析学生背景": "读背景",
    "结构化提取": "提取",
    "写入表单": "填表",
    "完成": "完成",
}


def _make_progress_cb(
    stream_box: Any,
    dispatcher: LeadInDispatcher,
    sm: LeadInTurnStateMachine,
    *,
    sse_stream: Any = None,
    sse_port: int = 0,
    sse_run_id: str = "",
    sse_url: str = "",
) -> Callable[..., None]:
    saw_partial = False

    def _on_progress(
        steps: list[str],
        partial: str,
        details: list[str] | None = None,
    ) -> None:
        nonlocal saw_partial
        from src.pages.prediction.ui.lead_in_wait import render_lead_in_wait

        variant = getattr(dispatcher, "_progress_variant", "default")
        detail_lines = list(details or [])
        path_hint = str(getattr(dispatcher, "_progress_path_hint", "") or "")

        if sse_stream is not None and getattr(sse_stream, "cancel_requested", False):
            dispatcher._cancelled = True
            raise LeadInCancelled()

        if sse_stream is not None:
            partial_clean = (partial or "").strip()
            if partial_clean:
                saw_partial = True
                sse_stream.publish_text(partial_clean)
            elif not saw_partial and detail_lines:
                sse_stream.publish_text(humanize_detail(detail_lines[-1]))

        st.session_state[_LEAD_IN_PROGRESS_STEPS] = steps
        st.session_state[_LEAD_IN_PROGRESS_TEXT] = partial
        st.session_state[PROGRESS_VARIANT_KEY] = variant
        st.session_state["_lead_in_progress_details"] = detail_lines
        st.session_state["_lead_in_path_hint"] = path_hint

        retry_val = st.session_state.get(_LEAD_IN_RETRY_COUNT, 0)
        elapsed_val = time.time() - st.session_state.get(_LEAD_IN_RUNNING_TS, time.time())

        sm.update(
            progress_steps=steps,
            progress_text=partial,
            progress_variant=variant,
            progress_details=detail_lines,
        )

        from src.pages.prediction.ui.lead_in_progress_copy import stage_index as _stage_idx

        cur_idx = _stage_idx(steps)
        detail_sig = "\0".join(detail_lines[-6:])
        last_idx = st.session_state.get("_lead_in_wait_ui_stage")
        last_sig = st.session_state.get("_lead_in_wait_ui_sig")
        last_ts = float(st.session_state.get("_lead_in_wait_ui_ts") or 0.0)
        last_retry = int(st.session_state.get("_lead_in_wait_ui_retry") or -1)
        now = time.time()

        if sse_stream is not None:
            # SSE 通道可用：只挂载一次，stage/明细通过 SSE meta 原地更新，
            # 避免每个进度 tick 销毁重建 iframe（WebGL 重建 / SSE 重连 / 动效重启）。
            name = PIPELINE[min(cur_idx, len(PIPELINE) - 1)]
            feed: list[str] = []
            for raw in detail_lines:
                line = humanize_detail(raw)
                if line and (not feed or feed[-1] != line):
                    feed.append(line)
            sse_stream.publish_meta(
                stage=_STAGE_SHORT.get(name, name),
                stage_index=cur_idx,
                stage_count=len(PIPELINE),
                details=feed[-6:],
                retry=retry_val,
                retry_max=MAX_RETRIES,
            )
            if last_idx is None:
                st.session_state["_lead_in_wait_ui_stage"] = cur_idx
                st.session_state["_lead_in_wait_ui_sig"] = detail_sig
                st.session_state["_lead_in_wait_ui_ts"] = now
                st.session_state["_lead_in_wait_ui_retry"] = retry_val
                with stream_box:
                    render_lead_in_wait(
                        steps,
                        partial,
                        elapsed=elapsed_val,
                        retry=retry_val,
                        variant=variant,
                        details=detail_lines,
                        path_hint=path_hint,
                        ctx=st.session_state.get("lead_in_ctx"),
                        applied=st.session_state.get("_lead_in_last_applied") or {},
                        sse_port=sse_port,
                        sse_run_id=sse_run_id,
                        sse_url=sse_url,
                        key="lead_in_wait_live",
                    )
            return

        should_mount = (
            last_idx is None
            or cur_idx != last_idx
            or retry_val != last_retry
            or detail_sig != last_sig
            or (now - last_ts) >= 2.0
        )
        if not should_mount:
            return

        st.session_state["_lead_in_wait_ui_stage"] = cur_idx
        st.session_state["_lead_in_wait_ui_sig"] = detail_sig
        st.session_state["_lead_in_wait_ui_ts"] = now
        st.session_state["_lead_in_wait_ui_retry"] = retry_val

        tick = int(st.session_state.get("_lead_in_wait_tick", 0)) + 1
        st.session_state["_lead_in_wait_tick"] = tick
        stream_box.empty()
        with stream_box:
            render_lead_in_wait(
                steps,
                partial,
                elapsed=elapsed_val,
                retry=retry_val,
                variant=variant,
                details=detail_lines,
                path_hint=path_hint,
                ctx=st.session_state.get("lead_in_ctx"),
                applied=st.session_state.get("_lead_in_last_applied") or {},
                sse_port=sse_port,
                sse_run_id=sse_run_id,
                sse_url=sse_url,
                key=f"lead_in_wait_live_{tick}",
            )

    return _on_progress


def _save_display(
    ctx: StudentContext,
    feedback: str,
    applied: dict[str, Any],
    low_conf_fields: dict[str, Any] | None = None,
    clarifying_questions: list[str] | None = None,
    *,
    sm: LeadInTurnStateMachine | None = None,
) -> None:
    feedback = sanitize_feedback(feedback or "")
    st.session_state[_DISMISS_KEY] = False
    if feedback:
        ctx.quick_assessment = feedback
        st.session_state["lead_in_ctx"] = ctx
    if applied:
        cleaned = {k: v for k, v in applied.items() if not str(k).startswith("_")}
        st.session_state[_LAST_APPLIED_KEY] = cleaned
    if low_conf_fields:
        st.session_state["_lead_in_low_conf_display"] = low_conf_fields
    else:
        st.session_state.pop("_lead_in_low_conf_display", None)
    if clarifying_questions:
        st.session_state["_lead_in_clarify_questions"] = clarifying_questions
    else:
        st.session_state.pop("_lead_in_clarify_questions", None)

    if sm is not None:
        cleaned = (
            {k: v for k, v in applied.items() if not str(k).startswith("_")} if applied else {}
        )
        sm.update(
            feedback_dismissed=False,
            last_applied_fields=cleaned,
            low_confidence_display=low_conf_fields or {},
            clarifying_questions=clarifying_questions or [],
        )


def _cleanup_display_state() -> None:
    for k in (
        _LEAD_IN_PROGRESS_STEPS,
        _LEAD_IN_PROGRESS_TEXT,
        PROGRESS_VARIANT_KEY,
        "_lead_in_path_hint",
        "_lead_in_path_kind",
        "_lead_in_source_text",
        "_lead_in_wait_ui_stage",
        "_lead_in_wait_ui_sig",
        "_lead_in_wait_ui_ts",
        "_lead_in_wait_ui_retry",
        "_lead_in_wait_tick",
        _SSE_PORT_KEY,
        _SSE_RUN_ID_KEY,
        _SSE_URL_KEY,
    ):
        st.session_state.pop(k, None)


def _requeue_pending_analyze_if_any() -> None:
    """本轮结束后，若有排队中的 analyze，转正并触发 rerun 续跑。"""
    pending = st.session_state.pop(_PENDING_ANALYZE_KEY, None)
    if not (pending and str(pending).strip()):
        return
    if sys.exc_info()[0] is not None:
        # 当前轮次异常中断：保留排队，待下次 rerun 再处理
        st.session_state[_PENDING_ANALYZE_KEY] = pending
        return
    st.session_state["_ghost_analyze_text"] = str(pending)
    logger.info("GHOST_REQUEUE | 排队 analyze 已就绪，触发续跑 text=%s", str(pending)[:80])
    st.rerun(scope="app")


def run_lead_in_dispatch(session_manager: Any) -> None:
    ctx: StudentContext = st.session_state.get("lead_in_ctx", StudentContext())
    sm = LeadInTurnStateMachine(session_manager)

    _use_harness = False
    try:
        from src.agent.harness import load_harness_config

        _use_harness = load_harness_config().enabled
    except Exception:
        pass

    dispatcher = LeadInDispatcher(state_machine=sm)

    new_text = st.session_state.pop("_ghost_analyze_text", None)
    dispatcher.enqueue(new_text, st.session_state)

    pending = dispatcher.dequeue(st.session_state)
    if not pending:
        if sm.is_busy() or st.session_state.get(_IN_PROGRESS_KEY):
            logger.warning("LEAD_IN_DISPATCH | busy 但无 pending，重置")
            dispatcher.mark_done(st.session_state)
        return

    if sm.is_busy() or st.session_state.get(_IN_PROGRESS_KEY, False):
        if dispatcher.is_recovering(st.session_state):
            if not dispatcher.should_retry(st.session_state):
                return
        else:
            st.session_state[_LEAD_IN_RETRY_COUNT] = 0
            sm.update(retry_count=0)

    if not new_text and not sm.is_busy() and not st.session_state.get(_IN_PROGRESS_KEY):
        return

    analyze_text = str(pending).strip()
    dispatcher.mark_running(st.session_state, analyze_text)
    st.session_state["_lead_in_source_text"] = analyze_text
    st.session_state["_lead_in_path_kind"] = "default"
    ctx.raw_input = analyze_text or ctx.raw_input
    st.session_state["lead_in_ctx"] = ctx

    sse_port = 0
    sse_run_id = ""
    sse_stream = None
    sse_url = ""
    try:
        sse_port = ensure_sse_server()
        sse_run_id = new_run_id()
        sse_stream = open_run(sse_run_id)
        if same_origin_sse_enabled():
            sse_url = f"/sse/{sse_run_id}"
            sse_port = 0
        st.session_state[_SSE_PORT_KEY] = sse_port
        st.session_state[_SSE_RUN_ID_KEY] = sse_run_id
        st.session_state[_SSE_URL_KEY] = sse_url
    except Exception:
        logger.warning(
            "LEAD_IN_DISPATCH | SSE 旁路启动失败，降级为无流式等待",
            exc_info=True,
        )
        sse_port = 0
        sse_run_id = ""
        sse_stream = None
        sse_url = ""

    logger.info(
        "LEAD_IN_DISPATCH | 开始处理 | text_hash=%s text=%s harness=%s",
        st.session_state.get(_LEAD_IN_RUNNING_HASH),
        analyze_text[:80],
        _use_harness,
    )
    try:
        from src.utils.analytics import track as _t

        _t("lead_in_start", text_length=len(analyze_text))
    except Exception:
        pass

    st.markdown(
        "<script>"
        "setTimeout(function(){"
        'var el=document.getElementById("hk-form-anchor");'
        'if(el)el.scrollIntoView({behavior:"smooth",block:"start"});'
        "},50);"
        "</script>",
        unsafe_allow_html=True,
    )

    from src.pages.prediction.ui.lead_in_wait import _ensure_iframe_transparent

    _ensure_iframe_transparent()
    stream_box = st.empty()
    dispatcher._cancelled = False
    progress_cb = _make_progress_cb(
        stream_box,
        dispatcher,
        sm,
        sse_stream=sse_stream,
        sse_port=sse_port,
        sse_run_id=sse_run_id,
        sse_url=sse_url,
    )
    dispatcher.on_progress = progress_cb
    dispatcher._progress_path_hint = path_narrative("default")
    progress_cb(["解析学生背景"], "", ["已收到输入，准备结构化提取"])

    completed_ok = False
    try:
        if _use_harness:
            from src.agent.harness import ConsultationHarness

            harness = ConsultationHarness()
            result = harness.run_lead_in_turn(
                session_manager, ctx, analyze_text, on_progress=progress_cb
            )
        else:
            result = dispatcher.dispatch(
                session_manager, ctx, analyze_text, session_state=st.session_state
            )
        completed_ok = bool(getattr(result, "handled", False))
    except LeadInCancelled:
        dispatcher.mark_done(st.session_state)
        st.session_state.pop(_PENDING_ANALYZE_KEY, None)
        ctx.quick_assessment = strip_emoji("已取消自动提取，请手动填写下方表单")
        st.session_state["lead_in_ctx"] = ctx
        _cleanup_display_state()
        logger.info("LEAD_IN_DISPATCH | 用户取消提取")
        return
    except Exception:
        ctx.quick_assessment = strip_emoji("AI 提取异常，请手动填写表单或稍后重试")
        st.session_state["lead_in_ctx"] = ctx
        logger.warning(
            "LEAD_IN_DISPATCH | 异常中断，保留 pending 供重试 | len=%d", len(analyze_text)
        )
        raise
    finally:
        close_run(sse_run_id, ok=completed_ok)
        if completed_ok and sse_stream is not None:
            time.sleep(1.2)
        stream_box.empty()
        _requeue_pending_analyze_if_any()

    if result.handled:
        _save_display(
            ctx,
            result.feedback,
            result.applied_fields,
            result.low_confidence_fields,
            result.clarifying_questions,
            sm=sm,
        )
        st.session_state["lead_in_ctx"] = ctx
        if result.should_expand_form:
            st.session_state["form_expander"] = True
            sm.update(form_expander_open=True)
        dispatcher.mark_done(st.session_state)
        st.session_state.pop(_PENDING_ANALYZE_KEY, None)
        _cleanup_display_state()
        logger.info(
            "LEAD_IN_DISPATCH | 完成 path=%s fields=%d", result.path, len(result.applied_fields)
        )
        if getattr(result, "validation_issues", None):
            logger.warning("LEAD_IN_DISPATCH | validation issues=%s", result.validation_issues)
    else:
        logger.error("LEAD_IN_DISPATCH | 全部路径失败 error=%s", result.error)
        dispatcher.mark_done(st.session_state)
        st.session_state.pop(_PENDING_ANALYZE_KEY, None)
        _cleanup_display_state()
        err = str(result.error or "")
        if "timeout" in err:
            user_msg = "处理超时，请稍后重试"
        elif "consistency" in err or "flush" in err:
            user_msg = "系统内部状态异常，请刷新页面重试"
        elif "cooldown" in err:
            user_msg = result.feedback or "上一次请求仍在处理中，请稍后重试"
        else:
            user_msg = "AI 提取失败，请手动填写表单或稍后重试"
        ctx.quick_assessment = strip_emoji(user_msg)
        st.session_state["lead_in_ctx"] = ctx
        try:
            from src.utils.analytics import track as _t

            _t("lead_in_all_paths_failed", error=str(result.error or "unknown"))
        except Exception:
            pass
