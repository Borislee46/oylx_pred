from __future__ import annotations

import random
import re

from src.adjustment.utils import (
    cache_resource,
    has_streamlit_runtime,
    is_effectively_empty,
)
from src.agent import TextPreprocessingAgent
from src.pages.prediction.core.ui_messages import (
    EXPERIENCE_ANALYSIS_MESSAGES,
    EXPERIENCE_VALIDATION_TEMPLATE,
    FIELD_NAME_MAP,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _get_streamlit():
    import streamlit as st

    return st


@cache_resource
def _get_text_agent():
    return TextPreprocessingAgent()


def _get_analysis_message(field_names: list[str]) -> str:
    if not field_names:
        return "正在核验软背景信息有效性"
    if len(field_names) == 1:
        msg = random.choice(EXPERIENCE_ANALYSIS_MESSAGES)
        return msg.format(field=field_names[0]) if "{field}" in msg else msg
    fields_text = "、".join(field_names)
    return f"正在核验软背景：{fields_text}（信息抽取与有效性检查）"


def _build_validation_status_message(
    *,
    field_name: str,
    idx: int,
    total: int,
    content_len: int,
    llm_enabled: bool,
) -> str:
    method = "LLM校验" if llm_enabled else "本地规则"
    return random.choice(EXPERIENCE_VALIDATION_TEMPLATE).format(
        idx=idx, total=total, field_name=field_name, length=content_len, method=method
    )


# ──  Fast-path thresholds for skipping LLM validation ──────────────────
# Fields with ≥MIN_LEN chars and ≥MIN_MEANINGFUL alpha/CJK chars are almost
# certainly real content — placeholder/template text is rarely this long.
# Skipping the LLM for these saves ~2-5s per prediction on the common case.
_FASTPATH_MIN_LEN = 20
_FASTPATH_MIN_MEANINGFUL = 10


def _check_local_rules(
    content: str,
    min_len: int = 6,
    min_meaningful: int = 4,
) -> bool:
    """Check if *content* passes basic local validation rules (no LLM).

    Returns True when the text has enough characters and enough alphabetic
    or CJK characters to be considered meaningful content rather than a
    placeholder or gibberish.
    """
    stripped = content.strip()
    if not stripped or len(stripped) < min_len:
        return False
    alpha_chars = sum(1 for c in stripped if "a" <= c.lower() <= "z")
    cjk_chars = sum(1 for c in stripped if "一" <= c <= "鿿")
    return (alpha_chars + cjk_chars) >= min_meaningful


def has_meaningful_experience_text(
    experience_details: dict[str, str] | None,
    *,
    progress_reporter: ProgressReporter | None = None,
) -> bool:
    if not experience_details:
        return False

    keys = ("research_details", "award_details", "internship_details", "paper_details")

    fields_to_validate: list[tuple[str, str]] = []
    for k in keys:
        content = str(experience_details.get(k, "")).strip()
        if not is_effectively_empty(content):
            fields_to_validate.append((k, content))

    if not fields_to_validate:
        return False

    app_config = load_app_config()
    llm_enabled = bool(app_config.get("OPEN_AI_BASE_URL") and app_config.get("OPEN_AI_API_KEY"))

    animator = None
    if progress_reporter is not None:
        from src.adjustment.ui_handler import LoadingMessageAnimator

        animator = LoadingMessageAnimator(progress_reporter=progress_reporter)
        field_names = [FIELD_NAME_MAP.get(k, k) for k, _ in fields_to_validate]
        animator.show(_get_analysis_message(field_names), force=True)
    else:
        st = _get_streamlit()
        if st is not None and has_streamlit_runtime():
            from src.adjustment.ui_handler import LoadingMessageAnimator

            animator = LoadingMessageAnimator()
            field_names = [FIELD_NAME_MAP.get(k, k) for k, _ in fields_to_validate]
            animator.show(_get_analysis_message(field_names), force=True)

    validated_keys: list[str] = []
    if llm_enabled:
        fast_pass_keys: list[str] = []
        needs_llm: dict[str, str] = {}
        for k, content in fields_to_validate:
            if _check_local_rules(content, min_len=20, min_meaningful=10):
                fast_pass_keys.append(k)
            elif _check_local_rules(content):  # basic threshold (6/4)
                needs_llm[k] = content
            # else: fails even basic rules — don't bother with LLM

        validated_keys.extend(fast_pass_keys)

        if needs_llm:
            if animator is not None:
                animator.show("正在核验软背景信息（批量LLM校验）", force=True)

            agent = _get_text_agent()
            batch_result = agent.validate_fields_batch(needs_llm)

            for k in needs_llm:
                is_valid = batch_result.get(k, False)
                if is_valid:
                    validated_keys.append(k)
                else:
                    st = _get_streamlit()
                    if st is not None and has_streamlit_runtime():
                        st.toast(f"{FIELD_NAME_MAP.get(k, k)}填写的内容无效")
        else:
            # All fields fast-passed — clear the animator early
            if animator is not None:
                animator.clear()

        logger.info(
            "经历文本校验完成 | fast_pass=%d llm=%d validated=%d",
            len(fast_pass_keys),
            len(needs_llm),
            len(validated_keys),
        )
    else:
        for k, content in fields_to_validate:
            field_name = FIELD_NAME_MAP.get(k, k)
            if animator is not None:
                msg = _build_validation_status_message(
                    field_name=field_name,
                    idx=1,
                    total=1,
                    content_len=len(content),
                    llm_enabled=False,
                )
                animator.show(msg, force=True)
            if _check_local_rules(content):
                validated_keys.append(k)

    if animator is not None:
        animator.clear()

    if not validated_keys:
        return False

    merged = " ".join(str(experience_details.get(k, "")) for k in validated_keys)
    merged = merged.strip().lower()
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", merged)
    is_valid = len(cleaned) >= 3
    logger.info(
        "\u7ecf\u5386\u6587\u672c\u6821\u9a8c\u5b8c\u6210 | validated_keys=%s llm=%s is_valid=%s cleaned_len=%d",
        validated_keys,
        llm_enabled,
        is_valid,
        len(cleaned),
    )
    return is_valid
