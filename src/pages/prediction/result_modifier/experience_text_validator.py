from __future__ import annotations

import random
import re

from src.agent import TextPreprocessingAgent
from src.pages.prediction.config.ui_messages import (
    EXPERIENCE_ANALYSIS_MESSAGES,
    EXPERIENCE_VALIDATION_TEMPLATE,
    FIELD_NAME_MAP,
)
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.result_modifier.streamlit_cache import cache_resource
from src.pages.prediction.result_modifier.utils import (
    has_streamlit_runtime,
    is_effectively_empty,
)
from src.utils.env_config_loader import load_app_config


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
        from src.pages.prediction.result_modifier.ui_handler import LoadingMessageAnimator

        animator = LoadingMessageAnimator(progress_reporter=progress_reporter)
        field_names = [FIELD_NAME_MAP.get(k, k) for k, _ in fields_to_validate]
        animator.show(_get_analysis_message(field_names), force=True)
    else:
        st = _get_streamlit()
        if st is not None and has_streamlit_runtime():
            from src.pages.prediction.result_modifier.ui_handler import LoadingMessageAnimator

            animator = LoadingMessageAnimator()
            field_names = [FIELD_NAME_MAP.get(k, k) for k, _ in fields_to_validate]
            animator.show(_get_analysis_message(field_names), force=True)

    validated_keys: list[str] = []
    if llm_enabled:
        fields_dict = dict(fields_to_validate)
        if animator is not None:
            animator.show("正在核验软背景信息（批量LLM校验）", force=True)

        agent = _get_text_agent()
        batch_result = agent.validate_fields_batch(fields_dict)

        for k, _ in fields_to_validate:
            is_valid = batch_result.get(k, False)
            if is_valid:
                validated_keys.append(k)
            else:
                st = _get_streamlit()
                if st is not None and has_streamlit_runtime():
                    st.toast(f"{FIELD_NAME_MAP.get(k, k)}填写的内容无效")
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
            stripped = content.strip()
            if stripped and len(stripped) >= 6:
                alpha_chars = sum(1 for c in stripped if "a" <= c.lower() <= "z")
                cjk_chars = sum(1 for c in stripped if "一" <= c <= "鿿")
                if alpha_chars + cjk_chars >= 4:
                    validated_keys.append(k)

    if animator is not None:
        animator.clear()

    if not validated_keys:
        return False

    merged = " ".join(str(experience_details.get(k, "")) for k in validated_keys)
    merged = merged.strip().lower()
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", merged)
    return len(cleaned) >= 3
