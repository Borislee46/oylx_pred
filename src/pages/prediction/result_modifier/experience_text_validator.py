from __future__ import annotations

import random
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from src.agent.text_preprocessing_prompts import build_field_validation_prompt
from src.pages.prediction.flow.progress_reporter import ProgressReporter
from src.pages.prediction.result_modifier.streamlit_cache import cache_data
from src.pages.prediction.result_modifier.utils import (
    EXPERIENCE_ANALYSIS_MESSAGES,
    FIELD_NAME_MAP,
    generate_content_hash,
    is_effectively_empty,
)
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _get_streamlit():
    try:
        import streamlit as st

        return st
    except ImportError:
        return None


def _has_streamlit_runtime(st) -> bool:
    runtime = getattr(st, "runtime", None)
    exists = getattr(runtime, "exists", None)
    return bool(callable(exists) and exists())


@cache_data(ttl=3600, show_spinner=False)
def _validate_field_with_llm_cached(field_type: str, content_hash: str, content: str) -> bool:
    if is_effectively_empty(content):
        return False

    app_config = load_app_config()
    api_url = app_config.get("OPEN_AI_BASE_URL")
    api_key = app_config.get("OPEN_AI_API_KEY")
    model = app_config.get("OPEN_AI_MODEL", "deepseek-v3.1")

    if not api_url or not api_key:
        logger.warning("LLM验证配置缺失，跳过验证")
        return True

    field_name = FIELD_NAME_MAP.get(field_type, field_type)
    logger.info(f"开始调用LLM验证字段: {field_name}, 内容哈希: {content_hash[:8]}...")

    prompt = build_field_validation_prompt(field_type, content)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    data = {
        "model": model,
        "messages": [{"content": [{"text": prompt, "type": "text"}], "role": "user"}],
        "thinking": {"type": "disabled"},
    }

    start_time = time.time()
    try:
        logger.debug(f"发送LLM请求: model={model}, url={api_url}")
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        elapsed_time = time.time() - start_time

        response.raise_for_status()
        response_json = response.json()

        if response_json.get("choices"):
            result = response_json["choices"][0].get("message", {}).get("content", "").strip()
            is_valid = result == "是"
            logger.info(
                f"LLM验证完成: {field_name}, 结果={is_valid}, 响应内容={result}, 耗时={elapsed_time:.2f}秒"
            )
            return is_valid

        error_info = response_json.get("error", {})
        logger.warning(
            f"LLM验证返回异常: {field_name}, 错误信息={error_info}, 耗时={elapsed_time:.2f}秒"
        )
        return True

    except (requests.exceptions.RequestException, ValueError, TypeError, KeyError) as e:
        elapsed_time = time.time() - start_time
        logger.warning(
            f"LLM验证失败: {field_name}, 异常={str(e)}, 耗时={elapsed_time:.2f}秒, 使用默认通过"
        )
        return True


def _validate_field_with_llm(field_type: str, content: str) -> bool:
    if is_effectively_empty(content):
        return False

    content_hash = generate_content_hash(f"{field_type}:{content}")
    return _validate_field_with_llm_cached(field_type, content_hash, content)


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
    llm_text = "LLM校验" if llm_enabled else "本地规则"
    safe_total = max(1, int(total))
    safe_idx = max(1, min(int(idx), safe_total))
    return f"软背景校验 {safe_idx}/{safe_total}：{field_name}（{content_len} 字｜{llm_text}）"


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
        if st is not None and _has_streamlit_runtime(st):
            from src.pages.prediction.result_modifier.ui_handler import LoadingMessageAnimator

            animator = LoadingMessageAnimator()
            field_names = [FIELD_NAME_MAP.get(k, k) for k, _ in fields_to_validate]
            animator.show(_get_analysis_message(field_names), force=True)

    validated_keys: list[str] = []
    total = len(fields_to_validate)
    for idx0, (k, content) in enumerate(fields_to_validate):
        field_name = FIELD_NAME_MAP.get(k, k)
        msg = _build_validation_status_message(
            field_name=field_name,
            idx=idx0 + 1,
            total=total,
            content_len=len(content),
            llm_enabled=llm_enabled,
        )
        if animator is not None:
            animator.show(msg, force=True)

        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_validate_field_with_llm, k, content)
            while animator is not None and not future.done():
                animator.tick()
                time.sleep(0.3)
            is_valid = future.result()

        if not is_valid:
            st = _get_streamlit()
            if st is not None and _has_streamlit_runtime(st):
                st.toast(f"{FIELD_NAME_MAP.get(k, k)}填写的内容无效")
            continue
        validated_keys.append(k)

    if animator is not None:
        animator.clear()

    if not validated_keys:
        return False

    merged = " ".join(str(experience_details.get(k, "")) for k in validated_keys)
    merged = merged.strip().lower()
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", merged)
    return len(cleaned) >= 3
