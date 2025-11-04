import hashlib
import re
import string
from typing import Any

import requests
import streamlit as st

from src.agent.prompts import build_field_validation_prompt
from src.utils.env_config_loader import load_app_config
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

FIELD_NAME_MAP = {
    "research_details": "研究",
    "award_details": "奖项",
    "internship_details": "实习",
    "paper_details": "论文",
}

INVALID_TOKENS = {
    "",
    "无",
    "暂无",
    "没有",
    "na",
    "n/a",
    "none",
    "null",
    "-",
    "--",
    "—",
    "/",
    "／",
    ".",
}


def is_effectively_empty(text: str | None) -> bool:
    if text is None:
        return True
    t = str(text).strip()
    if not t:
        return True
    tl = t.lower()
    if tl in INVALID_TOKENS:
        return True
    punct = string.punctuation + "·—-_/／\\|~`'\"，。；：、"
    stripped = tl.strip(punct)
    return len(stripped) == 0


def clip_probability(value: Any) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_content_hash(content: str) -> str:
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def has_valid_experience_details(experience_details: dict[str, str] | None) -> bool:
    if not experience_details:
        return False
    keys = ("research_details", "award_details", "internship_details", "paper_details")
    for k in keys:
        if not is_effectively_empty(experience_details.get(k, "")):
            return True
    return False


@st.cache_data(ttl=3600)
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

    try:
        response = requests.post(api_url, headers=headers, json=data, timeout=10)
        response.raise_for_status()
        response_json = response.json()

        if response_json.get("choices"):
            result = response_json["choices"][0].get("message", {}).get("content", "").strip()
            return result == "是"
        else:
            logger.warning(f"LLM验证返回异常: {response_json.get('error', {})}")
            return True
    except Exception as e:
        logger.warning(f"LLM验证失败，使用默认通过: {e}")
        return True


def _validate_field_with_llm(field_type: str, content: str) -> bool:
    if is_effectively_empty(content):
        return False

    content_hash = generate_content_hash(f"{field_type}:{content}")
    return _validate_field_with_llm_cached(field_type, content_hash, content)


def has_meaningful_experience_text(experience_details: dict[str, str] | None) -> bool:
    if not has_valid_experience_details(experience_details):
        return False

    if experience_details is None:
        return False

    keys = (
        "research_details",
        "award_details",
        "internship_details",
        "paper_details",
    )

    validated_keys = []
    for k in keys:
        content = str(experience_details.get(k, "")).strip()
        if not is_effectively_empty(content):
            if not _validate_field_with_llm(k, content):
                field_name = FIELD_NAME_MAP.get(k, k)
                st.toast(f"{field_name}填写的内容无效")
                continue
            validated_keys.append(k)

    if not validated_keys:
        return False

    merged = " ".join(str(experience_details.get(k, "")) for k in validated_keys)
    merged = merged.strip().lower()
    cleaned = re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", merged)
    return len(cleaned) >= 3
