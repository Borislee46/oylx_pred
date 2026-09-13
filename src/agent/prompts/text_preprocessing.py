from src.agent.field_labels import FIELD_LABEL
from src.agent.utils import truncate_text


def build_field_validation_prompt(field_type: str, content: str) -> str:
    field_name = FIELD_LABEL.get(field_type, field_type)
    text = truncate_text(content, 1200)

    return f"""请判断以下文本内容是否与"{field_name}"相关。

字段类型: {field_name}
文本内容: {text}

请只回答"是"或"否"，不要添加任何其他说明。
- 如果文本内容与{field_name}相关，回答"是"
- 如果文本内容与{field_name}不相关（例如在{field_name}中填写了其他类型的内容），回答"否"
- 如果文本内容为空或无效，回答"否"

你的回答:"""


def build_batch_validation_prompt(fields: dict[str, str]) -> str:
    lines: list[str] = []
    for key in ("research_details", "award_details", "internship_details", "paper_details"):
        content = fields.get(key, "")
        if content and content.strip():
            lines.append(f"{FIELD_LABEL.get(key, key)}：{truncate_text(content, 400)}")

    if not lines:
        return ""

    items = "\n".join(lines)

    return f"""{items}

返回格式: {{"research_details": true/false, "award_details": true/false, "internship_details": true/false, "paper_details": true/false}}
"""


def build_quality_verification_prompt(
    fields: dict[str, dict[str, object]],
) -> str:
    lines: list[str] = []
    for key in ("research_details", "award_details", "internship_details", "paper_details"):
        info = fields.get(key)
        if not info or not isinstance(info, dict):
            continue
        text = str(info.get("text", "") or "").strip()
        if not text:
            continue
        label = str(info.get("label", key))
        hits = info.get("signal_hits") or []
        hits_str = ", ".join(hits) if hits else "(无命中)"
        lines.append(f"### {label}\n文本：{truncate_text(text, 600)}\n词典命中标签：{hits_str}")

    if not lines:
        return ""

    items = "\n\n".join(lines)

    return f"""对以下经历文本做含金量校验，逐字段评估。

说明：每个字段的"词典命中标签"来自关键词子串匹配（SignalScorer），可能有误识别或漏检。

{items}

请按系统提示中的评估维度输出 JSON：
{{"research_details": {{"is_valid": true, "verified_tags": [...], "downgraded_tags": [...], "missed_signals": [...], "quality_level": "...", "concern": null}}, ...}}
"""


_QUALITY_PROMPT_CACHE: dict[str, str] = {}


def build_quality_verification_prompt_cached(
    fields: dict[str, dict[str, object]],
) -> str:
    key_parts = []
    for k in ("research_details", "award_details", "internship_details", "paper_details"):
        info = fields.get(k)
        if info and isinstance(info, dict):
            key_parts.append(f"{k}:{info.get('text', '')}")
    cache_key = "|".join(key_parts)
    if cache_key not in _QUALITY_PROMPT_CACHE:
        _QUALITY_PROMPT_CACHE[cache_key] = build_quality_verification_prompt(fields)
    return _QUALITY_PROMPT_CACHE[cache_key]
