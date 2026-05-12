from src.agent.utils import truncate_text

_FIELD_LABELS = {
    "research_details": "科研项目",
    "award_details": "获奖情况",
    "internship_details": "实习经历",
    "paper_details": "论文发表",
}


def build_field_validation_prompt(field_type: str, content: str) -> str:
    field_name = _FIELD_LABELS.get(field_type, field_type)
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
            lines.append(f"{_FIELD_LABELS.get(key, key)}：{truncate_text(content, 400)}")

    if not lines:
        return ""

    items = "\n".join(lines)

    return f"""判断以下经历字段是否包含实质性信息（非占位符/无效内容），返回JSON格式：

{items}

返回格式: {{"research_details": true/false, "award_details": true/false, "internship_details": true/false, "paper_details": true/false}}
只返回JSON，不要其他内容。"""
