from src.agent.utils import truncate_text


def build_field_validation_prompt(field_type: str, content: str) -> str:
    field_type_map = {
        "research_details": "科研项目",
        "award_details": "获奖情况",
        "internship_details": "实习经历",
        "paper_details": "论文发表",
    }
    field_name = field_type_map.get(field_type, field_type)

    text = truncate_text(content, 1200)

    return f"""请判断以下文本内容是否与"{field_name}"相关。

字段类型: {field_name}
文本内容: {text}

请只回答"是"或"否"，不要添加任何其他说明。
- 如果文本内容与{field_name}相关，回答"是"
- 如果文本内容与{field_name}不相关（例如在{field_name}中填写了其他类型的内容），回答"否"
- 如果文本内容为空或无效，回答"否"

你的回答:"""
