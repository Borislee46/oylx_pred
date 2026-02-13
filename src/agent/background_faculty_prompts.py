def build_background_faculty_prompt(
    background_major_original: str,
    base_faculty: str | None,
    valid_faculties: list[str],
    max_extra: int = 2,
) -> str:
    base = (base_faculty or "").strip()
    major = (background_major_original or "").strip()
    max_extra = int(max_extra) if max_extra is not None else 2
    if max_extra < 0:
        max_extra = 0

    return (
        "你是研究生申请背景识别助手。目标：判断一个“本科背景专业（原始名称）”可能对应哪些“专业大类/学院”。\n"
        "要求：\n"
        f'- 只输出 JSON object，格式固定：{{"extra_faculties": [..]}}。\n'
        f"- extra_faculties 只包含“在 base_faculty 之外可能还相关”的学院；最多 {max_extra} 个。\n"
        "- 只有在“明显跨学科/联合/双学位/多方向”或“专业与 base_faculty 明显不匹配/高度不确定”时，才输出 extra_faculties；否则必须输出空数组。\n"
        "- **严谨性**：如果无法判断或关联度极弱，请返回空数组。宁可少报，不可错报。\n"
        "- 只能从给定的 valid_faculties 里选，不要输出其它词。\n"
        "- 如果你认为不存在额外学院，输出空数组。\n"
        "- 不要解释，不要输出多余字段。\n\n"
        f"background_major_original: {major}\n"
        f"base_faculty: {base}\n"
        f"valid_faculties: {valid_faculties}\n"
    )
