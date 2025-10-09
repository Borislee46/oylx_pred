def format_fees(data):
    response = "关于留学费用，相关信息如下：\n"
    cost_keywords = ["费", "价", "成本", "欧元", "€", "补贴", "奖学金", "薪", "保险", "租"]
    relevant_items = []

    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                title = item.get("title", "")
                desc = item.get("desc", "")
                if any(keyword in title for keyword in cost_keywords) or any(
                    keyword in desc for keyword in cost_keywords
                ):
                    relevant_items.append(item)

    if not relevant_items:
        return "抱歉，我没有找到关于该国家费用的具体信息。您可以尝试询问“留学优势”，或许能找到一些相关内容。"

    for item in relevant_items:
        title = item.get("title", "N/A")
        desc = item.get("desc", "N/A")
        response += f"\n- **{title}**: {desc}"

    return response


def format_advantages(data):
    response = "该国的留学优势主要有：\n"
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                title = item.get("title", "N/A")
                desc = item.get("desc", "N/A")
                response += f"\n- {title}: {desc}"
            else:
                response += f"\n- {item}"
    else:
        response += f"\n数据格式错误：{type(data)}"
    return response


def format_programs(data):
    response = "关于项目与学费，信息如下：\n"
    if isinstance(data, list):
        for program in data:
            if isinstance(program, dict):
                name = program.get("name", "N/A")
                description = program.get("description", "N/A")
                response += f"\n- {name}: {description}\n"
            else:
                response += f"\n- {program}\n"
    else:
        response += f"\n数据格式错误：{type(data)}"
    return response


def format_tiers(data):
    response = "关于申请层级，请参考：\n"
    if isinstance(data, list):
        for tier in data:
            if isinstance(tier, dict):
                level = tier.get("level", "N/A")
                recommendations = tier.get("recommendations", {})
                response += f"\n**{level}**\n"
                if isinstance(recommendations, dict):
                    for category, schools in recommendations.items():
                        response += f"{category}：\n"
                        if isinstance(schools, list):
                            for school in schools:
                                response += f"  - {school}\n"
                        else:
                            response += f"  - {schools}\n"
                else:
                    response += f"  - {recommendations}\n"
            else:
                response += f"\n- {tier}\n"
    else:
        response += f"\n数据格式错误：{type(data)}"
    return response


def format_popular_majors(data):
    response = "该国的热门专业包括：\n"
    if isinstance(data, dict):
        for category, majors in data.items():
            response += f"\n**{category}**\n"
            if isinstance(majors, list):
                for major in majors:
                    if isinstance(major, dict):
                        name = major.get("name", major.get("major", "N/A"))
                        response += f"- {name}\n"
                    else:
                        response += f"- {major}\n"
            else:
                response += f"- {majors}\n"
    elif isinstance(data, list):
        for major in data:
            response += f"\n- {major}"
    return response


def format_language_pathways(data):
    response = "关于语言路径，有以下几种方式：\n"

    lang_map = {
        "German": "德语",
        "French": "法语",
        "Spanish": "西班牙语",
        "Italian": "意大利语",
        "Dutch": "荷兰语",
        "Norwegian": "挪威语",
        "Swedish": "瑞典语",
        "Danish": "丹麦语",
        "Finnish": "芬兰语",
        "Hungarian": "匈牙利语",
        "Russian": "俄语",
    }

    if "english_taught" in data:
        english_data = data["english_taught"]
        response += "\n**英语授课**\n"
        response += f"概述: {english_data.get('overview', 'N/A')}\n"

        if "academic_requirements" in english_data:
            response += "\n学术要求:\n"
            for req in english_data["academic_requirements"]:
                title = req.get("title", "N/A")
                requirement = req.get("requirement", "N/A")
                response += f"- {title}: {requirement}\n"

        if "language_requirements" in english_data:
            response += "\n语言要求:\n"
            for req in english_data["language_requirements"]:
                test = req.get("test", "N/A")
                score = req.get("score", "N/A")
                response += f"- {test}: {score}\n"

    for key, value in data.items():
        if key != "english_taught" and isinstance(value, dict):
            lang_name = key.replace("_taught", "").title()
            display_lang_name = lang_map.get(lang_name, lang_name)
            response += f"\n**{display_lang_name}授课**\n"
            response += f"概述: {value.get('overview', 'N/A')}\n"

            if "academic_requirements" in value:
                response += "\n学术要求:\n"
                for req in value["academic_requirements"]:
                    title = req.get("title", "N/A")
                    requirement = req.get("requirement", "N/A")
                    response += f"- {title}: {requirement}\n"

            if "language_requirements" in value:
                response += "\n语言要求:\n"
                for req in value["language_requirements"]:
                    test = req.get("test", "N/A")
                    score = req.get("score", "N/A")
                    response += f"- {test}: {score}\n"

    return response


def format_definitions(data):
    response = "这里是一些相关定义解析：\n"
    if isinstance(data, dict):
        key_terms = data.get("key_terms", [])
        if isinstance(key_terms, list):
            for term_data in key_terms:
                if isinstance(term_data, dict):
                    term = term_data.get("term", "N/A")
                    definition = term_data.get("definition", "N/A")
                    response += f"\n- {term}: {definition}\n"
                else:
                    response += f"\n- {term_data}\n"
        else:
            response += f"\n数据格式错误：key_terms应该是列表，但是是{type(key_terms)}"
    else:
        response += f"\n数据格式错误：{type(data)}"
    return response
