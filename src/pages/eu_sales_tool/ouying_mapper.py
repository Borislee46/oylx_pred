from src.pages.eu_sales_tool.formatter import (
    format_advantages,
    format_definitions,
    format_language_pathways,
    format_popular_majors,
    format_programs,
    format_tiers,
)
from src.pages.eu_sales_tool.mapper import TOPIC_TO_JSON_KEY_MAPPING
from src.pages.eu_sales_tool.utils import stream_text


def handle_ouying_query(topic_key, ouying_countries, all_country_data):
    json_key = TOPIC_TO_JSON_KEY_MAPPING.get(topic_key)

    if not json_key:
        available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
        yield from stream_text(
            f"抱歉，我没有找到相关信息。您可以问我关于欧英国家的其它信息，例如：{available_topics}。"
        )
        return

    response = f"关于欧英国家（{' '.join(ouying_countries)}）的{topic_key}：\n\n"

    for country in ouying_countries:
        if country in all_country_data:
            country_data = all_country_data[country]
            if json_key in country_data:
                response += f"**{country}**\n"
                data = country_data[json_key]

                if json_key == "advantages":
                    country_info = format_advantages(data)
                elif json_key == "undergraduate_programs":
                    country_info = format_programs(data)
                elif json_key == "university_tiers":
                    country_info = format_tiers(data)
                elif json_key == "popular_majors":
                    country_info = format_popular_majors(data)
                elif json_key == "language_pathways":
                    country_info = format_language_pathways(data)
                elif json_key == "definitions_and_planning":
                    country_info = format_definitions(data)
                else:
                    country_info = f"数据类型：{type(data)}"

                country_info = country_info.replace("该国的留学优势主要有：", "留学优势主要有：")
                country_info = country_info.replace("该国的热门专业包括：", "热门专业包括：")
                country_info = country_info.replace(
                    "关于项目与学费，信息如下：", "项目与学费信息："
                )
                country_info = country_info.replace("关于申请层级，请参考：", "申请层级信息：")
                country_info = country_info.replace(
                    "关于语言路径，有以下几种方式：", "语言路径信息："
                )
                country_info = country_info.replace("这里是一些相关定义解析：", "相关定义解析：")

                response += country_info + "\n\n"

    if not response.strip().endswith("：\n\n"):
        yield from stream_text(response)
    else:
        yield from stream_text("抱歉，我没有找到这些欧英国家的相关信息。")
