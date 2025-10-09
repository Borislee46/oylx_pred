import streamlit as st

from src.pages.eu_sales_tool.formatter import (
    format_advantages,
    format_definitions,
    format_fees,
    format_language_pathways,
    format_popular_majors,
    format_programs,
    format_tiers,
)
from src.pages.eu_sales_tool.mapper import TOPIC_KEYWORDS, TOPIC_TO_JSON_KEY_MAPPING
from src.pages.eu_sales_tool.ouying_mapper import handle_ouying_query
from src.pages.eu_sales_tool.utils import clean_user_input, stream_text


def generate_response(prompt, all_country_data):
    cleaned_prompt = clean_user_input(prompt)

    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = {
            "last_country_context": None,
            "last_topic_context": None,
            "ouying_context": False,
        }

    country_aliases = {
        "法国": ["法国", "france", "french"],
        "德国": ["德国", "germany", "german"],
        "西班牙": ["西班牙", "spain", "spanish"],
        "意大利": ["意大利", "italy", "italian"],
        "荷兰": ["荷兰", "netherlands", "dutch"],
        "瑞士": ["瑞士", "switzerland", "swiss"],
        "挪威": ["挪威", "norway", "norwegian"],
        "瑞典": ["瑞典", "sweden", "swedish"],
        "丹麦": ["丹麦", "denmark", "danish"],
        "芬兰": ["芬兰", "finland", "finnish"],
        "匈牙利": ["匈牙利", "hungary", "hungarian"],
        "俄罗斯": ["俄罗斯", "russia", "russian"],
        "爱尔兰": ["爱尔兰", "ireland", "irish"],
    }

    ouying_countries = ["丹麦", "芬兰", "匈牙利", "爱尔兰", "荷兰", "挪威", "瑞典", "瑞士"]

    if "欧英" in cleaned_prompt:
        st.session_state.conversation_context["ouying_context"] = True
        st.session_state.conversation_context["last_country_context"] = "欧英"

        found_topic_key = None
        sorted_keywords = sorted(TOPIC_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True)
        for keyword, topic_key in sorted_keywords:
            if keyword in cleaned_prompt:
                found_topic_key = topic_key
                break

        if found_topic_key:
            st.session_state.conversation_context["last_topic_context"] = found_topic_key
            yield from handle_ouying_query(found_topic_key, ouying_countries, all_country_data)
        else:
            available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
            yield from stream_text(
                f"您好！关于欧英国家（{' '.join(ouying_countries)}），您想了解哪方面的信息呢？\n\n我可以提供以下信息：{available_topics}。"
            )
        return

    found_country = None
    prompt_lower = cleaned_prompt.lower()

    for country_name in all_country_data.keys():
        if country_name in cleaned_prompt:
            found_country = country_name
            break

    if not found_country:
        for country_name, aliases in country_aliases.items():
            if country_name in all_country_data:
                for alias in aliases:
                    if alias.lower() in prompt_lower:
                        found_country = country_name
                        break
            if found_country:
                break

    found_topic_key = None
    sorted_keywords = sorted(TOPIC_KEYWORDS.items(), key=lambda x: len(x[0]), reverse=True)
    for keyword, topic_key in sorted_keywords:
        if keyword in cleaned_prompt:
            found_topic_key = topic_key
            break

    if not found_country and found_topic_key:
        if st.session_state.conversation_context["ouying_context"]:
            st.session_state.conversation_context["last_topic_context"] = found_topic_key
            yield from handle_ouying_query(found_topic_key, ouying_countries, all_country_data)
            return
        elif (
            st.session_state.conversation_context["last_country_context"]
            and st.session_state.conversation_context["last_country_context"] != "欧英"
        ):
            found_country = st.session_state.conversation_context["last_country_context"]
            if found_country in all_country_data:
                yield from handle_specific_country_query(
                    found_country, found_topic_key, all_country_data
                )
                return

    if found_country:
        st.session_state.conversation_context["last_country_context"] = found_country
        st.session_state.conversation_context["ouying_context"] = False

        if found_topic_key:
            st.session_state.conversation_context["last_topic_context"] = found_topic_key
            yield from handle_specific_country_query(
                found_country, found_topic_key, all_country_data
            )
        else:
            available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
            yield from stream_text(
                f"您好！关于{found_country}，您想了解哪方面的信息呢？\n\n我可以提供以下信息：{available_topics}。"
            )
        return

    if st.session_state.conversation_context["ouying_context"]:
        available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
        yield from stream_text(
            f"关于欧英国家（{' '.join(ouying_countries)}），您想了解哪方面的信息呢？\n\n我可以提供以下信息：{available_topics}。"
        )
        return
    elif (
        st.session_state.conversation_context["last_country_context"]
        and st.session_state.conversation_context["last_country_context"] != "欧英"
    ):
        available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
        country = st.session_state.conversation_context["last_country_context"]
        yield from stream_text(
            f"关于{country}，您想了解哪方面的信息呢？\n\n我可以提供以下信息：{available_topics}。"
        )
        return

    yield from stream_text(
        "抱歉，我无法识别您提到的国家。您可以尝试输入国家的全名，例如 '法国' 或 '德国'，或者询问 '欧英' 相关信息。"
    )


def handle_specific_country_query(country, topic_key, all_country_data):
    country_specific_data = all_country_data[country]
    json_key = TOPIC_TO_JSON_KEY_MAPPING.get(topic_key)

    if not json_key or json_key not in country_specific_data:
        available_topics = "、".join(TOPIC_TO_JSON_KEY_MAPPING.keys())
        yield from stream_text(
            f"抱歉，我没有找到关于{country}的'{topic_key}'信息。您可以问我关于{country}的其它信息，例如：{available_topics}。"
        )
        return

    data = country_specific_data[json_key]
    response_text = ""
    if json_key == "advantages":
        if topic_key == "费用信息":
            response_text = format_fees(data)
        else:
            response_text = format_advantages(data)
    elif json_key == "undergraduate_programs":
        response_text = format_programs(data)
    elif json_key == "university_tiers":
        response_text = format_tiers(data)
    elif json_key == "popular_majors":
        response_text = format_popular_majors(data)
    elif json_key == "language_pathways":
        response_text = format_language_pathways(data)
    elif json_key == "definitions_and_planning":
        response_text = format_definitions(data)
    else:
        response_text = "抱歉，对于这个主题我暂时没有特定的格式化输出。"

    yield from stream_text(response_text)
