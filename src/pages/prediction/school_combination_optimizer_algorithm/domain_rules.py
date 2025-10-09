DOMAIN_EXCLUDE_BY_BACKGROUND: dict[str, dict] = {
    "商科金融": {
        "categories": {"医学", "法学", "教育学", "工程学", "理学"},
        "keywords": [
            "pcll",
            "law",
            "法律",
            "医学",
            "医药",
            "药政",
            "教育",
            "teacher",
            "medic",
        ],
    },
    "管理学": {
        "categories": {"医学", "法学", "教育学", "工程学", "理学"},
        "keywords": ["pcll", "law", "法律", "医学", "医药", "药政", "教育", "teacher", "medic"],
    },
    "经济学": {
        "categories": {"医学", "法学", "教育学", "工程学", "理学"},
        "keywords": ["pcll", "law", "法律", "医学", "医药", "药政", "教育", "teacher", "medic"],
    },
    "金融学": {
        "categories": {"医学", "法学", "教育学", "工程学", "理学"},
        "keywords": ["pcll", "law", "法律", "医学", "医药", "药政", "教育", "teacher", "medic"],
    },
    "工商管理": {
        "categories": {"医学", "法学", "教育学", "工程学", "理学"},
        "keywords": ["pcll", "law", "法律", "医学", "医药", "药政", "教育", "teacher", "medic"],
    },
}


def get_exclude_rules_for_background(background_category: str | None) -> dict:
    if not background_category:
        return {}
    for key in DOMAIN_EXCLUDE_BY_BACKGROUND.keys():
        if key in background_category:
            return DOMAIN_EXCLUDE_BY_BACKGROUND[key]
    return DOMAIN_EXCLUDE_BY_BACKGROUND.get(background_category, {})
