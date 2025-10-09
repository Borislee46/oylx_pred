MAJOR_CATEGORIES = [
    "材料科学",
    "工商管理",
    "工程学",
    "工程管理",
    "教育技术学",
    "医学",
    "文学",
    "翻译学",
    "法学",
    "金融学",
    "艺术学",
    "理学",
    "管理学",
    "社会学",
    "生物学",
    "化学",
    "数学",
    "物理学",
    "计算机科学与技术",
    "管理科学与工程",
    "心理学",
    "经济学",
    "电子信息工程",
    "新闻传播学",
    "数据科学与大数据技术",
    "人工智能",
    "会计学",
    "市场营销",
    "历史学",
    "哲学",
    "电子与计算机工程",
    "统计学",
    "信息管理与信息系统",
    "信息系统",
    "信息科学与技术",
    "软件工程",
    "教育学",
    "信息工程",
    "土木工程",
    "人力资源",
    "商业分析",
    "音乐艺术",
]

STRICT_CATEGORIES = {
    "医学",
    "法学",
    "艺术学",
    "工程学",
    "理学",
    "生物学",
}

MODERATE_CATEGORIES = {
    "计算机科学与技术": 0.1,
    "电子信息工程": 0.1,
    "人工智能": 0.1,
    "软件工程": 0.2,
    "信息工程": 0.2,
    "数据科学与大数据技术": 0.2,
    "材料科学": 0.2,
    "化学": 0.2,
    "生物学": 0.2,
    "数学": 0.2,
    "物理学": 0.2,
    "统计学": 0.2,
    "土木工程": 0.2,
}


LOOSE_CATEGORIES = {
    "工商管理": 0.7,
    "金融学": 0.6,
    "管理学": 0.7,
    "经济学": 0.6,
    "会计学": 0.5,
    "市场营销": 0.7,
    "管理科学与工程": 0.6,
    "文学": 0.5,
    "翻译学": 0.5,
    "社会学": 0.5,
    "心理学": 0.5,
    "新闻传播学": 0.5,
    "教育学": 0.5,
    "教育技术学": 0.5,
    "历史学": 0.4,
    "哲学": 0.4,
    "信息管理与信息系统": 0.3,
    "信息系统": 0.3,
    "信息科学与技术": 0.3,
}


ALIASES: dict[str, str] = {
    "电子与计算机工程": "电子信息工程",
    "信息工程": "电子信息工程",
    "信息科学与技术": "计算机科学与技术",
    "信息系统": "信息管理与信息系统",
    "工程管理": "管理科学与工程",
    "商业分析": "管理科学与工程",
    "人力资源": "管理学",
    "音乐艺术": "艺术学",
    "翻译学": "文学",
    "历史学哲学": "历史学",
}


RELATED_GROUPS: dict[str, set[str]] = {
    "计算机与信息": {
        "计算机科学与技术",
        "软件工程",
        "人工智能",
        "数据科学与大数据技术",
        "电子信息工程",
        "信息管理与信息系统",
        "信息工程",
        "信息科学与技术",
    },
    "商科金融": {
        "工商管理",
        "管理学",
        "市场营销",
        "会计学",
        "金融学",
        "经济学",
        "管理科学与工程",
    },
    "文史社科": {
        "文学",
        "历史学",
        "哲学",
        "社会学",
        "新闻传播学",
        "教育学",
        "教育技术学",
        "心理学",
        "翻译学",
    },
    "理工基础": {
        "工程学",
        "理学",
        "统计学",
        "材料科学",
        "化学",
        "生物学",
        "数学",
        "物理学",
        "土木工程",
    },
}


def normalize_category(category: str) -> str:
    if not category:
        return ""
    category_clean = str(category).strip()
    return ALIASES.get(category_clean, category_clean)


def get_cross_major_limit(category: str) -> float:
    category = normalize_category(category)
    if category in STRICT_CATEGORIES:
        return 0.0
    elif category in MODERATE_CATEGORIES:
        return MODERATE_CATEGORIES[category]
    elif category in LOOSE_CATEGORIES:
        return LOOSE_CATEGORIES[category]
    else:
        return 0.5


def _get_category_strictness(category: str) -> int:
    category = normalize_category(category)
    if category in STRICT_CATEGORIES:
        return 2
    if category in MODERATE_CATEGORIES:
        return 1
    return 0


def get_category_similarity_threshold_adjustment(
    background_category: str, target_category: str
) -> float:
    background_category = normalize_category(background_category)
    target_category = normalize_category(target_category)

    if background_category == target_category:
        return 0.0

    background_strictness = _get_category_strictness(background_category)
    target_strictness = _get_category_strictness(target_category)

    for group_members in RELATED_GROUPS.values():
        if background_category in group_members and target_category in group_members:
            return 0.05

    max_strictness = max(background_strictness, target_strictness)

    if max_strictness == 2:
        return 0.3
    elif max_strictness == 1:
        return 0.2
    else:
        return 0.0
