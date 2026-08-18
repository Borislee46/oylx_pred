from __future__ import annotations

import pandas as pd

SIMILARITY_DEFAULT = 0.5

_MAJOR_FACULTY_KEYWORDS: dict[str, str] = {
    "工商管理": "商学院",
    "金融": "商学院",
    "会计": "商学院",
    "经济": "商学院",
    "市场": "商学院",
    "商业": "商学院",
    "商务": "商学院",
    "管理": "商学院",
    "计算机": "计算机学院",
    "数据科学": "计算机学院",
    "人工智能": "计算机学院",
    "软件": "计算机学院",
    "信息": "计算机学院",
    "医学": "医学院",
    "临床": "医学院",
    "护理": "医学院",
    "药学": "医学院",
    "公共卫生": "医学院",
    "中药": "医学院",
    "数学": "理学院",
    "物理": "理学院",
    "化学": "理学院",
    "生物": "理学院",
    "统计": "理学院",
    "环境科学": "理学院",
    "电子": "工程学院",
    "工程": "工程学院",
    "机械": "工程学院",
    "土木": "工程学院",
    "材料": "工程学院",
    "中文": "文学院",
    "文学": "文学院",
    "历史": "文学院",
    "哲学": "文学院",
    "语言": "文学院",
    "文化": "文学院",
    "翻译": "文学院",
    "英语": "文学院",
    "法律": "法学院",
    "法学": "法学院",
    "建筑": "建筑学院",
    "设计": "设计学院",
    "艺术": "艺术学院",
    "传媒": "社会科学院",
    "新闻": "社会科学院",
    "社会": "社会科学院",
    "政治": "社会科学院",
    "心理": "社会科学院",
    "国际关系": "社会科学院",
    "教育": "教育学院",
    "financial engineering": "商学院",
    "biomedical engineering": "医学院",
    "chemical engineering": "理学院",
    "computer science": "计算机学院",
    "information technology": "计算机学院",
    "data science": "计算机学院",
    "artificial intelligence": "计算机学院",
    "big data": "计算机学院",
    "software engineering": "计算机学院",
    "electronic": "工程学院",
    "electrical": "工程学院",
    "mechanical": "工程学院",
    "civil": "工程学院",
    "engineering": "工程学院",
    "manufacturing": "工程学院",
    "finance": "商学院",
    "accounting": "商学院",
    "accountancy": "商学院",
    "economics": "商学院",
    "marketing": "商学院",
    "business": "商学院",
    "management": "商学院",
    "global management": "商学院",
    "chinese culture": "文学院",
    "chinese and history": "文学院",
    "history": "文学院",
    "literature": "文学院",
    "language": "文学院",
    "translation": "文学院",
    "english": "文学院",
    "linguistics": "文学院",
    "law": "法学院",
    "legal": "法学院",
    "medicine": "医学院",
    "drug discovery": "医学院",
    "biomedical": "医学院",
    "public health": "医学院",
    "pharmacy": "医学院",
    "nursing": "医学院",
    "architecture": "建筑学院",
    "design": "设计学院",
    "social": "社会科学院",
    "psychology": "社会科学院",
    "journalism": "社会科学院",
    "communication": "社会科学院",
    "education": "教育学院",
    "mathematics": "理学院",
    "physics": "理学院",
    "chemistry": "理学院",
    "biology": "理学院",
    "statistics": "理学院",
}


def major_to_faculty(major: str) -> str:
    if not major:
        return "未知"
    m = major.lower()
    for keyword, faculty in sorted(
        _MAJOR_FACULTY_KEYWORDS.items(),
        key=lambda x: len(x[0]),
        reverse=True,
    ):
        if keyword.lower() in m:
            return faculty
    return "未知"


def get_major_similarity(
    bg_major: str,
    tg_major: str,
    sim_lookup: dict[tuple[str, str], float],
) -> float:
    key = (str(bg_major).strip().lower(), str(tg_major).strip().lower())
    return sim_lookup.get(key, SIMILARITY_DEFAULT)


def compute_cross_faculty_features(
    df: pd.DataFrame,
    whitelist: dict[str, list[str]],
    severity_lookup: dict[tuple[str, str], float],
) -> pd.DataFrame:
    if "faculty" in df.columns:
        bg_fac_col = df["faculty"].fillna("").apply(lambda x: x if x else "未知")
    else:
        bg_fac_col = df["background_major"].apply(major_to_faculty)

    tg_fac_col = df["target_major"].apply(major_to_faculty)

    scores = []
    for bg, tg in zip(bg_fac_col, tg_fac_col, strict=False):
        bg_str, tg_str = str(bg), str(tg)
        if bg_str == tg_str:
            scores.append(1.0)
        elif tg_str in whitelist.get(bg_str, []):
            scores.append(1.0)
        else:
            scores.append(severity_lookup.get((bg_str, tg_str), 0.30))

    df["cross_faculty_score"] = scores
    df["is_severe_cross_faculty"] = [1.0 if s <= 0.5 else 0.0 for s in scores]

    return df


def compute_bg_school_score(
    df: pd.DataFrame,
    school_score_map: dict[str, float],
) -> pd.DataFrame:
    default_score = school_score_map.get("_default", 0.50)

    if "background_university" in df.columns:
        df["bg_school_score"] = df["background_university"].apply(
            lambda x: school_score_map.get(str(x).strip(), default_score)
        )
    else:
        df["bg_school_score"] = default_score

    return df
