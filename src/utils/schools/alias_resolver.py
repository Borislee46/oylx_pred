from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from src.utils.schools.config_loader import UNIVERSITY_DISPLAY_ORDER, get_project_root
from src.utils.schools.data import load_school_base_data


@lru_cache(maxsize=1)
def _load_cases_university_counts() -> dict[str, int]:
    try:
        cases_path = get_project_root() / "src" / "ml" / "data" / "cases.feather"
        if cases_path.exists():
            df = pd.read_feather(cases_path)
            if "background_university" in df.columns:
                return df["background_university"].astype(str).str.strip().value_counts().to_dict()
    except Exception:
        pass
    return {}


def _get_target_universities_hk_sorted() -> list[str]:
    return [u for u in UNIVERSITY_DISPLAY_ORDER if "香港" in u or u.startswith("港")]


@lru_cache(maxsize=1)
def _get_school_level_df() -> pd.DataFrame | None:
    df = load_school_base_data()
    if df is None or "学校名称" not in df.columns:
        return None
    cols = ["学校名称"]
    if "school_level" in df.columns:
        cols.append("school_level")
    return df[cols].copy()


def resolve_background_school(alias: str) -> str:
    alias_clean = str(alias).strip()
    category = alias_clean.upper() if alias_clean.isascii() else alias_clean

    _FALLBACKS: dict[str, str] = {
        "985": "北京大学",
        "211": "北京工业大学",
        "双一流": "北京大学",
        "双非": "深圳大学",
        "普通本科": "深圳大学",
        "一本": "深圳大学",
        "二本": "广东工业大学",
        "三本": "广东东软学院",
    }
    fallback = _FALLBACKS.get(category)

    df = _get_school_level_df()
    if df is None:
        return fallback or alias_clean

    level_col = "school_level" if "school_level" in df.columns else None
    if level_col:
        matching = df[df[level_col].astype(str).str.strip() == alias_clean]
    else:
        matching = df

    if matching.empty:
        return fallback or alias_clean

    matching_names = matching["学校名称"].astype(str).str.strip().tolist()
    if not matching_names:
        return fallback or alias_clean

    if len(matching_names) == 1:
        return matching_names[0]

    counts = _load_cases_university_counts()
    best_name = max(matching_names, key=lambda n: counts.get(n, 0))
    return best_name if counts.get(best_name, 0) > 0 else (fallback or matching_names[0])


_CN_DIGIT_MAP: dict[str, int] = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_CN_DIGIT_PATTERN = re.compile(r"([一二三四五六七八九十]+)")


def _parse_cn_number(cn: str) -> int | None:
    cn = cn.strip()
    if not cn:
        return None
    if cn == "十":
        return 10
    if cn.startswith("十"):
        return 10 + _CN_DIGIT_MAP.get(cn[1], 0)
    if cn.endswith("十"):
        return _CN_DIGIT_MAP.get(cn[0], 0) * 10
    if "十" in cn:
        parts = cn.split("十")
        tens = _CN_DIGIT_MAP.get(parts[0], 0)
        ones = _CN_DIGIT_MAP.get(parts[1], 0)
        return tens * 10 + ones
    return _CN_DIGIT_MAP.get(cn)


def resolve_target_schools(alias: str) -> list[str]:
    alias_clean = str(alias).strip()
    m = re.match(r"(港|坡|新|澳门|马)(\d+)", alias_clean)
    if m:
        region = m.group(1)
        try:
            n = int(m.group(2))
        except ValueError:
            return []
    else:
        m = re.match(r"(港|坡|新|澳门|马)([一二三四五六七八九十]+)", alias_clean)
        if not m:
            return []
        region = m.group(1)
        n_raw = _parse_cn_number(m.group(2))
        if n_raw is None:
            return []
        n = n_raw

    if region in ("港",):
        hk_sorted = _get_target_universities_hk_sorted()
        return hk_sorted[:n]

    region_keywords: dict[str, str] = {
        "坡": "新加坡",
        "新": "新加坡",
        "澳门": "澳门",
        "马": "马来西亚",
    }
    kw = region_keywords.get(region, "")
    region_unis = [u for u in UNIVERSITY_DISPLAY_ORDER if kw in u]
    return region_unis[:n]


def get_category_display_label(alias: str) -> str:
    label_map: dict[str, str] = {
        "985": "985院校",
        "211": "211院校",
        "双一流": "双一流院校",
        "双非": "双非院校",
        "普通本科": "普通本科院校",
        "C9": "C9院校",
        "华五": "华五院校",
        "一本": "一本院校",
        "二本": "二本院校",
        "三本": "三本院校",
    }
    return label_map.get(str(alias).strip(), str(alias).strip())


def is_school_category_alias(value: str) -> bool:
    v = str(value).strip()
    if not v:
        return False

    if re.match(r"^(港|坡|新|澳门|马)(\d+|[一二三四五六七八九十]+)$", v):
        return True
    return v in {"985", "211", "双一流", "双非", "普通本科", "C9", "华五", "一本", "二本", "三本"}
