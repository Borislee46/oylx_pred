from functools import lru_cache

from src.utils.schools.config_loader import TARGET_COUNTRY_UNIVERSITY_MAP as _COUNTRY_MAP

_NON_MASTER_KEYWORDS = (
    "博士",
    "phd",
    "ph.d",
    "doctor",
    "doctoral",
    "本科",
    "学士",
    "bachelor",
    "undergrad",
    "undergraduate",
    "交换",
    "exchange",
    "预科",
    "foundation",
    "语言班",
    "副学士",
    "associate",
    "大专",
    "diploma",
)


@lru_cache(maxsize=1)
def _load_country_map() -> dict[str, list[str]]:
    return dict(_COUNTRY_MAP)


_REGION_ALIASES: dict[str, str] = {
    "香港": "中国香港",
    "hk": "中国香港",
    "hong kong": "中国香港",
    "hongkong": "中国香港",
    "澳门": "中国澳门",
    "macau": "中国澳门",
    "macao": "中国澳门",
    "新加坡": "新加坡",
    "singapore": "新加坡",
    "sg": "新加坡",
    "马来西亚": "马来西亚",
    "malaysia": "马来西亚",
    "my": "马来西亚",
    "大马": "马来西亚",
}


def normalize_region(raw: str) -> str:
    if not raw:
        return ""
    key = raw.strip().lower()
    return _REGION_ALIASES.get(key, raw)


def supported_countries() -> list[str]:
    return list(_load_country_map().keys())


def supported_schools() -> set[str]:
    out: set[str] = set()
    for schools in _load_country_map().values():
        out.update(schools)
    return out


def is_non_master_degree(degree_level: str) -> bool:
    if not degree_level:
        return False
    low = degree_level.strip().lower()
    return any(kw in low for kw in _NON_MASTER_KEYWORDS)


def evaluate_scope(
    country: str = "",
    schools: list[str] | None = None,
    degree_level: str = "",
) -> dict:
    issues: list[str] = []
    countries = supported_countries()

    normalized_country = normalize_region(country) if country else ""
    if country and normalized_country not in countries:
        issues.append(f"目前仅支持 {('、'.join(countries))} 地区，暂不支持「{country}」。")

    unsupported_schools: list[str] = []
    if schools:
        known = supported_schools()
        for s in schools:
            s_str = str(s).strip()
            if not s_str:
                continue
            if s_str in known or any(s_str in k or k in s_str for k in known):
                continue
            unsupported_schools.append(s_str)
        if unsupported_schools:
            issues.append("以下目标院校不在支持范围内：" + "、".join(unsupported_schools) + "。")

    if is_non_master_degree(degree_level):
        issues.append(f"本系统仅支持硕士（master）申请预测，暂不支持「{degree_level}」层次。")

    return {
        "in_scope": not issues,
        "issues": issues,
        "unsupported_schools": unsupported_schools,
        "supported_countries": countries,
    }
