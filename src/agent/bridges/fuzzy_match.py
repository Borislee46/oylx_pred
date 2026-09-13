from __future__ import annotations

import re
from functools import lru_cache

from rapidfuzz import fuzz, process

from src.utils.logger import setup_logger

_log = setup_logger("bridges_fuzzy", "agent")


_UNIVERSITY_ALIAS_MAP: dict[str, str] = {
    # ── 中国香港 ──
    "HKU": "香港大学",
    "THE UNIVERSITY OF HONG KONG": "香港大学",
    "UNIVERSITY OF HONG KONG": "香港大学",
    "CUHK": "香港中文大学",
    "THE CHINESE UNIVERSITY OF HONG KONG": "香港中文大学",
    "CHINESE UNIVERSITY OF HONG KONG": "香港中文大学",
    "HKUST": "香港科技大学",
    "THE HONG KONG UNIVERSITY OF SCIENCE AND TECHNOLOGY": "香港科技大学",
    "HONG KONG UNIVERSITY OF SCIENCE AND TECHNOLOGY": "香港科技大学",
    "POLYU": "香港理工大学",
    "THE HONG KONG POLYTECHNIC UNIVERSITY": "香港理工大学",
    "HONG KONG POLYTECHNIC UNIVERSITY": "香港理工大学",
    "CITYU": "香港城市大学",
    "CITY UNIVERSITY OF HONG KONG": "香港城市大学",
    "CUHKSZ": "香港中文大学 (深圳校区)",
    "CUHK-SHENZHEN": "香港中文大学 (深圳校区)",
    "CUHK SHENZHEN": "香港中文大学 (深圳校区)",
    "THE CHINESE UNIVERSITY OF HONG KONG SHENZHEN": "香港中文大学 (深圳校区)",
    "CHINESE UNIVERSITY OF HONG KONG SHENZHEN": "香港中文大学 (深圳校区)",
    "HKBU": "香港浸会大学",
    "HONG KONG BAPTIST UNIVERSITY": "香港浸会大学",
    "LINGNANU": "香港岭南大学",
    "LINGNAN": "香港岭南大学",
    "LU": "香港岭南大学",
    "HKLU": "香港岭南大学",
    "LINGNAN UNIVERSITY": "香港岭南大学",
    "EDUHK": "香港教育大学",
    "THE EDUCATION UNIVERSITY OF HONG KONG": "香港教育大学",
    "EDUCATION UNIVERSITY OF HONG KONG": "香港教育大学",
    "HKMU": "香港都会大学",
    "OUHK": "香港都会大学",
    "THE HONG KONG METROPOLITAN UNIVERSITY": "香港都会大学",
    "HONG KONG METROPOLITAN UNIVERSITY": "香港都会大学",
    "HSUHK": "香港恒生大学",
    "HSU": "香港恒生大学",
    "THE HANG SENG UNIVERSITY OF HONG KONG": "香港恒生大学",
    "HANG SENG UNIVERSITY OF HONG KONG": "香港恒生大学",
    "HANG SENG UNIVERSITY": "香港恒生大学",
    "HKCHC": "香港珠海学院",
    "CHUHAI": "香港珠海学院",
    "CHU HAI": "香港珠海学院",
    "CHU HAI COLLEGE": "香港珠海学院",
    "HONG KONG CHU HAI COLLEGE": "香港珠海学院",
    # ── 新加坡 ──
    "NUS": "新加坡国立大学",
    "NATIONAL UNIVERSITY OF SINGAPORE": "新加坡国立大学",
    "NTU": "新加坡南洋理工大学",
    "NANYANG TECHNOLOGICAL UNIVERSITY": "新加坡南洋理工大学",
    "SMU": "新加坡管理大学",
    "SINGAPORE MANAGEMENT UNIVERSITY": "新加坡管理大学",
    # ── 中国澳门 ──
    # "UM" 同时是 University of Macau 与 University of Malaya 的缩写；
    # 本业务语境下默认指向澳门大学，马来亚大学用全称 / UMALAYA 区分。
    "UM": "澳门大学",
    "UMAC": "澳门大学",
    "UMACAU": "澳门大学",
    "UNIVERSITY OF MACAU": "澳门大学",
    "MUST": "澳门科技大学",
    "MACAU UNIVERSITY OF SCIENCE AND TECHNOLOGY": "澳门科技大学",
    "MACAO UNIVERSITY OF SCIENCE AND TECHNOLOGY": "澳门科技大学",
    "MPU": "澳门理工大学",
    "IPM": "澳门理工大学",
    "MACAO POLYTECHNIC UNIVERSITY": "澳门理工大学",
    "MACAU POLYTECHNIC UNIVERSITY": "澳门理工大学",
    # CityU 默认指香港城市大学；澳门城市大学用全称 / CITYU MACAU 区分。
    "CITY UNIVERSITY OF MACAU": "澳门城市大学",
    "CITYU MACAU": "澳门城市大学",
    "CITYU-MACAU": "澳门城市大学",
    # ── 马来西亚 ──
    "UNIVERSITY OF MALAYA": "马来亚大学",
    "UMALAYA": "马来亚大学",
    "UPM": "马来西亚博特拉大学",
    "UNIVERSITI PUTRA MALAYSIA": "马来西亚博特拉大学",
    "USM": "马来西亚理科大学",
    "UNIVERSITI SAINS MALAYSIA": "马来西亚理科大学",
    "UKM": "马来西亚国立大学",
    "UNIVERSITI KEBANGSAAN MALAYSIA": "马来西亚国立大学",
}

_MAJOR_DISCIPLINE_MAP: dict[str, str] = {
    "CS": "Computer Science",
    "计算机": "Computer Science",
    "计算机科学": "Computer Science",
    "计算机科学与技术": "Computer Science",
    "信息技术": "Information Technology",
    "软件工程": "Software Engineering",
    "人工智能": "Artificial Intelligence",
    "AI": "Artificial Intelligence",
    "数据科学": "Data Science",
    "DS": "Data Science",
    "大数据": "Data Science",
    "金融": "Finance",
    "金融学": "Finance",
    "金融工程": "Financial Engineering",
    "会计": "Accounting",
    "商业分析": "Business Analytics",
    "BA": "Business Analytics",
    "市场营销": "Marketing",
    "管理": "Management",
    "工商管理": "Business Administration",
    "MBA": "Business Administration",
    "经济": "Economics",
    "经济学": "Applied Economics",
    "电子工程": "Electronic Engineering",
    "电子": "Electronic Engineering",
    "EE": "Electronic Engineering",
    "电气": "Electrical Engineering",
    "机械": "Mechanical Engineering",
    "土木": "Civil Engineering",
    "材料": "Materials Engineering",
    "生物医学": "Biomedical Engineering",
    "化学工程": "Chemical Engineering",
    "统计": "Statistics",
    "统计学": "Statistics",
    "数学": "Mathematics",
    "物理": "Physics",
    "化学": "Chemistry",
    "生物": "Biology",
    "法律": "Law",
    "法学": "Law",
    "传媒": "Media",
    "传播": "Communication",
    "设计": "Design",
    "建筑": "Architecture",
    "教育": "Education",
    "公共管理": "Public Management",
    "供应链": "Supply Chain Management",
    "心理学": "Psychology",
}

_BG_MAJOR_ALIAS_MAP: dict[str, str] = {
    "CS": "计算机科学与技术",
    "EE": "电子信息工程",
    "AI": "人工智能",
    "DS": "数据科学与大数据技术",
    "BA": "商业分析",
    "MBA": "工商管理",
}

_GPA_SCALE_FROM_TEXT = re.compile(r"/\s*(100|10|5\.0|4\.3|4\.0|4)\b")


def _infer_gpa_scale_from_text(text: str) -> str | None:
    if not text:
        return None
    match = _GPA_SCALE_FROM_TEXT.search(str(text))
    if not match:
        return None
    token = match.group(1)
    return "4.0" if token == "4" else token


def _chars_in_order(query: str, candidate: str) -> bool:
    if not query or not candidate:
        return False
    pos = 0
    for ch in query:
        pos = candidate.find(ch, pos)
        if pos == -1:
            return False
        pos += 1
    return True


def _fuzzy_match(query: str, candidates: list[str]) -> tuple[str, float]:
    if not candidates:
        _log.debug("FUZZY_MATCH | query=%s | no candidates → conf=0.0", query)
        return query, 0.0

    for c in candidates:
        if c == query or c.casefold() == query.casefold():
            _log.debug("FUZZY_MATCH | query=%s → exact match %s conf=1.0", query, c)
            return c, 1.0

    if query.isascii() or len(query) <= 4:
        for c in candidates:
            if query in c or c in query:
                _log.debug("FUZZY_MATCH | query=%s → substring match %s conf=0.95", query, c)
                return c, 0.95

    cutoff = 65 if query.isascii() else 88
    match = process.extractOne(query, candidates, scorer=fuzz.partial_ratio, score_cutoff=cutoff)
    if match:
        conf = match[1] / 100.0
        _log.debug(
            "FUZZY_MATCH | query=%s → rapidfuzz %s conf=%.2f cutoff=%d",
            query,
            match[0],
            conf,
            cutoff,
        )
        return match[0], conf

    if query and candidates:
        for c in candidates:
            if _chars_in_order(query, c):
                _log.debug("FUZZY_MATCH | query=%s → char-order match %s conf=0.70", query, c)
                return c, 0.70

    alias_key = query.upper()
    if alias_key.startswith("THE "):
        alias_key = alias_key[4:]
    alias = _UNIVERSITY_ALIAS_MAP.get(alias_key)
    if alias and alias in candidates:
        _log.debug("FUZZY_MATCH | query=%s → alias in candidates %s conf=0.85", query, alias)
        return alias, 0.85
    if alias:
        _log.debug("FUZZY_MATCH | query=%s → alias not in candidates %s conf=0.60", query, alias)
        return alias, 0.60

    _log.debug("FUZZY_MATCH | query=%s → no match conf=0.0", query)
    return query, 0.0


@lru_cache(maxsize=1)
def _load_target_major_options() -> list[str]:
    try:
        from src.pages.prediction.app_data import load_school_major_details_df

        df = load_school_major_details_df()
        if df is not None and "专业英文名称" in df.columns:
            return sorted(df["专业英文名称"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    return []


def _fuzzy_match_major(raw: str) -> str:
    if not raw or not raw.strip():
        return raw

    query = raw.strip()
    options = _load_target_major_options()
    if not options:
        return query

    _DEGREE_SUFFIX_RE = r"(硕士|博士|学士|研究生|Master|PhD|Doctor|Bachelor|MSc|MPhil|MA|MBA).*$"
    stripped = re.sub(_DEGREE_SUFFIX_RE, "", query).strip()
    if stripped and stripped != query:
        discipline = _MAJOR_DISCIPLINE_MAP.get(stripped)
        if discipline:
            match = process.extractOne(
                discipline, options, scorer=fuzz.partial_ratio, score_cutoff=70
            )
            if match:
                _log.debug(
                    "FUZZY_MATCH_MAJOR | query=%s → stripped+discipline %s → %s",
                    query,
                    discipline,
                    match[0],
                )
                return match[0]
        match = process.extractOne(stripped, options, scorer=fuzz.partial_ratio, score_cutoff=75)
        if match:
            _log.debug("FUZZY_MATCH_MAJOR | query=%s → stripped direct %s", query, match[0])
            return match[0]

    match = process.extractOne(query, options, scorer=fuzz.partial_ratio, score_cutoff=75)
    if match:
        _log.debug("FUZZY_MATCH_MAJOR | query=%s → direct match %s", query, match[0])
        return match[0]

    discipline = _MAJOR_DISCIPLINE_MAP.get(query)
    if not discipline:
        for k, v in _MAJOR_DISCIPLINE_MAP.items():
            if k.casefold() == query.casefold():
                discipline = v
                break
    if discipline:
        match = process.extractOne(discipline, options, scorer=fuzz.partial_ratio, score_cutoff=70)
        if match:
            _log.debug("FUZZY_MATCH_MAJOR | query=%s → alias %s → %s", query, discipline, match[0])
            return match[0]

    for opt in options:
        if _chars_in_order(query, opt):
            _log.debug("FUZZY_MATCH_MAJOR | query=%s → char-order match %s", query, opt)
            return opt

    _log.debug("FUZZY_MATCH_MAJOR | query=%s → no match, returning raw", query)
    return query


def fuzzy_match_major_smart(raw: str, *, limit: int = 10, min_score: float = 500.0) -> list[str]:
    exact = _fuzzy_match_major(raw)
    if exact != raw and exact.strip():
        return [exact]

    options = _load_target_major_options()
    if not options:
        return [raw]

    results = fuzzy_search_multi(
        raw,
        options,
        use_expansion=True,
        limit=limit,
        min_score=min_score,
    )
    if results:
        names = [r[0] for r in results]
        _log.info(
            "FUZZY_MAJOR_SMART | query=%s → %d matches: %s",
            raw,
            len(names),
            names[:5],
        )
        return names

    return [raw]


@lru_cache(maxsize=1)
def _load_school_major_index() -> dict[str, set[str]]:
    try:
        from src.pages.prediction.app_data import load_school_major_details_df

        df = load_school_major_details_df()
        if df is not None and "学校" in df.columns and "专业英文名称" in df.columns:
            result: dict[str, set[str]] = {}
            for _, row in df.iterrows():
                school = str(row["学校"]).strip() if row["学校"] else ""
                major = str(row["专业英文名称"]).strip() if row["专业英文名称"] else ""
                if school and major:
                    result.setdefault(school, set()).add(major)
            return result
    except Exception:
        pass
    return {}


def cross_reference_school_majors(
    schools: list[str],
    majors: list[str],
) -> list[str]:
    """Filter majors to only those offered by the target schools.

    Args:
        schools: List of target university names (e.g. ["香港大学", "香港中文大学"]).
        majors:  List of matched major names (e.g. ["Master of Applied Linguistics", ...]).

    Returns:
        Subset of *majors* that exist in ``school_major_details`` for any of
        the given *schools*.  If no school has any of the majors, returns
        the original list untouched (to avoid silently zeroing out results
        due to name mismatches).
    """
    if not schools or not majors:
        return majors

    index = _load_school_major_index()
    if not index:
        return majors

    valid: set[str] = set()
    for s in schools:
        school_majors = index.get(s)
        if school_majors:
            valid.update(school_majors)
        else:
            for key in index:
                if fuzz.partial_ratio(s, key) >= 85:
                    valid.update(index[key])

    if not valid:
        return majors

    filtered = [m for m in majors if m in valid]
    if not filtered:
        _log.warning(
            "CROSS_REF | 所有 major 在校方数据中未命中 | schools=%s majors=%s",
            schools,
            majors[:5],
        )
        return majors

    removed = len(majors) - len(filtered)
    if removed > 0:
        _log.info(
            "CROSS_REF | filtered %d majors not offered by schools | kept=%d",
            removed,
            len(filtered),
        )

    return filtered


def fuzzy_search_multi(
    query: str,
    candidates: list[str],
    *,
    use_expansion: bool = True,
    limit: int = 15,
    min_score: float = 50.0,
) -> list[tuple[str, float]]:
    from src.utils.search import build_search_expander, smart_search

    expander = build_search_expander() if use_expansion else None

    results = smart_search(
        query,
        candidates,
        expander=expander,
        limit=min(limit, 30),
    )

    filtered = [(c, s) for c, s, _ in results if s >= min_score]
    return filtered[:limit]
