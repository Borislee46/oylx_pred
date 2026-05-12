"""Resolve school category aliases ("985", "港3", etc.) to concrete school names.

Used by form_bridge to map fuzzy aliases that the LLM can't expand on its own.
Leverages school_base_df for 985/211 classification and prediction_rules.json for
target university ordering.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from src.utils.app_data_loader import load_school_base_data


def _load_cases_university_counts() -> dict[str, int]:
    """Return background_university → frequency from historical cases."""
    try:
        cases_path = (
            Path(__file__).parent.parent / "machine_learning_models" / "data" / "cases.feather"
        )
        if cases_path.exists():
            df = pd.read_feather(cases_path)
            if "background_university" in df.columns:
                return df["background_university"].astype(str).str.strip().value_counts().to_dict()
    except Exception:
        pass
    return {}


def _get_target_universities_hk_sorted() -> list[str]:
    """HK target universities in display order, from prediction_rules.json."""
    try:
        import json

        rules_path = Path(__file__).parent.parent.parent / "config" / "prediction_rules.json"
        if rules_path.exists():
            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            all_unis = rules.get("UNIVERSITY_DISPLAY_ORDER", [])
            return [u for u in all_unis if "香港" in u or u.startswith("港")]
    except Exception:
        pass
    return [
        "香港大学",
        "香港中文大学",
        "香港科技大学",
        "香港理工大学",
        "香港城市大学",
        "香港浸会大学",
        "香港岭南大学",
        "香港教育大学",
    ]


@lru_cache(maxsize=1)
def _get_school_level_df() -> pd.DataFrame | None:
    """school_base_df filtered to useful columns, cached."""
    df = load_school_base_data()
    if df is None or "学校名称" not in df.columns:
        return None
    cols = ["学校名称"]
    if "school_level" in df.columns:
        cols.append("school_level")
    return df[cols].copy()


def resolve_background_school(alias: str) -> str:
    """Map a category alias like "985" / "211" to a representative school name.

    Picks the highest-frequency school of that category from historical cases.
    Falls back to a well-known representative if no case data is available.
    """
    alias_clean = str(alias).strip()
    category = alias_clean.upper() if alias_clean.isascii() else alias_clean

    # Known fallbacks (used when case data is unavailable)
    _FALLBACKS: dict[str, str] = {
        "985": "北京大学",
        "211": "北京工业大学",
        "双一流": "北京大学",
        "双非": "深圳大学",
        "普通本科": "深圳大学",
    }
    fallback = _FALLBACKS.get(category)

    df = _get_school_level_df()
    if df is None:
        return fallback or alias_clean

    # Filter by school_level
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

    # Pick the most frequent one from historical cases
    counts = _load_cases_university_counts()
    best_name = max(matching_names, key=lambda n: counts.get(n, 0))
    return best_name if counts.get(best_name, 0) > 0 else (fallback or matching_names[0])


def resolve_target_schools(alias: str) -> list[str]:
    """Map "港N" / "坡N" to the top N target schools in that region.

    Examples:
        "港3"  → ["香港大学", "香港中文大学", "香港科技大学"]
        "港5"  → top 5 HK schools
        "港8"  → all 8 HK schools
    """
    import re

    alias_clean = str(alias).strip()
    m = re.match(r"(港|坡|新|澳门|马)(\d+)", alias_clean)
    if not m:
        return []

    region = m.group(1)
    try:
        n = int(m.group(2))
    except ValueError:
        return []

    if region in ("港",):
        hk_sorted = _get_target_universities_hk_sorted()
        return hk_sorted[:n]

    # For other regions, load from prediction_rules
    try:
        import json

        rules_path = Path(__file__).parent.parent.parent / "config" / "prediction_rules.json"
        if rules_path.exists():
            rules = json.loads(rules_path.read_text(encoding="utf-8"))
            all_unis = rules.get("UNIVERSITY_DISPLAY_ORDER", [])
            region_keywords: dict[str, str] = {
                "坡": "新加坡",
                "新": "新加坡",
                "澳门": "澳门",
                "马": "马来西亚",
            }
            kw = region_keywords.get(region, "")
            region_unis = [u for u in all_unis if kw in u]
            return region_unis[:n]
    except Exception:
        pass

    return []


def is_school_category_alias(value: str) -> bool:
    """Check if a value looks like a school category alias rather than a school name."""
    v = str(value).strip()
    if not v:
        return False
    import re

    if re.match(r"^(港|坡|新|澳门|马)\d+$", v):
        return True
    return v in {"985", "211", "双一流", "双非", "普通本科", "C9", "华五"}
