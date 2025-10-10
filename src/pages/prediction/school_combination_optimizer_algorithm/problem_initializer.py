from typing import Any, Dict

import pandas as pd


def build_major_category_cache(
    details_df: pd.DataFrame | None,
) -> Dict[str, str]:
    major_category_cache: Dict[str, str] = {}

    if details_df is None or details_df.empty:
        return major_category_cache

    required_cols = ["学校", "专业英文名称", "专业大类"]
    for col in required_cols:
        if col not in details_df.columns:
            return major_category_cache

    try:
        df = details_df[required_cols].copy()
        for col in required_cols:
            df[col] = df[col].astype(str).str.strip()

        valid = df.replace({"": pd.NA}).dropna(subset=["学校", "专业英文名称", "专业大类"])
        keys = (valid["学校"] + "|" + valid["专业英文名称"]).tolist()
        cats = valid["专业大类"].tolist()
        for k, c in zip(keys, cats):
            if k and c:
                major_category_cache[k] = c
    except Exception:
        try:
            for _, row in details_df.iterrows():
                uni = row.get("学校", "")
                maj = row.get("专业英文名称", "")
                cat = row.get("专业大类", "")
                if uni and maj and cat:
                    cache_key = f"{uni}|{maj}"
                    major_category_cache[cache_key] = cat
        except Exception:
            pass

    return major_category_cache


def build_new_major_cache(all_schools_data: list[dict[str, Any]]) -> Dict[str, Any]:
    new_major_cache: Dict[str, Any] = {}
    for school in all_schools_data:
        university = school.get("university", "")
        major = school.get("major", "")
        if university and major:
            cache_key = f"{university}|{major}"
            if cache_key not in new_major_cache:
                new_major_cache[cache_key] = school.get("is_new_major", False)
    return new_major_cache
