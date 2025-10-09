from typing import Any, Dict

import pandas as pd


def build_major_category_cache(
    background_major: str, details_df: pd.DataFrame | None
) -> tuple[str | None, Dict[str, str]]:
    background_major_category = None
    major_category_cache: Dict[str, str] = {}

    if details_df is None or details_df.empty:
        return background_major_category, major_category_cache

    required_cols = ["学校", "专业英文名称", "专业大类"]
    for col in required_cols:
        if col not in details_df.columns:
            return background_major_category, major_category_cache

    try:
        df = details_df[required_cols].copy()
        for col in required_cols:
            df[col] = df[col].astype(str).str.strip()

        bg_matches = df[df["专业英文名称"] == background_major]
        if not bg_matches.empty:
            background_major_category = str(bg_matches.iloc[0]["专业大类"]) or None

        valid = df.replace({"": pd.NA}).dropna(subset=["学校", "专业英文名称", "专业大类"])
        keys = (valid["学校"] + "|" + valid["专业英文名称"]).tolist()
        cats = valid["专业大类"].tolist()
        for k, c in zip(keys, cats):
            if k and c:
                major_category_cache[k] = c
    except Exception:
        try:
            bg_major_matches = details_df[details_df["专业英文名称"] == background_major]
            if not bg_major_matches.empty:
                background_major_category = bg_major_matches.iloc[0].get("专业大类", "") or None

            for _, row in details_df.iterrows():
                uni = row.get("学校", "")
                maj = row.get("专业英文名称", "")
                cat = row.get("专业大类", "")
                if uni and maj and cat:
                    cache_key = f"{uni}|{maj}"
                    major_category_cache[cache_key] = cat
        except Exception:
            pass

    if not background_major_category and background_major:
        try:
            bm = str(background_major).strip().lower()
            keyword_to_category = [
                ("金融", "金融学"),
                ("finance", "金融学"),
                ("经济", "经济学"),
                ("econom", "经济学"),
                ("贸", "经济学"),
                ("business", "工商管理"),
                ("工商", "工商管理"),
                ("管理", "管理学"),
                ("management", "管理学"),
                ("会计", "会计学"),
                ("account", "会计学"),
                ("市场", "市场营销"),
                ("marketing", "市场营销"),
            ]
            for kw, cat in keyword_to_category:
                if kw in bm:
                    background_major_category = cat
                    break
        except Exception:
            pass

    return background_major_category, major_category_cache


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
