import hashlib
from typing import Dict, List, Set

import pandas as pd
import streamlit as st

from src.pages.prediction.input_form_components.form_config import (
    TARGET_COUNTRY_UNIVERSITY_MAP,
    UNIVERSITY_SORT_ORDER,
)


@st.cache_data
def build_target_base_df_cached(
    unique_targets_df: pd.DataFrame | None, details_df: pd.DataFrame | None
) -> tuple[pd.DataFrame, Dict[str, str]]:
    return build_target_base_df(unique_targets_df, details_df)


def build_target_base_df(
    cases_df: pd.DataFrame | None, details_df: pd.DataFrame | None
) -> tuple[pd.DataFrame, Dict[str, str]]:
    base_df = pd.DataFrame(columns=["target_university", "target_major"])
    if cases_df is not None and all(
        col in cases_df.columns for col in ["target_university", "target_major"]
    ):
        base_df = cases_df[["target_university", "target_major"]].drop_duplicates()

    if details_df is not None:
        cols = ["学校", "专业英文名称", "专业大类"]
        has_agg = "专业英文名称_聚合" in details_df.columns
        if has_agg:
            cols.append("专业英文名称_聚合")
        details_subset = details_df[cols].rename(
            columns={
                "学校": "target_university",
                "专业英文名称": "target_major",
                "专业大类": "major_category",
                **({"专业英文名称_聚合": "target_major_agg"} if has_agg else {}),
            }
        )
        base_df = pd.merge(
            base_df,
            details_subset,
            on=["target_university", "target_major"],
            how="left",
        )
        if "target_major_agg" not in base_df.columns:
            base_df["target_major_agg"] = base_df["target_major"]

    country_uni_map = TARGET_COUNTRY_UNIVERSITY_MAP
    university_country_map = {
        uni: country for country, unis in country_uni_map.items() for uni in unis
    }
    if "target_university" in base_df.columns:
        base_df["country"] = base_df["target_university"].map(university_country_map)

    return base_df, university_country_map


def compute_selection_cache_key(
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    selected_majors: Set[str],
) -> str:
    key_parts = (
        tuple(sorted(selected_countries)),
        tuple(sorted(selected_universities)),
        tuple(sorted(selected_categories)),
        tuple(sorted(selected_majors)),
    )
    key_string = str(key_parts).encode("utf-8")
    return hashlib.sha256(key_string).hexdigest()


def _get_options(df: pd.DataFrame, column: str, selections: Set[str]) -> Set[str]:
    return set(df[column].dropna().unique()).union(selections)


def _filter_df_for_options(
    base_df: pd.DataFrame,
    countries: Set[str] | None = None,
    universities: Set[str] | None = None,
    categories: Set[str] | None = None,
    majors: Set[str] | None = None,
) -> pd.DataFrame:
    df = base_df
    if countries and "country" in df.columns:
        df = df[df["country"].isin(countries)]
    if universities and "target_university" in df.columns:
        df = df[df["target_university"].isin(universities)]
    if categories and "major_category" in df.columns:
        df = df[df["major_category"].isin(categories)]
    if majors and "target_major_agg" in df.columns:
        df = df[df["target_major_agg"].isin(majors)]
    return df.copy() if df is not base_df else df


def compute_options(
    base_df: pd.DataFrame,
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    selected_majors: Set[str],
) -> tuple[List[str], List[str], List[str], List[str]]:
    df_for_country = _filter_df_for_options(
        base_df,
        universities=selected_universities,
        categories=selected_categories,
        majors=selected_majors,
    )
    options_for_country_select = sorted(
        list(_get_options(df_for_country, "country", selected_countries))
    )

    df_for_uni = _filter_df_for_options(
        base_df,
        countries=selected_countries,
        categories=selected_categories,
        majors=selected_majors,
    )
    sort_order_map = {uni: i for i, uni in enumerate(UNIVERSITY_SORT_ORDER)}
    sort_key = lambda uni: (sort_order_map.get(uni, len(UNIVERSITY_SORT_ORDER)), uni)
    options_for_uni_select = sorted(
        list(_get_options(df_for_uni, "target_university", selected_universities)),
        key=sort_key,
    )

    df_for_cat = _filter_df_for_options(
        base_df,
        countries=selected_countries,
        universities=selected_universities,
        majors=selected_majors,
    )
    options_for_category_select = sorted(
        list(_get_options(df_for_cat, "major_category", selected_categories))
    )

    df_for_major = _filter_df_for_options(
        base_df,
        countries=selected_countries,
        universities=selected_universities,
        categories=selected_categories,
    )
    options_for_major_select = sorted(
        list(_get_options(df_for_major, "target_major_agg", selected_majors))
    )

    return (
        options_for_country_select,
        options_for_uni_select,
        options_for_category_select,
        options_for_major_select,
    )


def expand_aggregated_majors_for_prediction(
    base_df: pd.DataFrame,
    selected_countries: Set[str],
    selected_universities: Set[str],
    selected_categories: Set[str],
    aggregated_to_use: List[str],
) -> List[str]:
    if not aggregated_to_use:
        return []

    df_for_pred_major = base_df.copy()
    if selected_countries and "country" in df_for_pred_major.columns:
        df_for_pred_major = df_for_pred_major[df_for_pred_major["country"].isin(selected_countries)]
    if selected_universities and "target_university" in df_for_pred_major.columns:
        df_for_pred_major = df_for_pred_major[
            df_for_pred_major["target_university"].isin(selected_universities)
        ]
    if selected_categories and "major_category" in df_for_pred_major.columns:
        df_for_pred_major = df_for_pred_major[
            df_for_pred_major["major_category"].isin(selected_categories)
        ]

    if (
        "target_major" in df_for_pred_major.columns
        and "target_major_agg" in df_for_pred_major.columns
    ):
        return (
            df_for_pred_major[df_for_pred_major["target_major_agg"].isin(aggregated_to_use)][
                "target_major"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
    return []
