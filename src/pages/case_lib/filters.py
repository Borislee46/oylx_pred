import pandas as pd
import streamlit as st

from src.pages.case_lib import config


@st.cache_data
def get_base_filter_options(_df):
    options = {}

    if config.YEAR_COL in _df.columns:
        series = _df[config.YEAR_COL].dropna()
        series = series[series.astype(str) != ""]
        options["years"] = sorted(series.astype(str).unique().tolist(), reverse=True)

    if config.TARGET_COUNTRY_COL in _df.columns:
        series = _df[config.TARGET_COUNTRY_COL].dropna()
        series = series[series.astype(str) != ""]
        options["countries"] = sorted(series.unique().tolist())

    school_level_options = []

    if config.DOMESTIC_UNI_CLASSIFICATION_COL in _df.columns:
        domestic_series = _df[config.DOMESTIC_UNI_CLASSIFICATION_COL].dropna()
        domestic_series = domestic_series[domestic_series.astype(str) != ""]
        domestic_options = [
            f"{config.DOMESTIC_PREFIX}{option}" for option in domestic_series.unique().tolist()
        ]
        school_level_options.extend(domestic_options)

    if config.OVERSEAS_UNI_QS_RANK_COL in _df.columns:
        overseas_series = _df[config.OVERSEAS_UNI_QS_RANK_COL].dropna()
        overseas_series = overseas_series[overseas_series.astype(str) != ""]
        overseas_options = [
            f"{config.OVERSEAS_PREFIX}{option}" for option in overseas_series.unique().tolist()
        ]
        school_level_options.extend(overseas_options)

    if school_level_options:
        options["school_levels"] = sorted(school_level_options)
        options["has_domestic_classification"] = (
            config.DOMESTIC_UNI_CLASSIFICATION_COL in _df.columns
        )
        options["has_overseas_classification"] = config.OVERSEAS_UNI_QS_RANK_COL in _df.columns

    available_background_uni_col = next(
        (col for col in config.BACKGROUND_UNI_COLS if col in _df.columns), None
    )
    options["available_background_uni_col"] = available_background_uni_col
    if available_background_uni_col:
        series = _df[available_background_uni_col].dropna()
        series = series[series.astype(str) != ""]
        options["background_unis"] = sorted(series.unique().tolist())

    available_system_col = next((col for col in config.SYSTEM_COLS if col in _df.columns), None)
    options["available_system_col"] = available_system_col

    if available_system_col:
        series = _df[available_system_col].dropna()
        series = series[series.astype(str) != ""]
        options["体系"] = sorted(series.unique().tolist())

    if config.BACKGROUND_MAJOR_COL in _df.columns:
        series = _df[config.BACKGROUND_MAJOR_COL].dropna()
        series = series[series.astype(str) != ""]
        options["background_majors"] = sorted(series.unique().tolist())

    available_target_major_col = _get_target_major_col_for_category(_df)
    options["available_target_major_col"] = available_target_major_col
    if available_target_major_col:
        series = _df[available_target_major_col].dropna()
        series = series[series.astype(str) != ""]
        options["target_majors"] = sorted(series.unique().tolist())

    if config.ADMISSION_STATUS_COL in _df.columns:
        series = _df[config.ADMISSION_STATUS_COL].dropna()
        series = series[series.astype(str) != ""]
        series = series[series != "未申请"]
        options["admission_statuses"] = sorted(series.unique().tolist())

    return options


def _get_target_major_col_for_category(df):
    if "category" not in df.columns:
        return next((col for col in config.TARGET_MAJOR_COLS if col in df.columns), None)

    categories = df["category"].unique()

    if "undergrad" in categories:
        return "申请专业"
    elif "grad" in categories:
        return "专业英文名称修正"
    elif "phd" in categories:
        return "申请专业"

    return next((col for col in config.TARGET_MAJOR_COLS if col in df.columns), None)


@st.cache_data
def get_target_unis_options(_df, selected_countries: tuple = ()):
    filtered_df = _df
    if selected_countries and config.TARGET_COUNTRY_COL in _df.columns:
        filtered_df = _df[_df[config.TARGET_COUNTRY_COL].isin(list(selected_countries))]

    if config.TARGET_UNI_COL in filtered_df.columns:
        series = filtered_df[config.TARGET_UNI_COL]
        if pd.api.types.is_categorical_dtype(series):
            values = series.cat.categories.tolist()
            return sorted(
                [v for v in values if v and str(v) != "" and str(v) not in config.INVALID_VALUES]
            )
        series = series.dropna()
        series = series[series.astype(str) != ""]
        return sorted(series.unique().tolist())
    return []


def apply_filters(df, selections, filter_options):
    filtered_df = df

    if config.ADMISSION_STATUS_COL in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[config.ADMISSION_STATUS_COL] != "未申请"]

    if selections["admission_statuses"] and config.ADMISSION_STATUS_COL in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df[config.ADMISSION_STATUS_COL].isin(selections["admission_statuses"])
        ]

    if selections["years"] and config.YEAR_COL in filtered_df.columns:
        years_set = set(selections["years"])
        filtered_df = filtered_df[filtered_df[config.YEAR_COL].isin(years_set)]

    if selections["countries"] and config.TARGET_COUNTRY_COL in filtered_df.columns:
        countries_set = set(selections["countries"])
        filtered_df = filtered_df[filtered_df[config.TARGET_COUNTRY_COL].isin(countries_set)]

    if selections["target_unis"] and config.TARGET_UNI_COL in filtered_df.columns:
        target_unis_set = set(selections["target_unis"])
        filtered_df = filtered_df[filtered_df[config.TARGET_UNI_COL].isin(target_unis_set)]

    if selections["school_levels"]:
        domestic_selections = [
            s.replace(config.DOMESTIC_PREFIX, "")
            for s in selections["school_levels"]
            if s.startswith(config.DOMESTIC_PREFIX)
        ]
        overseas_selections = [
            s.replace(config.OVERSEAS_PREFIX, "")
            for s in selections["school_levels"]
            if s.startswith(config.OVERSEAS_PREFIX)
        ]

        school_level_mask = pd.Series(False, index=filtered_df.index)

        if domestic_selections and config.DOMESTIC_UNI_CLASSIFICATION_COL in filtered_df.columns:
            domestic_mask = filtered_df[config.DOMESTIC_UNI_CLASSIFICATION_COL].isin(
                domestic_selections
            )
            school_level_mask |= domestic_mask

        if overseas_selections and config.OVERSEAS_UNI_QS_RANK_COL in filtered_df.columns:
            overseas_mask = filtered_df[config.OVERSEAS_UNI_QS_RANK_COL].isin(overseas_selections)
            school_level_mask |= overseas_mask

        filtered_df = filtered_df[school_level_mask]

    language_col, language_score = selections.get("language_filter", (None, None))
    if language_col and language_score and language_col in filtered_df.columns:
        min_score, max_score = language_score
        numeric_series = filtered_df[language_col]
        filtered_df = filtered_df[numeric_series.between(min_score, max_score, inclusive="both")]

    if selections["background_majors"] and config.BACKGROUND_MAJOR_COL in filtered_df.columns:
        majors_set = set(selections["background_majors"])
        filtered_df = filtered_df[filtered_df[config.BACKGROUND_MAJOR_COL].isin(majors_set)]

    available_target_major_col = filter_options.get("available_target_major_col")
    if (
        selections["target_majors"]
        and available_target_major_col
        and available_target_major_col in filtered_df.columns
    ):
        target_majors_set = set(selections["target_majors"])
        filtered_df = filtered_df[filtered_df[available_target_major_col].isin(target_majors_set)]

    available_background_uni_col = filter_options.get("available_background_uni_col")
    if (
        selections["background_unis"]
        and available_background_uni_col
        and available_background_uni_col in filtered_df.columns
    ):
        background_unis_set = set(selections["background_unis"])
        filtered_df = filtered_df[
            filtered_df[available_background_uni_col].isin(background_unis_set)
        ]

    available_system_col = filter_options.get("available_system_col")
    if (
        selections.get("体系")
        and available_system_col
        and available_system_col in filtered_df.columns
    ):
        systems_set = set(selections["体系"])
        filtered_df = filtered_df[filtered_df[available_system_col].isin(systems_set)]

    return filtered_df
