import glob
import os

import pandas as pd
import streamlit as st

from src.pages.case_lib import config


@st.cache_data(ttl=300)
def get_available_categories():
    file_pattern = os.path.join(config.DATA_DIR, config.CASES_FILE_PATTERN)
    files = glob.glob(file_pattern)
    categories = [os.path.basename(f).replace("cases_", "").replace(".feather", "") for f in files]
    return categories


@st.cache_data
def load_data_by_categories(categories):
    """
    加载指定类别的案例数据

    Returns:
        pd.DataFrame: 合并后的数据框，如果出错则返回空DataFrame
    """
    if not categories:
        return pd.DataFrame()

    df_list = []
    errors = []
    warnings = []

    for category in categories:
        file_path = os.path.join(config.DATA_DIR, f"cases_{category}.feather")
        if os.path.exists(file_path):
            try:
                df = pd.read_feather(file_path)
                if df.empty:
                    warnings.append(f"{category} 数据文件为空")
                else:
                    df_list.append(df)
            except Exception as e:
                errors.append(f"加载 {category} 数据时出错: {e}")
        else:
            warnings.append(f"未找到 {category} 的数据文件")

    # 将错误和警告存储到session state，由UI层统一显示
    if errors:
        st.session_state["case_lib_data_errors"] = errors
    if warnings:
        st.session_state["case_lib_data_warnings"] = warnings

    if not df_list:
        return pd.DataFrame()

    try:
        combined_df = pd.concat(df_list, ignore_index=True, sort=False)
    except Exception as e:
        error_msg = f"合并数据时出错: {e}"
        st.session_state["case_lib_data_errors"] = [error_msg]
        return pd.DataFrame()

    # 加载学校基础信息并映射国家
    if os.path.exists(config.SCHOOL_BASE_PATH):
        try:
            school_base_df = pd.read_feather(config.SCHOOL_BASE_PATH)

            if (
                config.TARGET_UNI_COL in combined_df.columns
                and config.SCHOOL_NAME_COL in school_base_df.columns
                and config.SOURCE_COUNTRY_COL in school_base_df.columns
            ):
                school_country_map = school_base_df.set_index(config.SCHOOL_NAME_COL)[
                    config.SOURCE_COUNTRY_COL
                ].to_dict()
                combined_df[config.TARGET_COUNTRY_COL] = combined_df[config.TARGET_UNI_COL].map(
                    school_country_map
                )
        except Exception as e:
            if "case_lib_data_warnings" not in st.session_state:
                st.session_state["case_lib_data_warnings"] = []
            st.session_state["case_lib_data_warnings"].append(
                f"处理 school_base.feather 时出错: {e}"
            )

    columns_to_standardize = set()
    columns_to_standardize.update(
        [
            config.YEAR_COL,
            config.TARGET_COUNTRY_COL,
            config.ADMISSION_STATUS_COL,
            config.TARGET_UNI_COL,
            config.CATEGORY_COL,
            "就读专业1",
        ]
    )
    columns_to_standardize.update(config.BACKGROUND_UNI_COLS)
    columns_to_standardize.update(config.TARGET_MAJOR_COLS)

    for col in [c for c in columns_to_standardize if c in combined_df.columns]:
        if combined_df[col].dtype == "object":
            combined_df[col] = combined_df[col].astype(str).str.strip()
            mask = combined_df[col].isin(config.INVALID_VALUES + [""])
            combined_df.loc[mask, col] = None

    numeric_cols = ["IELTS分数", "TOEFL分数", "GRE分数", "GMAT分数", "IELTS", "TOEFL"]
    for col in [c for c in numeric_cols if c in combined_df.columns]:
        combined_df[col] = pd.to_numeric(combined_df[col], errors="coerce")

    category_candidate_cols = [
        config.YEAR_COL,
        config.TARGET_COUNTRY_COL,
        config.ADMISSION_STATUS_COL,
        config.TARGET_UNI_COL,
        "申请专业",
        "专业英文名称修正",
    ]
    for col in [c for c in category_candidate_cols if c in combined_df.columns]:
        try:
            combined_df[col] = combined_df[col].astype("category")
        except Exception:
            pass

    return combined_df
