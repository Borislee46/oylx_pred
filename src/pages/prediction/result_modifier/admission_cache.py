from typing import Any

import pandas as pd
import streamlit as st

from src.pages.prediction.result_modifier.utils import compute_dataframe_hash
from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


@st.cache_data
def get_admitted_combinations_for_major(
    cases_df_hash: str, cases_df_tuple: tuple[tuple[Any, ...], ...], background_major: str
) -> set[tuple[str, str]]:
    if not cases_df_tuple or not cases_df_hash:
        return set()

    try:
        bg_major_clean = str(background_major).strip()
        admitted_combinations = set()
        for row in cases_df_tuple:
            if len(row) >= 4 and row[0] == 1 and row[3] == bg_major_clean:
                admitted_combinations.add((str(row[1]), str(row[2])))
        return admitted_combinations
    except (IndexError, TypeError, ValueError) as e:
        logger.warning(f"解析录取组合失败: {str(e)}")
        return set()
    except Exception as e:
        logger.error(f"获取录取组合时发生未知错误: {str(e)}", exc_info=True)
        return set()


def get_admitted_combinations_from_dataframe(
    cases_df: pd.DataFrame, background_major: str
) -> set[tuple[str, str]]:
    if cases_df is None or cases_df.empty:
        return set()

    required_cols = [
        "admitted",
        "target_university",
        "target_major",
        "background_major",
    ]
    if not all(col in cases_df.columns for col in required_cols):
        logger.warning("数据框缺少必需的列")
        return set()

    try:
        df_hash = compute_dataframe_hash(cases_df[required_cols])
        cases_df_tuple = tuple(cases_df[required_cols].itertuples(index=False, name=None))
        return get_admitted_combinations_for_major(df_hash, cases_df_tuple, background_major)
    except Exception as e:
        logger.error(f"从DataFrame获取录取组合失败: {str(e)}", exc_info=True)
        return set()
