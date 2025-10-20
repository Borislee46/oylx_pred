import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

MATRIX_PATHS = {
    "correlation": "src/machine_learning_models/data/correlation_matrix.feather",
    "pair_weight": "src/machine_learning_models/data/correlation_pair_weight.feather",
    "overlap_count": "src/machine_learning_models/data/correlation_overlap_count.feather",
}


def _build_cache_key(file_path: str) -> str:
    mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    return f"{file_path}|{mtime}|np={np.__version__}|pd={pd.__version__}"


def _validate_matrix_format(df: pd.DataFrame) -> bool:
    row_ids = list(map(str, df.index))
    col_ids = list(map(str, df.columns))
    return row_ids == col_ids


@st.cache_data
def _load_feather_matrix_cached(file_path: str, cache_key: str) -> Optional[pd.DataFrame]:
    try:
        if not os.path.exists(file_path):
            logger.warning(f"矩阵文件 '{file_path}' 不存在。")
            return None

        df = pd.read_feather(file_path)

        if "index" in df.columns:
            df.index = pd.Index(df["index"].astype(str))
            df = df.drop(columns=["index"])

        if not _validate_matrix_format(df):
            logger.error(f"矩阵文件 '{file_path}' 格式不正确。")
            return None

        return df

    except Exception as e:
        logger.error(f"加载 Feather 文件 '{file_path}' 时出错: {e}")
        return None


def _load_matrix(file_path: str) -> Optional[pd.DataFrame]:
    cache_key = _build_cache_key(file_path)
    return _load_feather_matrix_cached(file_path, cache_key)


def load_correlation_matrix() -> Optional[pd.DataFrame]:
    return _load_matrix(MATRIX_PATHS["correlation"])


def load_correlation_pair_weight() -> Optional[pd.DataFrame]:
    return _load_matrix(MATRIX_PATHS["pair_weight"])


def load_correlation_overlap_count() -> Optional[pd.DataFrame]:
    return _load_matrix(MATRIX_PATHS["overlap_count"])


def load_correlation_and_weight() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    return load_correlation_matrix(), load_correlation_pair_weight()
