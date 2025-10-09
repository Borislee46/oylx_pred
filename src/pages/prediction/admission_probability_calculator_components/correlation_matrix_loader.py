import os

import numpy as np
import pandas as pd
import streamlit as st

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")


def _build_cache_key(file_path: str) -> str:
    try:
        mtime = os.path.getmtime(file_path) if os.path.exists(file_path) else 0
    except Exception:
        mtime = 0
    return f"{mtime}|np={np.__version__}|pd={pd.__version__}"


@st.cache_data
def _load_correlation_matrix_cached(correlation_matrix_path: str, cache_key: str):
    try:
        absolute_matrix_path = correlation_matrix_path
        if os.path.exists(absolute_matrix_path):
            try:
                df_pd = pd.read_feather(absolute_matrix_path)
                has_index = "index" in df_pd.columns
                if has_index:
                    index_vals = df_pd["index"].astype(str).tolist()
                    df_pd = df_pd.drop(columns=["index"])
                    try:
                        df_pd.index = pd.Index(index_vals, dtype=str)
                    except Exception:
                        df_pd.index = pd.Index([str(i) for i in index_vals])

                if isinstance(df_pd, pd.DataFrame) and list(map(str, df_pd.columns)) == list(
                    map(str, df_pd.index)
                ):
                    return df_pd

                logger.error(f"相关系数矩阵文件 '{correlation_matrix_path}' 格式不正确。")
                return None
            except Exception as e:
                logger.error(f"加载 Feather 文件时出错: {e}")
                return None
        else:
            logger.warning(f"相关系数矩阵文件 '{correlation_matrix_path}' 不存在。")
            return None
    except Exception as e:
        logger.error(f"加载相关系数矩阵时出错: {e}", exc_info=True)
        return None


def load_correlation_matrix():
    correlation_matrix_path = "src/machine_learning_models/data/correlation_matrix.feather"

    cache_key = _build_cache_key(correlation_matrix_path)
    return _load_correlation_matrix_cached(correlation_matrix_path, cache_key)


def load_correlation_pair_weight():
    weight_matrix_path = "src/machine_learning_models/data/correlation_pair_weight.feather"
    cache_key = _build_cache_key(weight_matrix_path)
    return _load_correlation_matrix_cached(weight_matrix_path, cache_key)


def load_correlation_overlap_count():
    overlap_matrix_path = "src/machine_learning_models/data/correlation_overlap_count.feather"
    cache_key = _build_cache_key(overlap_matrix_path)
    return _load_correlation_matrix_cached(overlap_matrix_path, cache_key)


def load_correlation_and_weight():
    corr = load_correlation_matrix()
    weight = load_correlation_pair_weight()
    return corr, weight
