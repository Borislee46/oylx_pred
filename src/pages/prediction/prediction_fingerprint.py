import pandas as pd

from src.utils.logger import setup_logger

prediction_handler_logger = setup_logger("page3", "prediction")


def compute_list_fingerprint(lst: list[str]) -> tuple[int, int]:
    if not lst:
        return (0, 0)
    try:
        sorted_list = sorted(lst)

        list_tuple = tuple(sorted_list)
        return (len(lst), hash(list_tuple))
    except Exception as e:
        prediction_handler_logger.warning(f"计算列表指纹失败: {e}")
        return (len(lst), 0)


def compute_df_fingerprint(df: pd.DataFrame | None) -> int:
    if df is None or df.empty:
        return 0

    try:
        from pandas.util import hash_pandas_object

        key_cols = [
            c
            for c in ["background_university", "target_university", "target_major"]
            if c in df.columns
        ]

        if key_cols:
            return int(hash_pandas_object(df[key_cols]).sum())

        return len(df)
    except Exception as e:
        prediction_handler_logger.warning(f"计算DataFrame指纹失败: {e}")
        return len(df)
