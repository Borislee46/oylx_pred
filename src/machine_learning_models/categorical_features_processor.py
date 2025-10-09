import pandas as pd


def prepare_categorical_columns(df, columns):
    for col in columns:
        series = df[col]
        if not pd.api.types.is_categorical_dtype(series):
            df[col] = series.astype("category")
    return df
