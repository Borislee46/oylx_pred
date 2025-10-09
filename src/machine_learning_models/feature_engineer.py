import numpy as np
import pandas as pd
from categorical_features_processor import prepare_categorical_columns
from data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    IRRELEVANT_COLUMNS,
    TEXT_COLUMNS,
)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    data = data.drop(columns=IRRELEVANT_COLUMNS, errors="ignore")

    if data.isnull().any().any():
        for col in TEXT_COLUMNS:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna("")

        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in TEXT_COLUMNS]

        nan_numeric_columns = [col for col in numeric_columns if data[col].isnull().any()]

        if nan_numeric_columns:
            try:
                for col in nan_numeric_columns:
                    data[col] = data[col].interpolate(method="nearest")
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(method="ffill")
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(method="bfill")
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(data[col].median())
            except Exception as e:
                for col in nan_numeric_columns:
                    data[col] = data[col].fillna(data[col].median())
                    if data[col].isnull().any():
                        data[col] = data[col].fillna(0)

    existing_categorical_columns = [col for col in CATEGORICAL_COLUMNS if col in data.columns]

    if existing_categorical_columns:
        data = prepare_categorical_columns(data, existing_categorical_columns)

    for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
        if col in data.columns:
            numeric_col_series = pd.to_numeric(data[col], errors="coerce")
            if numeric_col_series.notna().any():
                cap_value = numeric_col_series.quantile(0.99)
                if pd.notna(cap_value):
                    current_max = numeric_col_series.max()
                    if pd.notna(current_max) and cap_value < current_max:
                        data[col] = numeric_col_series.clip(upper=cap_value)

    for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
            data[col] = np.log1p(data[col])

    if "toefl" in data.columns and "ielts" in data.columns:
        data["toefl_norm"] = data["toefl"].apply(lambda x: x / 120 if pd.notna(x) and x > 0 else 0)
        data["ielts_norm"] = data["ielts"].apply(lambda x: x / 9 if pd.notna(x) and x > 0 else 0)
        data["language_score"] = np.maximum(data["toefl_norm"], data["ielts_norm"])
        data.drop(
            columns=["toefl", "ielts", "toefl_norm", "ielts_norm"], inplace=True, errors="ignore"
        )

    elif "toefl" in data.columns:
        data["language_score"] = data["toefl"].apply(
            lambda x: x / 120 if pd.notna(x) and x > 0 else 0
        )
        data.drop(columns=["toefl"], inplace=True, errors="ignore")

    elif "ielts" in data.columns:
        data["language_score"] = data["ielts"].apply(
            lambda x: x / 9 if pd.notna(x) and x > 0 else 0
        )
        data.drop(columns=["ielts"], inplace=True, errors="ignore")

    return data


class FeatureEngineer:
    def __init__(self):
        self.numeric_medians = {}
        self.cap_values = {}
        self.existing_categorical_columns = []
        self.columns_seen = None

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        data = df.copy()

        data = data.drop(columns=IRRELEVANT_COLUMNS, errors="ignore")

        for col in TEXT_COLUMNS:
            if col in data.columns:
                if data[col].isnull().any():
                    pass

        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in TEXT_COLUMNS]

        for col in numeric_columns:
            series = data[col]
            if series.isnull().any():
                self.numeric_medians[col] = series.median()

        for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            if col in data.columns:
                numeric_col_series = pd.to_numeric(data[col], errors="coerce")
                if numeric_col_series.notna().any():
                    cap_value = numeric_col_series.quantile(0.99)
                    if pd.notna(cap_value):
                        self.cap_values[col] = float(cap_value)

        self.existing_categorical_columns = [
            col for col in CATEGORICAL_COLUMNS if col in data.columns
        ]
        self.columns_seen = data.columns.tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()

        data = data.drop(columns=IRRELEVANT_COLUMNS, errors="ignore")

        for col in TEXT_COLUMNS:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna("")

        for col, med in self.numeric_medians.items():
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col] = data[col].fillna(med)

        if self.existing_categorical_columns:
            data = prepare_categorical_columns(data, self.existing_categorical_columns)

        for col, cap in self.cap_values.items():
            if col in data.columns:
                numeric_col_series = pd.to_numeric(data[col], errors="coerce")
                data[col] = numeric_col_series.clip(upper=cap)

        for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
                data[col] = np.log1p(data[col])

        if "toefl" in data.columns and "ielts" in data.columns:
            data["toefl_norm"] = data["toefl"].apply(
                lambda x: x / 120 if pd.notna(x) and x > 0 else 0
            )
            data["ielts_norm"] = data["ielts"].apply(
                lambda x: x / 9 if pd.notna(x) and x > 0 else 0
            )
            data["language_score"] = np.maximum(data["toefl_norm"], data["ielts_norm"])
            data.drop(
                columns=["toefl", "ielts", "toefl_norm", "ielts_norm"],
                inplace=True,
                errors="ignore",
            )
        elif "toefl" in data.columns:
            data["language_score"] = data["toefl"].apply(
                lambda x: x / 120 if pd.notna(x) and x > 0 else 0
            )
            data.drop(columns=["toefl"], inplace=True, errors="ignore")
        elif "ielts" in data.columns:
            data["language_score"] = data["ielts"].apply(
                lambda x: x / 9 if pd.notna(x) and x > 0 else 0
            )
            data.drop(columns=["ielts"], inplace=True, errors="ignore")

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
