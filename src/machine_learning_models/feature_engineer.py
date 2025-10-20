import numpy as np
import pandas as pd
from categorical_features_processor import prepare_categorical_columns
from data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    IRRELEVANT_COLUMNS,
    TEXT_COLUMNS,
)


class FeatureEngineer:
    def __init__(self):
        self.numeric_medians = {}
        self.cap_values = {}
        self.existing_categorical_columns = []

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        data = data.drop(columns=IRRELEVANT_COLUMNS, errors="ignore")

        for col in TEXT_COLUMNS:
            if col in data.columns and data[col].isnull().any():
                data[col] = data[col].fillna("")

        return data

    def _handle_numeric_missing(self, data: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        numeric_columns = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
        numeric_columns = [col for col in numeric_columns if col not in TEXT_COLUMNS]

        if is_fit:
            for col in numeric_columns:
                self.numeric_medians[col] = data[col].median()
        for col in numeric_columns:
            if col in self.numeric_medians and data[col].isnull().any():
                data[col] = data[col].fillna(self.numeric_medians[col])

        return data

    def _handle_count_columns(self, data: pd.DataFrame, is_fit: bool = False) -> pd.DataFrame:
        for col in COUNT_COLUMNS_FOR_LOG_TRANSFORM:
            if col in data.columns:
                numeric_col_series = pd.to_numeric(data[col], errors="coerce")

                if is_fit and numeric_col_series.notna().any():
                    cap_value = numeric_col_series.quantile(0.99)
                    if pd.notna(cap_value):
                        self.cap_values[col] = float(cap_value)

                if col in self.cap_values:
                    data[col] = numeric_col_series.clip(upper=self.cap_values[col])

                data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0)
                data[col] = np.log1p(data[col])

        return data

    def _handle_language_scores(self, data: pd.DataFrame) -> pd.DataFrame:
        toefl_exists = "toefl" in data.columns
        ielts_exists = "ielts" in data.columns

        if toefl_exists and ielts_exists:
            toefl_norm = pd.to_numeric(data["toefl"], errors="coerce").fillna(0) / 120
            ielts_norm = pd.to_numeric(data["ielts"], errors="coerce").fillna(0) / 9
            data["language_score"] = np.maximum(toefl_norm, ielts_norm)
            data.drop(columns=["toefl", "ielts"], inplace=True, errors="ignore")
        elif toefl_exists:
            data["language_score"] = pd.to_numeric(data["toefl"], errors="coerce").fillna(0) / 120
            data.drop(columns=["toefl"], inplace=True, errors="ignore")
        elif ielts_exists:
            data["language_score"] = pd.to_numeric(data["ielts"], errors="coerce").fillna(0) / 9
            data.drop(columns=["ielts"], inplace=True, errors="ignore")

        return data

    def fit(self, df: pd.DataFrame) -> "FeatureEngineer":
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=True)
        data = self._handle_count_columns(data, is_fit=True)
        data = self._handle_language_scores(data)
        self.existing_categorical_columns = [
            col for col in CATEGORICAL_COLUMNS if col in data.columns
        ]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=False)
        if self.existing_categorical_columns:
            data = prepare_categorical_columns(data, self.existing_categorical_columns)
        data = self._handle_count_columns(data, is_fit=False)
        data = self._handle_language_scores(data)
        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    engineer = FeatureEngineer()
    return engineer.fit_transform(df)
