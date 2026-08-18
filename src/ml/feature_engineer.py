from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data_config import (
    CATEGORICAL_COLUMNS,
    COUNT_COLUMNS_FOR_LOG_TRANSFORM,
    IRRELEVANT_COLUMNS,
    TEXT_COLUMNS,
)


class FeatureEngineer:
    def __init__(
        self,
        use_internalized_features: bool = False,
        use_target_encoding: bool = False,
    ):
        self.numeric_medians = {}
        self.cap_values = {}
        self.existing_categorical_columns = []
        self.categorical_levels = {}

        self.use_internalized_features = use_internalized_features

        self.use_target_encoding = use_target_encoding
        self._te_active: bool = False
        self._te = None

        if use_target_encoding:
            from .te_encoder import TargetEncoder

            self._te = TargetEncoder()

        if use_internalized_features:
            from ._shared_features import (
                compute_bg_school_score,
                compute_cross_faculty_features,
            )

            self._compute_cross_faculty = compute_cross_faculty_features
            self._compute_bg_school = compute_bg_school_score

            self._sim_lookup: dict[tuple[str, str], float] = {}

            config_dir = Path(__file__).resolve().parents[2] / "config"

            self._school_score_map: dict[str, float] = {}
            tiers_path = config_dir / "school_tiers.json"
            if tiers_path.exists():
                with open(tiers_path, encoding="utf-8") as f:
                    tiers = json.load(f)
                scores = tiers.get("scores", {})
                for s in tiers.get("c9", []):
                    self._school_score_map[s] = scores.get("c9", 1.00)
                for s in tiers.get("985", []):
                    self._school_score_map[s] = scores.get("985", 0.85)
                for s in tiers.get("211", []):
                    self._school_score_map[s] = scores.get("211", 0.65)
                self._school_score_map["_default"] = scores.get("other", 0.50)

            self._faculty_whitelist: dict[str, list[str]] = {}
            self._faculty_severity: dict[tuple[str, str], float] = {}
            rules_path = config_dir / "cross_faculty_rules.json"
            if rules_path.exists():
                with open(rules_path, encoding="utf-8") as f:
                    rules = json.load(f)
                self._faculty_whitelist = rules.get("whitelist", {})
                severity = rules.get("severity", {})
                for level, data in severity.items():
                    if not isinstance(data, dict) or level.startswith("_"):
                        continue
                    mult = {"light": 0.70, "medium": 0.50, "heavy": 0.30}.get(level, 0.30)
                    for pair in data.get("pairs", []):
                        self._faculty_severity[(pair[0], pair[1])] = mult

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        irrelevant = list(IRRELEVANT_COLUMNS)
        if self.use_target_encoding and "faculty" in irrelevant:
            irrelevant.remove("faculty")
        data = data.drop(columns=irrelevant, errors="ignore")

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

    def _handle_categorical_alignment(
        self, data: pd.DataFrame, is_fit: bool = False
    ) -> pd.DataFrame:
        for col in self.existing_categorical_columns:
            if col not in data.columns:
                continue

            if is_fit:
                if not pd.api.types.is_categorical_dtype(data[col]):
                    data[col] = data[col].astype("category")
                self.categorical_levels[col] = data[col].cat.categories
            else:
                data[col] = pd.Categorical(
                    data[col], categories=self.categorical_levels.get(col), ordered=False
                )
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

    def load_similarity_cache(self, sim_cache_df) -> None:
        if not self.use_internalized_features:
            return
        if sim_cache_df is not None and not sim_cache_df.empty:
            for _, row in sim_cache_df.iterrows():
                bg = str(row.get("bg_major", "")).strip().lower()
                tgt = str(row.get("target_major", "")).strip().lower()
                sim = float(row.get("similarity", 0.5))
                self._sim_lookup[(bg, tgt)] = sim

    def fit(self, df: pd.DataFrame) -> FeatureEngineer:
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=True)
        data = self._handle_count_columns(data, is_fit=True)
        data = self._handle_language_scores(data)
        self.existing_categorical_columns = [
            col for col in CATEGORICAL_COLUMNS if col in data.columns
        ]
        self._handle_categorical_alignment(data, is_fit=True)

        if self.use_target_encoding and self._te is not None:
            self._te.fit(data, target_col="admitted")

        return self

    def transform(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        data = self._preprocess_data(df)
        data = self._handle_numeric_missing(data, is_fit=False)
        data = self._handle_count_columns(data, is_fit=False)
        data = self._handle_language_scores(data)

        if self.use_target_encoding and self._te is not None:
            if self.use_internalized_features:
                from ._shared_features import get_major_similarity

                if "background_major" in data.columns and "target_major" in data.columns:
                    data["major_similarity"] = data.apply(
                        lambda r: get_major_similarity(
                            str(r.get("background_major", "")),
                            str(r.get("target_major", "")),
                            self._sim_lookup,
                        ),
                        axis=1,
                    )
                data = self._compute_cross_faculty(
                    data, self._faculty_whitelist, self._faculty_severity
                )
                if "background_university" in data.columns:
                    data = self._compute_bg_school(data, self._school_score_map)
                else:
                    data["bg_school_score"] = 0.50
                data["school_level_gap"] = 0

            te_cols_present = [c for c in self._te.columns if c in data.columns]

            if is_train and "admitted" in data.columns:
                te_df = self._te.transform_train(data, target_col="admitted")
            else:
                te_df = self._te.transform(data)

            data = data.drop(columns=te_cols_present, errors="ignore")
            for col in CATEGORICAL_COLUMNS:
                if col in data.columns:
                    data = data.drop(columns=[col], errors="ignore")

            data = pd.concat([data.reset_index(drop=True), te_df.reset_index(drop=True)], axis=1)
            self._te_active = True
        else:
            data = self._handle_categorical_alignment(data, is_fit=False)
            self._te_active = False

            if self.use_internalized_features:
                from ._shared_features import get_major_similarity

                if "background_major" in data.columns and "target_major" in data.columns:
                    data["major_similarity"] = data.apply(
                        lambda r: get_major_similarity(
                            str(r.get("background_major", "")),
                            str(r.get("target_major", "")),
                            self._sim_lookup,
                        ),
                        axis=1,
                    )
                data = self._compute_cross_faculty(
                    data, self._faculty_whitelist, self._faculty_severity
                )
                if "background_university" in data.columns:
                    data = self._compute_bg_school(data, self._school_score_map)
                else:
                    data["bg_school_score"] = 0.50
                data["school_level_gap"] = 0

        return data

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df, is_train=True)

    def get_state(self) -> dict:
        state = {
            "numeric_medians": {k: float(v) for k, v in self.numeric_medians.items()},
            "cap_values": {k: float(v) for k, v in self.cap_values.items()},
            "existing_categorical_columns": list(self.existing_categorical_columns),
            "categorical_levels": {
                col: [str(level) for level in levels.tolist()]
                for col, levels in self.categorical_levels.items()
            },
        }
        if self.use_target_encoding and self._te is not None:
            state["te_state"] = self._te.get_state()
        return state
