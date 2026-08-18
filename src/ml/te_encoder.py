from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.utils.numeric import sigmoid_k

_DEFAULT_TE_COLUMNS = [
    "target_university",
    "target_major",
    "background_university",
    "background_major",
    "faculty",
]
_DEFAULT_TE_K = 10
_DEFAULT_TE_S = 5
_DEFAULT_TE_FOLDS = 5
_DEFAULT_RANDOM_STATE = 42


def sigmoid_shrinkage(n: int, k: int = _DEFAULT_TE_K, s: float = _DEFAULT_TE_S) -> float:
    return sigmoid_k(float(n), 1.0 / s, float(k))


def compute_category_stats(df: pd.DataFrame, col: str, target: str) -> dict[str, dict]:
    stats = {}
    for cat, group in df.groupby(col, observed=False):
        n = len(group)
        pos = int(group[target].sum())
        stats[str(cat)] = {"n": n, "pos": pos, "mean": pos / n if n > 0 else 0.0}
    return stats


class TargetEncoder:
    def __init__(
        self,
        columns=None,
        k=_DEFAULT_TE_K,
        s=_DEFAULT_TE_S,
        cv_folds=_DEFAULT_TE_FOLDS,
        random_state=_DEFAULT_RANDOM_STATE,
    ):
        self.columns = columns or list(_DEFAULT_TE_COLUMNS)
        self.k = k
        self.s = s
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.stats: dict[str, dict[str, dict]] = {}  # col → {cat → {n, pos, mean}}
        self.global_mean: float = 0.0
        self.feature_names: list[str] = []

    def fit(self, df: pd.DataFrame, target_col: str = "admitted") -> TargetEncoder:
        self.global_mean = float(df[target_col].mean())
        for col in self.columns:
            if col in df.columns:
                self.stats[col] = compute_category_stats(df, col, target_col)
        return self

    def transform_train(self, df: pd.DataFrame, target_col: str = "admitted") -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        y = df[target_col]

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for col in self.columns:
            if col not in df.columns:
                continue

            col_name = f"te_{col}"
            result[col_name] = self.global_mean

            for train_idx, val_idx in skf.split(df, y):
                fold_df = df.iloc[train_idx]
                fold_stats = compute_category_stats(fold_df, col, target_col)

                val_series = pd.Series(self.global_mean, index=val_idx, dtype=float)
                for cat, info in fold_stats.items():
                    mask = df.iloc[val_idx][col].astype(str) == cat
                    if mask.any():
                        shrinkage = sigmoid_shrinkage(info["n"], self.k, self.s)
                        te_value = shrinkage * info["mean"] + (1 - shrinkage) * self.global_mean
                        val_series[mask.values] = te_value

                result.iloc[val_idx, result.columns.get_loc(col_name)] = val_series.values

        self.feature_names = list(result.columns)
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)
        for col in self.columns:
            if col in df.columns:
                result[f"te_{col}"] = self._encode_column(df, col)
        self.feature_names = list(result.columns)
        return result

    def _encode_column(self, df: pd.DataFrame, col: str) -> pd.Series:
        stats = self.stats.get(col, {})
        encoded = pd.Series(self.global_mean, index=df.index, dtype=float)

        for cat, info in stats.items():
            mask = df[col].astype(str) == cat
            if mask.any():
                shrinkage = sigmoid_shrinkage(info["n"], self.k, self.s)
                te_value = shrinkage * info["mean"] + (1 - shrinkage) * self.global_mean
                encoded[mask] = te_value

        known = df[col].astype(str).isin(stats.keys())
        encoded[~known] = self.global_mean

        return encoded

    def get_state(self) -> dict:
        return {
            "columns": self.columns,
            "k": self.k,
            "s": self.s,
            "global_mean": self.global_mean,
            "stats": {
                col: {
                    cat: {"n": info["n"], "pos": info["pos"], "mean": info["mean"]}
                    for cat, info in col_stats.items()
                }
                for col, col_stats in self.stats.items()
            },
        }

    def load_state(self, state: dict) -> TargetEncoder:
        self.columns = state.get("columns", self.columns)
        self.k = state.get("k", self.k)
        self.s = state.get("s", self.s)
        self.global_mean = state.get("global_mean", 0.0)
        self.stats = state.get("stats", {})
        return self
