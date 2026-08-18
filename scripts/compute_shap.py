"""
Compute SHAP values for the production XGBoost model.

Generates reports/v10_shap_explanation/shap_values.npz and
reports/v10_shap_explanation/shap_summary.json,
consumed by reports/v10_shap_explanation/run_shap_analysis.py.

Method: shap.TreeExplainer with tree_path_dependent (required because
the model uses enable_categorical=True).

Usage:
    python scripts/compute_shap.py
    python scripts/compute_shap.py --n-eval 5000 --n-interaction 1000
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import shap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ml"))

N_EVAL = 10000
N_INTERACTION = 2000
RANDOM_SEED = 42


def load_newest_model() -> tuple[object, str]:
    """Load newest XGBoost booster from pre-trained_models/."""
    import xgboost as xgb

    model_dir = PROJECT_ROOT / "src" / "ml" / "pre-trained_models"
    candidates = sorted(
        model_dir.glob("xgboost_*.ubj.gz"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            model_dir.glob("xgboost_*.ubj"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise FileNotFoundError(f"No xgboost model found in {model_dir}")

    path = str(candidates[0])
    booster = xgb.Booster()
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            booster.load_model(bytearray(f.read()))
    else:
        booster.load_model(path)

    print(f"Loaded model: {Path(path).name}")
    return booster, Path(path).stem


def load_evaluation_data(n_eval: int) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Load training data, apply FeatureEngineer, sample n_eval rows.

    Returns: X_eval (np array), eval_indices, feature_names, df_sample.
    """
    import pandas as pd

    from src.ml.data_config import IRRELEVANT_COLUMNS, TARGET_COLUMN
    from src.ml.feature_engineer import FeatureEngineer

    data_path = PROJECT_ROOT / "src" / "ml" / "data" / "cases.feather"
    df = pd.read_feather(data_path)
    print(f"Training data: {len(df)} rows")

    engineer = FeatureEngineer()
    df_eng = engineer.fit_transform(df)
    print(f"Engineered features: {len(df_eng.columns)} columns")

    feature_cols = [c for c in df_eng.columns if c not in IRRELEVANT_COLUMNS and c != TARGET_COLUMN]
    X = df_eng[feature_cols].copy()
    feature_names = list(X.columns)
    print(f"Feature columns ({len(feature_names)}): {feature_names}")

    # Sample evaluation rows
    rng = np.random.default_rng(RANDOM_SEED)
    n_total = len(df)
    eval_indices = rng.choice(n_total, size=min(n_eval, n_total), replace=False)

    # Build X_eval: categorical as integer codes, numerical as float64
    X_subset = X.iloc[eval_indices]
    col_arrays = []
    for c in feature_names:
        col_dtype = str(X_subset[c].dtype)
        if col_dtype == "category":
            col_arrays.append(X_subset[c].cat.codes.values.astype(np.float64))
        else:
            col_arrays.append(X_subset[c].values.astype(np.float64))
    X_eval = np.column_stack(col_arrays)
    print(f"Evaluation data: {X_eval.shape}")

    return X_eval, eval_indices.tolist(), feature_names, df.iloc[eval_indices]


def main():
    parser = argparse.ArgumentParser(description="Compute SHAP values for XGBoost model")
    parser.add_argument("--n-eval", type=int, default=N_EVAL)
    parser.add_argument("--n-interaction", type=int, default=N_INTERACTION)
    args = parser.parse_args()

    # ── Load model ─────────────────────────────────────────────────
    booster, model_name = load_newest_model()

    # ── Load evaluation data ───────────────────────────────────────
    X_eval, eval_indices, feature_names, df_sample = load_evaluation_data(args.n_eval)

    # ── SHAP TreeExplainer (tree_path_dependent) ───────────────────
    print(f"\nComputing SHAP values for {len(X_eval)} samples × {len(feature_names)} features...")
    t0 = time.time()

    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",
    )
    expected_value_raw = explainer.expected_value
    if hasattr(expected_value_raw, "__len__") and not isinstance(expected_value_raw, float):
        expected_value = float(np.asarray(expected_value_raw).flat[0])
    else:
        expected_value = float(expected_value_raw)
    print(f"  Expected value (log-odds): {expected_value:.6f}")

    shap_values = explainer.shap_values(X_eval)
    print(f"  SHAP values shape: {shap_values.shape}")

    elapsed_shap = time.time() - t0
    print(f"  Compute time: {elapsed_shap:.1f}s")

    # ── Interactions (subset) ──────────────────────────────────────
    n_interact = min(args.n_interaction, len(X_eval))
    print(f"\nComputing SHAP interactions for {n_interact} samples...")
    t1 = time.time()

    X_interact = X_eval[:n_interact]
    shap_interactions = explainer.shap_interaction_values(X_interact)
    print(f"  Interactions shape: {shap_interactions.shape}")

    elapsed_interact = time.time() - t1
    print(f"  Interaction compute time: {elapsed_interact:.1f}s")

    # ── Feature importance ranking ─────────────────────────────────
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs_shap)
    feature_importance = []
    for rank, idx in enumerate(order, 1):
        feature_importance.append(
            {
                "rank": rank,
                "feature": feature_names[idx],
                "mean_abs_shap": round(float(mean_abs_shap[idx]), 6),
            }
        )

    categorical_share = (
        sum(
            fi["mean_abs_shap"]
            for fi in feature_importance
            if any(k in fi["feature"] for k in ["university", "major"])
        )
        / mean_abs_shap.sum()
        * 100
    )
    print("\n  Feature importance:")
    for fi in feature_importance:
        bar = "█" * int(fi["mean_abs_shap"] / mean_abs_shap.max() * 30)
        print(f"    {fi['rank']:2d}. {fi['feature']:<25s} {fi['mean_abs_shap']:.4f}  {bar}")
    print(f"  Categorical share: {categorical_share:.1f}%")

    # ── Top interactions ───────────────────────────────────────────
    mean_abs_interaction = np.abs(shap_interactions).mean(axis=0)
    np.fill_diagonal(mean_abs_interaction, 0)
    top_n = 15
    top_pairs = []
    for i in range(len(feature_names)):
        for j in range(i + 1, len(feature_names)):
            top_pairs.append((mean_abs_interaction[i, j], i, j))
    top_pairs.sort(reverse=True)

    top_interactions = []
    for val, i, j in top_pairs[:top_n]:
        top_interactions.append(
            {
                "rank": len(top_interactions) + 1,
                "pair": [feature_names[i], feature_names[j]],
                "mean_abs_interaction": round(float(val), 6),
            }
        )
    print("\n  Top interactions:")
    for ti in top_interactions[:5]:
        print(f"    {ti['pair'][0]} × {ti['pair'][1]}: {ti['mean_abs_interaction']:.4f}")

    # ── Save ───────────────────────────────────────────────────────
    npz_path = PROJECT_ROOT / "reports" / "shap_values.npz"
    np.savez_compressed(
        npz_path,
        values=shap_values,
        feature_names=np.array(feature_names),
        X_eval=X_eval,
        eval_indices=np.array(eval_indices),
    )
    print(f"\n[OK] {npz_path}  ({npz_path.stat().st_size / 1024:.0f} KB)")

    summary = {
        "model": model_name,
        "n_background": 0,  # tree_path_dependent uses training data implicitly
        "n_eval": len(X_eval),
        "expected_value": round(expected_value, 6),
        "compute_time_shap_s": round(elapsed_shap, 1),
        "compute_time_interaction_s": round(elapsed_interact, 1),
        "feature_importance": feature_importance,
        "top_interactions": top_interactions,
    }
    summary_path = PROJECT_ROOT / "reports" / "shap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] {summary_path}")

    print(f"\nDone. Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
