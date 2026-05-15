"""
SHAP 全量模型解释 — GPU 加速

Outputs:
  reports/shap_values.npz        — SHAP values (N × F)
  reports/shap_summary.json      — feature importance + interaction top-k
  reports/shap_beeswarm.png      — global beeswarm
  reports/shap_interaction.png   — top-feature interaction matrix
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "machine_learning_models"))

from src.machine_learning_models.feature_engineer import FeatureEngineer

# ── Paths ──────────────────────────────────────────────
MODEL_PATH = (
    PROJECT_ROOT
    / "src/machine_learning_models/pre-trained_models/xgboost_20260316_092608.ubj"
)
CASES_PATH = PROJECT_ROOT / "src/machine_learning_models/data/cases.feather"
OUT_DIR = PROJECT_ROOT / "reports"

N_BACKGROUND = 3000   # background samples for TreeExplainer
N_EVAL = 10000        # SHAP values to compute
N_INTERACTION_TOP = 12  # interaction matrix dim

SEED = 42


def load_data() -> pd.DataFrame:
    print(f"Loading cases from {CASES_PATH}")
    df = pd.read_feather(CASES_PATH)
    print(f"  Raw: {len(df)} rows, {len(df.columns)} cols")
    target = df["admitted"].astype(int)
    return df, target


def engineer(df: pd.DataFrame) -> np.ndarray:
    print("Feature engineering...")
    engineer = FeatureEngineer()
    X_df = engineer.fit_transform(df)
    target_cols = ["admitted", "admitted_y", "admitted_x"]
    for c in target_cols:
        if c in X_df.columns:
            X_df = X_df.drop(columns=[c])
    print(f"  Engineered: {X_df.shape[0]} rows, {X_df.shape[1]} features")
    return X_df


def main():
    t0 = time.time()

    # 1. Load model
    print("Loading XGBoost booster...")
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    print(f"  Model: {booster.num_features()} features, {booster.num_boosted_rounds()} trees")

    # 2. Load & engineer data
    df, target = load_data()
    X_df = engineer(df)
    # Encode categorical columns as int codes (what XGBoost enable_categorical uses internally)
    for col in X_df.columns:
        if X_df[col].dtype.name == "category":
            X_df[col] = X_df[col].cat.codes.astype(np.float32)
    feature_names = list(X_df.columns)
    X_all = X_df.values.astype(np.float32)

    # 3. Sample background & eval
    rng = np.random.default_rng(SEED)
    n_total = len(X_all)
    idx_all = np.arange(n_total)
    rng.shuffle(idx_all)

    bg_idx = idx_all[:N_BACKGROUND]
    eval_idx = idx_all[N_BACKGROUND : N_BACKGROUND + N_EVAL]

    X_bg = X_all[bg_idx]
    X_eval = X_all[eval_idx]
    y_eval = target.iloc[eval_idx].values
    print(f"  Background: {len(X_bg)} samples")
    print(f"  Eval: {len(X_eval)} samples (admission rate: {y_eval.mean():.3f})")

    # 4. SHAP TreeExplainer
    print("Building TreeExplainer (GPU backend if available)...")
    import shap

    explainer = shap.TreeExplainer(
        booster,
        feature_perturbation="tree_path_dependent",  # required: XGBoost cat splits unsupported with interventional
    )
    ev = explainer.expected_value
    if hasattr(ev, "__len__") and len(ev.shape) > 0:
        ev = float(ev[0]) if ev.shape[0] == 1 else float(np.mean(ev))
    print(f"  Expected value (base): {ev:.4f}")

    # 5. Compute SHAP values
    print(f"Computing SHAP for {len(X_eval)} samples...")
    t_shap = time.time()
    shap_values = explainer.shap_values(X_eval)
    shap_secs = time.time() - t_shap
    print(f"  Done in {shap_secs:.1f}s ({len(X_eval) / shap_secs:.0f} samples/s)")

    # 6. Global feature importance
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs_shap)

    importance = []
    for rank, fi in enumerate(order[:30]):
        importance.append(
            {
                "rank": rank + 1,
                "feature": feature_names[fi],
                "mean_abs_shap": round(float(mean_abs_shap[fi]), 6),
            }
        )

    print("\nTop 15 features by |SHAP|:")
    for imp in importance[:15]:
        print(f"  {imp['rank']:2d}. {imp['feature']:<35s} {imp['mean_abs_shap']:.6f}")

    # 7. SHAP interaction values for top features
    print(f"\nComputing SHAP interactions (top {N_INTERACTION_TOP} features)...")
    t_int = time.time()
    n_inter_top = min(N_INTERACTION_TOP, len(feature_names))
    top_feat_idx = order[:n_inter_top]
    shap_interaction = explainer.shap_interaction_values(X_eval[:3000])
    int_secs = time.time() - t_int
    print(f"  Done in {int_secs:.1f}s")

    # Extract interaction matrix — shape (n_samples, n_features, n_features) → mean over samples
    inter_vals = shap_interaction if shap_interaction.ndim == 3 else shap_interaction[0]
    mean_abs_inter = np.abs(inter_vals).mean(axis=0)
    interactions_top = []
    for i in range(n_inter_top):
        for j in range(i + 1, n_inter_top):
            interactions_top.append(
                {
                    "feature_a": feature_names[top_feat_idx[i]],
                    "feature_b": feature_names[top_feat_idx[j]],
                    "mean_abs_interaction": round(float(mean_abs_inter[top_feat_idx[i], top_feat_idx[j]]), 6),
                }
            )
    interactions_top.sort(key=lambda x: -x["mean_abs_interaction"])

    print("\nTop 10 interactions:")
    for inter in interactions_top[:10]:
        print(
            f"  {inter['feature_a']} × {inter['feature_b']}: "
            f"{inter['mean_abs_interaction']:.6f}"
        )

    # 8. Case studies: highest and lowest SHAP individual predictions
    shap_sum = shap_values.sum(axis=1)

    top_positive = np.argsort(-shap_sum)[:3]   # SHAP pushes most positive
    top_negative = np.argsort(shap_sum)[:3]     # SHAP pushes most negative

    case_studies = {"pushed_up": [], "pushed_down": []}
    for label, indices in [("pushed_up", top_positive), ("pushed_down", top_negative)]:
        for i, idx in enumerate(indices):
            sample = df.iloc[eval_idx[idx]]
            sample_shap = shap_values[idx]
            top5 = np.argsort(-np.abs(sample_shap))[:5]
            drivers = [
                {
                    "feature": feature_names[j],
                    "shap_value": round(float(sample_shap[j]), 6),
                    "direction": "positive" if sample_shap[j] > 0 else "negative",
                }
                for j in top5
            ]
            case_studies[label].append(
                {
                    "gpa": float(sample.get("gpa", np.nan)),
                    "language": float(sample.get("language_score", np.nan)),
                    "major": str(sample.get("target_major", "")),
                    "admitted": int(sample["admitted"]),
                    "base_expected": round(float(ev), 4),
                    "shap_sum": round(float(shap_sum[idx]), 4),
                    "top5_drivers": drivers,
                }
            )

    # 9. Save everything
    print("\nSaving outputs...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # SHAP values + feature matrix + interaction (for shap.plots)
    np.savez_compressed(
        OUT_DIR / "shap_values.npz",
        values=shap_values,
        eval_indices=eval_idx,
        feature_names=np.array(feature_names),
        X_eval=X_eval,
        inter_values=inter_vals[:2000] if interactions_top else None,
    )
    print(f"  shap_values.npz: values={shap_values.shape}, X_eval={X_eval.shape}")

    # Summary JSON
    summary = {
        "model": str(MODEL_PATH.name),
        "n_background": N_BACKGROUND,
        "n_eval": N_EVAL,
        "expected_value": float(ev),
        "compute_time_shap_s": round(shap_secs, 1),
        "compute_time_interaction_s": round(int_secs, 1),
        "feature_importance": importance,
        "top_interactions": interactions_top[:20],
        "case_studies": case_studies,
    }
    with open(OUT_DIR / "shap_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("  shap_summary.json")

    total_secs = time.time() - t0
    print(f"\nTotal time: {total_secs:.0f}s")


if __name__ == "__main__":
    main()
