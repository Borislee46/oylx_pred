"""WritePrint — Ridge model fitting, caching, and reporting."""

import threading
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_absolute_error

from src.utils.logger import setup_logger
from src.pages.write_print.features import (
    _ensure_nltk_data,
    compute_features,
    extract_text_from_pdf,
    clean_turnitin_report,
    parse_filename_score,
    FEATURE_NAMES,
    SAMPLE_DIR,
)

_log = setup_logger("page3", "write_print")


def fit_model(sample_dir=SAMPLE_DIR):
    """Fit Ridge model on sample PDFs, return (model, scaler, X, y, names)."""
    t0 = time.perf_counter()
    pdf_files = sorted(sample_dir.glob("*.pdf"))
    features_list, scores_list, names_list, texts_list = [], [], [], []

    for pdf_path in pdf_files:
        expected_score = parse_filename_score(pdf_path.name)
        if expected_score is None:
            continue
        raw = extract_text_from_pdf(str(pdf_path))
        text = clean_turnitin_report(raw)
        if len(text) < 200:
            continue
        features_list.append(compute_features(text))
        scores_list.append(expected_score)
        names_list.append(pdf_path.name)
        texts_list.append(text)

    X = np.array(features_list)
    y = np.array(scores_list, dtype=float)

    if len(y) < 3:
        raise ValueError(
            f"Need at least 3 training samples, found {len(y)}. "
            f"Check that {sample_dir} has PDFs with scores in filenames "
            f"(e.g. 'essay-50%.pdf')."
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    alphas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    best_alpha, best_mae = alphas[0], float('inf')
    loo = LeaveOneOut()

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        cv_scores = cross_val_score(
            ridge, X_scaled, y, cv=loo, scoring='neg_mean_absolute_error',
        )
        mae = -cv_scores.mean()
        if mae < best_mae:
            best_mae, best_alpha = mae, alpha

    model = Ridge(alpha=best_alpha)
    model.fit(X_scaled, y)

    _learn_impacts(texts_list, y, scaler, model)

    elapsed = (time.perf_counter() - t0) * 1000
    _log.info(
        f"MODEL INIT | samples={len(y)} features={X.shape[1]} "
        f"alpha={model.alpha} loo_mae={best_mae:.1f}% "
        f"elapsed={elapsed:.0f}ms"
    )
    return model, scaler, X, y, names_list


_model_cache = None
_scaler_cache = None
_cache_lock = threading.Lock()


def get_model(sample_dir=SAMPLE_DIR):
    """Return (model, scaler). Thread-safe, fits once. Wrapped by @st.cache_resource."""
    global _model_cache, _scaler_cache
    _ensure_nltk_data()
    if _model_cache is None:
        with _cache_lock:
            if _model_cache is None:
                _model_cache, _scaler_cache, _, _, _ = fit_model(sample_dir)
    return _model_cache, _scaler_cache


# ── per-category impact weight learning ─────────────────────────────────────

_IMPACT_WEIGHTS = None


def _learn_impacts(texts: list, actual_scores, scaler, model):
    global _IMPACT_WEIGHTS
    from collections import defaultdict
    from src.pages.write_print.engine import analyze

    cat_residuals = defaultdict(list)

    for text, actual in zip(texts, actual_scores):
        result = analyze(text, scaler, model)
        if 'error' in result:
            continue
        residual = actual - result['score']

        cat_counts = defaultdict(int)
        for fix in result['local_fixes']:
            cat_counts[fix['category']] += 1

        for cat, count in cat_counts.items():
            if count > 0:
                cat_residuals[cat].append(residual / max(count, 1))

    weights = {}
    for cat, residuals in cat_residuals.items():
        if residuals:
            weights[cat] = sum(residuals) / len(residuals)

    _IMPACT_WEIGHTS = weights
    return weights


def get_impact_weights():
    return _IMPACT_WEIGHTS


def print_fit_report(model, scaler, X, y, names_list):
    """Print detailed fitting report."""
    X_scaled = scaler.transform(X)
    y_pred = model.predict(X_scaled)
    train_mae = mean_absolute_error(y, y_pred)
    n = len(y)

    loo = LeaveOneOut()
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=loo, scoring='neg_mean_absolute_error',
    )
    loo_mae = -cv_scores.mean()

    print(f"\n{'='*70}")
    print(f"FITTED SCORES  (Ridge alpha={model.alpha}, "
          f"LOO-CV MAE={loo_mae:.1f}%, Train MAE={train_mae:.1f}%)")
    print(f"{'='*70}")
    print(f"{'File':<50} {'Actual':>6}  {'Fitted':>6}  {'Delta':>6}")
    print(f"{'-'*70}")
    for i in range(n):
        short_name = names_list[i][:48]
        print(f"{short_name:<50} {y[i]:>5.0f}%  {y_pred[i]:>5.0f}%  "
              f"{y_pred[i]-y[i]:>+5.0f}%")

    print(f"\n{'='*70}")
    print(f"FEATURE WEIGHTS  (standardized, intercept={model.intercept_:.1f})")
    print(f"{'='*70}")
    for name, w in zip(FEATURE_NAMES, model.coef_):
        print(f"  {name:<25}  {w:+.2f}")

    print(f"\n{'='*70}")
    print("FEATURE-TARGET PEARSON r")
    print(f"{'='*70}")
    for i, name in enumerate(FEATURE_NAMES):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        print(f"  {name:<25}  r={corr:+.3f}")

    print(f"\n  LOO-CV MAE = {loo_mae:.1f}%  |  Train MAE = {train_mae:.1f}%")
    if loo_mae < 15:
        print("  Good fit, generalises well.")
    elif loo_mae < 22:
        print("  Acceptable fit given only 9 samples. More data would tighten it.")
    else:
        print("  Modest fit. Consider adding more samples.")
