import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from xgboost import XGBClassifier

from .data_config import (
    CALIBRATION_METHOD,
    CATEGORICAL_COLUMNS,
    DEFAULT_PREDICTION_THRESHOLD,
    MONOTONE_DECREASING_WHITELIST,
    MONOTONE_INCREASING_WHITELIST,
    N_ITER,
    THRESHOLD_SCAN_STEPS,
)
from .hyperparameter_tuning import tune_hyperparameters


def extract_calibration_params(model):
    if not hasattr(model, "calibrated_classifiers_") or not model.calibrated_classifiers_:
        return None

    cc = model.calibrated_classifiers_[0]
    calibrators = getattr(cc, "calibrators", None)

    if not calibrators:
        return None

    if len(calibrators) == 1:
        calibrator = calibrators[0]
    else:
        pos_index = (
            cc.classes_.tolist().index(1) if hasattr(cc, "classes_") and 1 in cc.classes_ else 1
        )
        calibrator = calibrators[pos_index] if pos_index < len(calibrators) else None

    if calibrator is None:
        return None

    if hasattr(calibrator, "a_") and hasattr(calibrator, "b_"):
        return {
            "method": "sigmoid",
            "params": {"a": float(calibrator.a_), "b": float(calibrator.b_)},
        }
    elif hasattr(calibrator, "X_thresholds_") and hasattr(calibrator, "y_thresholds_"):
        return {
            "method": "isotonic",
            "params": {
                "x_thresholds": calibrator.X_thresholds_.tolist(),
                "y_thresholds": calibrator.y_thresholds_.tolist(),
            },
        }

    return None


def _predict_positive_class_proba(model, X):
    probas = model.predict_proba(X)
    if probas is None:
        raise ValueError("模型未返回有效概率")
    if probas.ndim == 1:
        return probas.astype(float)
    return probas[:, 1].astype(float)


def _safe_probability_metric(metric_fn, y_true, y_score):
    try:
        return float(metric_fn(y_true, y_score))
    except ValueError:
        return None


def _build_threshold_scan(y_true, y_score):
    thresholds = np.linspace(0.0, 1.0, THRESHOLD_SCAN_STEPS)
    best = None

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        score = float(f1_score(y_true, y_pred, average="binary", zero_division=0))
        candidate = {
            "threshold": float(threshold),
            "f1": score,
            "precision": float(precision_score(y_true, y_pred, average="binary", zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, average="binary", zero_division=0)),
        }
        if best is None or candidate["f1"] > best["f1"]:
            best = candidate

    return best


def evaluate_model(
    model,
    X_test,
    y_test,
    feature_names=None,
    prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
):
    y_score = _predict_positive_class_proba(model, X_test)
    y_pred = (y_score >= prediction_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "prediction_threshold": float(prediction_threshold),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="binary", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="binary", zero_division=0)),
        "roc_auc": _safe_probability_metric(roc_auc_score, y_test, y_score),
        "average_precision": _safe_probability_metric(average_precision_score, y_test, y_score),
        "log_loss": _safe_probability_metric(log_loss, y_test, y_score),
        "brier_score": _safe_probability_metric(brier_score_loss, y_test, y_score),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "positive_rate_actual": float(np.mean(y_test)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "best_f1_threshold_scan": _build_threshold_scan(y_test, y_score),
    }

    feature_importance = _extract_feature_importance(model, feature_names)
    return metrics, feature_importance


_CV_METRIC_KEYS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
    "log_loss",
    "brier_score",
]


def cross_validate_model(
    X,
    y,
    model_name="xgboost",
    cv=5,
    n_repeats=3,
    sample_weight=None,
    prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
):
    rskf = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=2025)

    all_metrics: list[dict] = []
    for train_idx, test_idx in rskf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        sw_tr = None
        if sample_weight is not None:
            try:
                sw_tr = sample_weight.iloc[train_idx]
            except AttributeError:
                sw_tr = sample_weight[train_idx]

        calibrated_model, _, _, _ = train_model(
            X_tr,
            y_tr,
            model_name,
            auto_tune=False,
            sample_weight=sw_tr,
            prediction_threshold=prediction_threshold,
        )

        fold_metrics, _ = evaluate_model(
            calibrated_model,
            X_te,
            y_te,
            prediction_threshold=prediction_threshold,
        )
        all_metrics.append(fold_metrics)

    cv_results: dict[str, dict] = {}
    for key in _CV_METRIC_KEYS:
        values = [m[key] for m in all_metrics if m.get(key) is not None]
        if values:
            cv_results[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
                "n_folds": len(values),
            }

    cv_results["_meta"] = {
        "cv": cv,
        "n_repeats": n_repeats,
        "total_folds": len(all_metrics),
    }
    return cv_results


def _extract_feature_importance(model, feature_names):
    if feature_names is None:
        return None

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        all_importances = []
        for cc in model.calibrated_classifiers_:
            if hasattr(cc, "estimator") and hasattr(cc.estimator, "feature_importances_"):
                importances = cc.estimator.feature_importances_
                if importances is not None:
                    all_importances.append(importances)

        if all_importances:
            try:
                if all(imp.shape == all_importances[0].shape for imp in all_importances):
                    mean_importances = np.mean(all_importances, axis=0)
                    return _build_importance_dict(mean_importances, feature_names)
            except Exception:
                pass
    else:
        base_est = getattr(model, "base_estimator", None) or getattr(model, "estimator", None)
        if base_est and hasattr(base_est, "feature_importances_"):
            return _build_importance_dict(base_est.feature_importances_, feature_names)

    return None


def _build_importance_dict(importances, feature_names):
    indices = np.argsort(importances)[::-1]
    return {
        feature_names[idx]: float(importances[idx]) for idx in indices if idx < len(feature_names)
    }


def _build_monotone_constraints(feature_names, categorical_features):
    constraints = []
    for feature in feature_names:
        if feature in categorical_features:
            constraints.append(0)
        elif feature in MONOTONE_INCREASING_WHITELIST:
            constraints.append(1)
        elif feature in MONOTONE_DECREASING_WHITELIST:
            constraints.append(-1)
        else:
            constraints.append(0)
    return tuple(constraints)


def _calculate_scale_pos_weight(y_train):
    if hasattr(y_train, "iloc") and not y_train.empty:
        n_zeros = len(y_train[y_train == 0])
        n_ones = len(y_train[y_train == 1])
        if n_ones > 0:
            scale_pos_weight = n_zeros / n_ones
            return scale_pos_weight
    return 1.0


def _create_base_estimator(model_class, params, model_name):
    try:
        return model_class(**params)
    except TypeError:
        cleaned_params = {k: v for k, v in params.items() if k in model_class().get_params()}
        return model_class(**cleaned_params)


def _prepare_sample_weight(sample_weight, indices):
    if sample_weight is None:
        return {}

    try:
        sw = sample_weight.astype("float32") if hasattr(sample_weight, "astype") else sample_weight
        sw_subset = sw.iloc[indices] if hasattr(sw, "iloc") else sw
        return {"sample_weight": sw_subset}
    except Exception:
        return {"sample_weight": sample_weight}


def train_model(
    X_train,
    y_train,
    model_name,
    auto_tune=None,
    sample_weight=None,
    prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
    optimization_metric="log_loss",
):
    categorical_feature_names = [col for col in CATEGORICAL_COLUMNS if col in X_train.columns]

    monotone_constraints = (
        _build_monotone_constraints(X_train.columns.tolist(), categorical_feature_names)
        if model_name == "xgboost"
        else None
    )

    scale_pos_weight = _calculate_scale_pos_weight(y_train) if model_name == "xgboost" else 1.0

    if auto_tune:
        model_params = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            cv=3,
            n_iter=N_ITER,
            n_jobs=-1,
            monotone_constraints=monotone_constraints,
            sample_weight=sample_weight,
            scale_pos_weight=scale_pos_weight,
            prediction_threshold=prediction_threshold,
            optimization_metric=optimization_metric,
        )
    else:
        model_params = {
            "objective": "binary:logistic",
            "random_state": 42,
            "enable_categorical": True,
            "tree_method": "hist",
            "n_estimators": 590,
            "max_depth": 14,
            "learning_rate": 0.024596380220963276,
            "subsample": 0.9840892835990516,
            "colsample_bytree": 0.52392474198857,
            "min_child_weight": 1,
            "gamma": 0.1100355162008238,
            "reg_alpha": 0.6962645037580109,
            "reg_lambda": 0.041412250557364715,
        }

    if model_name == "xgboost":
        if auto_tune:
            model_params.update(
                {
                    "objective": "binary:logistic",
                    "random_state": 42,
                    "enable_categorical": True,
                    "tree_method": "hist",
                }
            )
        model_params["scale_pos_weight"] = scale_pos_weight
        model_params["monotone_constraints"] = monotone_constraints

    for key in ["early_stopping_rounds", "eval_set", "eval_metric", "callbacks"]:
        model_params.pop(key, None)

    base_estimator = _create_base_estimator(XGBClassifier, model_params, model_name)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, calib_idx = next(splitter.split(X_train, y_train))
    X_tr, X_cal = X_train.iloc[train_idx], X_train.iloc[calib_idx]
    y_tr, y_cal = y_train.iloc[train_idx], y_train.iloc[calib_idx]

    sw_tr = _prepare_sample_weight(sample_weight, train_idx).get("sample_weight", None)
    fit_params = {"sample_weight": sw_tr} if sw_tr is not None else {}
    base_estimator.fit(X_tr, y_tr, **fit_params)

    calibrated_model = CalibratedClassifierCV(
        FrozenEstimator(base_estimator), method=CALIBRATION_METHOD
    )
    cal_fit_params = _prepare_sample_weight(sample_weight, calib_idx)
    calibrated_model.fit(X_cal, y_cal, **cal_fit_params)

    calibration_params = extract_calibration_params(calibrated_model)

    return calibrated_model, model_params, CALIBRATION_METHOD, calibration_params
