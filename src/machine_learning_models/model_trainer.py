import numpy as np
from data_config import (
    CALIBRATION_METHOD,
    CATEGORICAL_COLUMNS,
    MONOTONE_DECREASING_WHITELIST,
    MONOTONE_INCREASING_WHITELIST,
    N_ITER,
)
from hyperparameter_tuning import tune_hyperparameters
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedShuffleSplit
from xgboost import XGBClassifier


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


def evaluate_model(model, X_test, y_test, feature_names=None):
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="binary")),
        "recall": float(recall_score(y_test, y_pred, average="binary")),
        "f1": float(f1_score(y_test, y_pred, average="binary")),
    }

    feature_importance = _extract_feature_importance(model, feature_names)
    return metrics, feature_importance


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


def train_model(X_train, y_train, model_name, auto_tune=None, sample_weight=None):
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
        )
    else:
        model_params = {
            "objective": "binary:logistic",
            "random_state": 42,
            "enable_categorical": True,
            "n_estimators": 375,
            "max_depth": 10,
            "learning_rate": 0.16804401335949273,
            "subsample": 0.8713709892173486,
            "colsample_bytree": 0.923086999703449,
            "min_child_weight": 8,
            "gamma": 0.360787817587864,
            "reg_alpha": 0.6332579825935543,
            "reg_lambda": 0.1080629839688434,
        }

    if model_name == "xgboost":
        if auto_tune:
            model_params.update(
                {
                    "objective": "binary:logistic",
                    "random_state": 42,
                    "enable_categorical": True,
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

    fit_params = _prepare_sample_weight(sample_weight, train_idx)
    base_estimator.fit(X_tr, y_tr, **fit_params)

    calibrated_model = CalibratedClassifierCV(
        base_estimator, method=CALIBRATION_METHOD, cv="prefit"
    )
    cal_fit_params = _prepare_sample_weight(sample_weight, calib_idx)
    calibrated_model.fit(X_cal, y_cal, **cal_fit_params)

    calibration_params = extract_calibration_params(calibrated_model)

    return calibrated_model, model_params, CALIBRATION_METHOD, calibration_params
