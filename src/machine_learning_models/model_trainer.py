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

    if calibrators is not None and len(calibrators) > 0:
        calibrator = None
        if len(calibrators) == 1:
            calibrator = calibrators[0]
        else:
            pos_index = 1
            if hasattr(cc, "classes_"):
                classes = list(cc.classes_)
                if 1 in classes:
                    pos_index = classes.index(1)
            if pos_index < len(calibrators):
                calibrator = calibrators[pos_index]

        if calibrator is not None:
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
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="binary")
    recall = recall_score(y_test, y_pred, average="binary")
    f1 = f1_score(y_test, y_pred, average="binary")
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    feature_importance = None

    if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        all_importances = []
        for idx, calibrated_classifier in enumerate(model.calibrated_classifiers_):
            if hasattr(calibrated_classifier, "estimator") and hasattr(
                calibrated_classifier.estimator, "feature_importances_"
            ):
                importances = calibrated_classifier.estimator.feature_importances_
                if importances is not None:
                    all_importances.append(importances)
                else:
                    pass
            else:
                estimator_type_name = "未知估计器"
                if hasattr(calibrated_classifier, "estimator"):
                    estimator_type_name = type(calibrated_classifier.estimator).__name__

        if all_importances and feature_names is not None:
            if len(all_importances) < len(model.calibrated_classifiers_):
                pass
            try:
                first_shape = all_importances[0].shape
                if not all(imp.shape == first_shape for imp in all_importances):
                    pass
                else:
                    mean_importances = np.mean(all_importances, axis=0)
                    indices = np.argsort(mean_importances)[::-1]
                    feature_importance = {}
                    for i, feature_idx in enumerate(indices):
                        if feature_idx < len(feature_names):
                            feature_importance[feature_names[feature_idx]] = float(
                                mean_importances[feature_idx]
                            )
                        else:
                            pass
            except Exception as e:
                feature_importance = None
        elif not all_importances:
            pass
        elif feature_names is None:
            pass
    else:
        try:
            base_est = None
            base_est = getattr(model, "base_estimator", None) or getattr(model, "estimator", None)
            if base_est is not None and hasattr(base_est, "feature_importances_"):
                importances = base_est.feature_importances_
                if feature_names is not None and importances is not None:
                    indices = np.argsort(importances)[::-1]
                    feature_importance = {}
                    for feature_idx in indices:
                        if feature_idx < len(feature_names):
                            feature_importance[feature_names[feature_idx]] = float(
                                importances[feature_idx]
                            )
        except Exception:
            feature_importance = None

    return metrics, feature_importance


def train_model(X_train, y_train, model_name, auto_tune=None, sample_weight=None):
    categorical_feature_names = [col for col in CATEGORICAL_COLUMNS if col in X_train.columns]

    monotone_constraints = None
    if model_name == "xgboost":
        feature_names = X_train.columns.tolist()
        constraints = []
        for feature in feature_names:
            if feature in categorical_feature_names:
                constraints.append(0)
            else:
                if feature in MONOTONE_INCREASING_WHITELIST:
                    constraints.append(1)
                elif feature in MONOTONE_DECREASING_WHITELIST:
                    constraints.append(-1)
                else:
                    constraints.append(0)
        monotone_constraints = tuple(constraints)

    base_model_params = {}
    base_model_class = None

    scale_pos_weight = 1.0
    if model_name == "xgboost":
        if hasattr(y_train, "iloc") and not y_train.empty:
            n_zeros = len(y_train[y_train == 0])
            n_ones = len(y_train[y_train == 1])
            if n_ones > 0:
                scale_pos_weight = n_zeros / n_ones
                print(
                    f"类别分布 - 负样本: {n_zeros}, 正样本: {n_ones}, scale_pos_weight: {scale_pos_weight:.4f}"
                )
        else:
            pass

    if not auto_tune:
        if model_name == "xgboost":
            base_model_params = {
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
                "scale_pos_weight": scale_pos_weight,
                "monotone_constraints": monotone_constraints,
            }
            base_model_class = XGBClassifier

        final_base_params = base_model_params

    else:
        best_params = tune_hyperparameters(
            X_train=X_train,
            y_train=y_train,
            model_name=model_name,
            cv=3,
            n_iter=N_ITER,
            n_jobs=-1,
            monotone_constraints=monotone_constraints,
        )

        if model_name == "xgboost":
            base_model_params = best_params
            base_model_params.update(
                {"random_state": 42, "enable_categorical": True, "objective": "binary:logistic"}
            )
            base_model_params["scale_pos_weight"] = scale_pos_weight
            base_model_params["monotone_constraints"] = monotone_constraints
            base_model_class = XGBClassifier

        final_base_params = best_params

    final_base_params.pop("early_stopping_rounds", None)
    final_base_params.pop("eval_set", None)
    final_base_params.pop("eval_metric", None)
    final_base_params.pop("callbacks", None)

    try:
        base_estimator = base_model_class(**final_base_params)
    except TypeError as e:
        cleaned_params = {
            k: v for k, v in final_base_params.items() if k in base_model_class().get_params()
        }
        try:
            base_estimator = base_model_class(**cleaned_params)
            final_base_params = cleaned_params
        except Exception as final_e:
            raise ValueError(
                f"无法为 CalibratedClassifierCV 配置基础模型 {model_name}"
            ) from final_e

    calibration_method = CALIBRATION_METHOD

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, calib_idx = next(splitter.split(X_train, y_train))
    X_tr, X_cal = X_train.iloc[train_idx], X_train.iloc[calib_idx]
    y_tr, y_cal = y_train.iloc[train_idx], y_train.iloc[calib_idx]

    fit_params = {}
    if sample_weight is not None:
        try:
            sw = (
                sample_weight.astype("float32")
                if hasattr(sample_weight, "astype")
                else sample_weight
            )
        except Exception:
            sw = sample_weight
        try:
            sw_tr = sw.iloc[train_idx]
        except Exception:
            sw_tr = sw
        fit_params = {"sample_weight": sw_tr}

    base_estimator.fit(X_tr, y_tr, **fit_params)

    calibrated_model = CalibratedClassifierCV(
        base_estimator, method=calibration_method, cv="prefit"
    )

    cal_fit_params = {}
    if sample_weight is not None:
        try:
            sw_cal = sw.iloc[calib_idx]
        except Exception:
            sw_cal = sw
        cal_fit_params = {"sample_weight": sw_cal}

    calibrated_model.fit(X_cal, y_cal, **cal_fit_params)

    calibration_params = extract_calibration_params(calibrated_model)

    return calibrated_model, final_base_params, calibration_method, calibration_params
