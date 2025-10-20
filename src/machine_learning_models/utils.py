import json
import math
import os
from datetime import datetime

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV


def _save_json(data, model_dir, model_name, timestamp, suffix):
    filename = os.path.join(model_dir, f"{model_name}_{timestamp}_{suffix}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return filename


def _extract_xgb_model(model):
    if isinstance(model, xgb.XGBClassifier):
        return model

    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            inner_cc = model.calibrated_classifiers_[0]
            inner_model = getattr(inner_cc, "estimator", None) or getattr(
                inner_cc, "base_estimator", None
            )
            if inner_model:
                return inner_model
        return getattr(model, "base_estimator", None)

    return None


def save_model(
    model, model_name, feature_names, x_test, calibration_params=None, level_fallback_mapping=None
):
    if model_name != "xgboost":
        return []

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "pre-trained_models")
    os.makedirs(model_dir, exist_ok=True)

    xgb_model = _extract_xgb_model(model)
    if not isinstance(xgb_model, xgb.XGBClassifier):
        return []

    booster = xgb_model.get_booster()

    if feature_names:
        booster.set_attr(feature_names=json.dumps(feature_names, ensure_ascii=False))

    if calibration_params and isinstance(model, CalibratedClassifierCV):
        booster.set_attr(calibration_params=json.dumps(calibration_params, ensure_ascii=False))

    if level_fallback_mapping:
        booster.set_attr(
            level_fallback_mapping=json.dumps(level_fallback_mapping, ensure_ascii=False)
        )

    model_filename = os.path.join(model_dir, f"{model_name}_{timestamp}.ubj")
    xgb_model.save_model(model_filename)

    return [model_filename]


def replace_nan_with_none(obj):
    if isinstance(obj, dict):
        return {k: replace_nan_with_none(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_none(elem) for elem in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    else:
        return obj


def save_evaluation_results(
    model_name,
    metrics,
    feature_importance=None,
    auto_tune_method=None,
    model_params=None,
    sampling_method=None,
    calibration_method=None,
    package_versions=None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(current_dir, "evaluation_results")
    os.makedirs(result_dir, exist_ok=True)

    result_filename = os.path.join(result_dir, f"{model_name}_evaluation_{timestamp}.json")

    result_data = {
        "model_name": model_name,
        "timestamp": timestamp,
        "metrics": metrics,
    }

    optional_fields = {
        "feature_importance": feature_importance,
        "auto_tune_method": auto_tune_method,
        "model_params": model_params,
        "sampling_method": sampling_method,
        "calibration_method": calibration_method,
        "package_versions": package_versions,
    }
    result_data.update({k: v for k, v in optional_fields.items() if v is not None})

    result_data_cleaned = replace_nan_with_none(result_data)
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(result_data_cleaned, f, ensure_ascii=False, indent=4)

    return result_filename
