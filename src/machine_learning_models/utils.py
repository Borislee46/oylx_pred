import json
import math
import os
from datetime import datetime

import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV


def save_model(
    model, model_name, feature_names, x_test, calibration_params=None, level_fallback_mapping=None
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, "pre-trained_models")
    os.makedirs(model_dir, exist_ok=True)

    filenames_saved = []

    if model_name == "xgboost":
        try:
            if isinstance(model, CalibratedClassifierCV):
                inner_model = None
                if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
                    inner_cc = model.calibrated_classifiers_[0]
                    inner_model = getattr(inner_cc, "estimator", None) or getattr(
                        inner_cc, "base_estimator", None
                    )
                if inner_model is None:
                    inner_model = getattr(model, "base_estimator", None)

                if isinstance(inner_model, xgb.XGBClassifier):
                    model_filename = os.path.join(model_dir, f"{model_name}_{timestamp}.model")
                    inner_model.save_model(model_filename)
                    filenames_saved.append(model_filename)

                    features_filename = os.path.join(
                        model_dir, f"{model_name}_{timestamp}_features.json"
                    )
                    with open(features_filename, "w", encoding="utf-8") as f:
                        json.dump(feature_names, f, ensure_ascii=False, indent=2)
                    filenames_saved.append(features_filename)

                    if calibration_params:
                        calib_filename = os.path.join(
                            model_dir, f"{model_name}_{timestamp}_calibration.json"
                        )
                        with open(calib_filename, "w", encoding="utf-8") as f:
                            json.dump(calibration_params, f, ensure_ascii=False, indent=2)
                        filenames_saved.append(calib_filename)

                    # 保存 level_fallback_mapping
                    if level_fallback_mapping:
                        fallback_filename = os.path.join(
                            model_dir, f"{model_name}_{timestamp}_level_fallback.json"
                        )
                        with open(fallback_filename, "w", encoding="utf-8") as f:
                            json.dump(level_fallback_mapping, f, ensure_ascii=False, indent=2)
                        filenames_saved.append(fallback_filename)
            elif isinstance(model, xgb.XGBClassifier):
                model_filename = os.path.join(model_dir, f"{model_name}_{timestamp}.model")
                model.save_model(model_filename)
                filenames_saved.append(model_filename)

                features_filename = os.path.join(
                    model_dir, f"{model_name}_{timestamp}_features.json"
                )
                with open(features_filename, "w", encoding="utf-8") as f:
                    json.dump(feature_names, f, ensure_ascii=False, indent=2)
                filenames_saved.append(features_filename)

                # 保存 level_fallback_mapping
                if level_fallback_mapping:
                    fallback_filename = os.path.join(
                        model_dir, f"{model_name}_{timestamp}_level_fallback.json"
                    )
                    with open(fallback_filename, "w", encoding="utf-8") as f:
                        json.dump(level_fallback_mapping, f, ensure_ascii=False, indent=2)
                    filenames_saved.append(fallback_filename)
        except Exception:
            raise ValueError(f"保存模型 {model_name} 失败")

    return filenames_saved


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
    result_data = {"model_name": model_name, "timestamp": timestamp, "metrics": metrics}
    if feature_importance is not None:
        result_data["feature_importance"] = feature_importance
    if auto_tune_method is not None:
        result_data["auto_tune_method"] = auto_tune_method
    if model_params is not None:
        result_data["model_params"] = model_params
    if sampling_method is not None:
        result_data["sampling_method"] = sampling_method
    if calibration_method is not None:
        result_data["calibration_method"] = calibration_method
    if package_versions is not None:
        result_data["package_versions"] = package_versions
    result_data_cleaned = replace_nan_with_none(result_data)
    with open(result_filename, "w", encoding="utf-8") as f:
        json.dump(result_data_cleaned, f, ensure_ascii=False, indent=4)
    return result_filename
