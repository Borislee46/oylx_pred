import argparse
import os
import platform
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_config import DEFAULT_PREDICTION_THRESHOLD
from data_loader import load_data
from model_trainer import evaluate_model, train_model

import utils
from src.utils.model_loader import load_model_dependencies

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


def get_package_versions():
    packages = ["xgboost", "numpy", "pandas", "scipy", "joblib"]
    versions = {"python": platform.python_version()}

    for package in packages:
        module = __import__(package)
        versions[package] = module.__version__
    import sklearn

    versions["scikit-learn"] = sklearn.__version__

    return versions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="xgboost", choices=["xgboost"])
    parser.add_argument("--auto_tune", action="store_true")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases.feather")

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        sample_weight_train,
        level_fallback_mapping,
        feature_engineer_state,
    ) = load_data(data_path)

    model, model_params, calibration_method, calibration_params = train_model(
        X_train,
        y_train,
        args.model,
        auto_tune=args.auto_tune,
        sample_weight=sample_weight_train,
        prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
    )

    metrics, feature_importance = evaluate_model(
        model,
        X_test,
        y_test,
        feature_names,
        prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
    )

    package_versions = get_package_versions()
    saved_model_paths = utils.save_model(
        model,
        args.model,
        feature_names,
        X_test,
        calibration_params=calibration_params,
        level_fallback_mapping=level_fallback_mapping,
        prediction_threshold=DEFAULT_PREDICTION_THRESHOLD,
        feature_engineer_state=feature_engineer_state,
        model_metadata={
            "auto_tune_method": "optuna" if args.auto_tune else None,
            "model_params": model_params,
            "calibration_method": calibration_method,
            "package_versions": package_versions,
        },
    )
    metrics["post_train_smoke_check"] = run_post_train_smoke_check(
        args.model,
        saved_model_paths[-1] if saved_model_paths else None,
        X_test,
        feature_names,
        DEFAULT_PREDICTION_THRESHOLD,
    )

    utils.save_evaluation_results(
        model_name=args.model,
        metrics=metrics,
        feature_importance=feature_importance,
        auto_tune_method="optuna" if args.auto_tune else None,
        model_params=model_params,
        calibration_method=calibration_method,
        package_versions=package_versions,
    )


def run_post_train_smoke_check(
    model_name,
    model_path,
    x_test,
    expected_feature_names,
    expected_threshold,
):
    if not model_path:
        raise RuntimeError("模型保存失败，无法执行训练后自检")

    (
        loaded_model,
        loaded_feature_names,
        _level_fallback_mapping,
        _feature_engineer_state,
        loaded_threshold,
    ) = load_model_dependencies(
        os.path.dirname(model_path),
        model_name,
        model_path=model_path,
    )

    if loaded_model is None:
        raise RuntimeError("训练后自检失败：模型重新加载失败")
    if loaded_feature_names != expected_feature_names:
        raise RuntimeError("训练后自检失败：feature_names 不匹配")
    if abs(float(loaded_threshold) - float(expected_threshold)) > 1e-9:
        raise RuntimeError("训练后自检失败：prediction_threshold 不匹配")

    sample = x_test.head(min(5, len(x_test))).copy()
    if sample.empty:
        return {"status": "skipped", "reason": "empty_test_set"}

    probas = loaded_model.predict_proba(sample)
    if probas is None or len(probas) != len(sample):
        raise RuntimeError("训练后自检失败：predict_proba 返回无效结果")

    return {"status": "passed", "checked_rows": int(len(sample))}


if __name__ == "__main__":
    main()
