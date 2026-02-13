import argparse
import os
import platform
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_loader import load_data
from model_trainer import evaluate_model, train_model

import utils

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
    parser.add_argument("--sampling_method", type=str, default="smote")
    parser.add_argument("--auto_tune", action="store_true")
    args = parser.parse_args()

    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "cases.feather")

    X_train, X_test, y_train, y_test, feature_names, sample_weight_train, level_fallback_mapping = (
        load_data(data_path, sampling_method=args.sampling_method)
    )

    model, model_params, calibration_method, calibration_params = train_model(
        X_train, y_train, args.model, auto_tune=args.auto_tune, sample_weight=sample_weight_train
    )

    metrics, feature_importance = evaluate_model(model, X_test, y_test, feature_names)

    utils.save_model(
        model,
        args.model,
        feature_names,
        X_test,
        calibration_params=calibration_params,
        level_fallback_mapping=level_fallback_mapping,
    )

    utils.save_evaluation_results(
        model_name=args.model,
        metrics=metrics,
        feature_importance=feature_importance,
        auto_tune_method="optuna" if args.auto_tune else None,
        model_params=model_params,
        sampling_method=args.sampling_method,
        calibration_method=calibration_method,
        package_versions=get_package_versions(),
    )


if __name__ == "__main__":
    main()
