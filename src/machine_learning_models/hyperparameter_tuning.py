import multiprocessing
import warnings

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


def tune_hyperparameters(
    X_train, y_train, model_name, cv=3, n_iter=50, n_jobs=1, monotone_constraints=None
):
    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    n_cpus = multiprocessing.cpu_count()
    xgb_n_jobs = max(1, n_cpus // n_jobs) if n_jobs > 0 else 1

    def objective(trial):
        if model_name != "xgboost":
            raise ValueError(f"模型 {model_name} 未配置用于手动CV的objective函数。")

        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 0.5),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 1),
            "n_jobs": xgb_n_jobs,
        }

        intermediate_scores = []
        for step, (train_idx, val_idx) in enumerate(stratified_kfold.split(X_train, y_train)):
            X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            model = XGBClassifier(
                **params,
                random_state=42,
                enable_categorical=True,
                monotone_constraints=monotone_constraints,
                early_stopping_rounds=15,
                tree_method="hist",
            )
            model.fit(
                X_fold_train, y_fold_train,
                eval_set=[(X_fold_val, y_fold_val)],
                verbose=False,
            )

            score = f1_score(y_fold_val, model.predict(X_fold_val), average="binary")
            intermediate_scores.append(score)
            trial.report(score, step)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(intermediate_scores)

    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=max(0, cv // 2 - 1))
    study = optuna.create_study(direction="maximize", pruner=pruner)
    study.optimize(objective, n_trials=n_iter, n_jobs=n_jobs)
    best_params = study.best_params

    return best_params
