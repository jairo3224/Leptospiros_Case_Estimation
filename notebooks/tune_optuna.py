"""Experiment 4: hyperparameter tuning with Optuna (time-aware).

Tunes XGBoost (Poisson loss) using expanding-window validation INSIDE the training years
(validate on 2013, 2014, ... 2018, each time training only on earlier years).
The 2019-2022 test set is used only once at the end.

Usage:  python tune_optuna.py --horizon 2 --features case_weather --trials 40
Needs:  pip install optuna
Outputs: results/best_params_h{H}_{FEATURES}.json, results/tuning_compare_h{H}_{FEATURES}.csv
"""
import argparse
import json

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_poisson_deviance
from xgboost import XGBRegressor

from common import FEATURE_SETS, RESULTS_DIR, evaluate, load_data, make_xgb, split_by_year, target_col

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", type=int, default=2)
ap.add_argument("--features", choices=list(FEATURE_SETS), default="case_weather")
ap.add_argument("--trials", type=int, default=40)
args = ap.parse_args()
h, cols, ycol = args.horizon, FEATURE_SETS[args.features], target_col(args.horizon)

df = load_data()
train, test = split_by_year(df, h)
train = train.assign(target_date=train["date"] + pd.Timedelta(weeks=h))
GAP = pd.Timedelta(weeks=5)


def dev(y, p):
    return mean_poisson_deviance(y, np.clip(p, 1e-3, None))


def build(params):
    return XGBRegressor(objective="count:poisson", random_state=42, n_jobs=-1, **params)


def cv_score(params):
    scores = []
    for year in range(2013, 2019):
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        tr = train[train["target_date"] < start - GAP]
        va = train[(train["date"] >= start) & (train["target_date"] <= end)]
        m = build(params)
        m.fit(tr[cols], tr[ycol])
        scores.append(dev(va[ycol], m.predict(va[cols])))
    return float(np.mean(scores))


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
    }
    return cv_score(params)


optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=args.trials, show_progress_bar=False)
best = study.best_params
print("Best parameters:", best)
print("Best CV deviance:", round(study.best_value, 4))

default = make_xgb("count:poisson")
default.fit(train[cols], train[ycol])
tuned = build(best)
tuned.fit(train[cols], train[ycol])
rows = []
for name, model in [("default_settings", default), ("tuned_settings", tuned)]:
    rows.append({"model": name, **evaluate(test[ycol], model.predict(test[cols]))})
cmp_df = pd.DataFrame(rows)
tag = f"h{h}_{args.features}"
cmp_df.to_csv(RESULTS_DIR / f"tuning_compare_{tag}.csv", index=False)
(RESULTS_DIR / f"best_params_{tag}.json").write_text(json.dumps(best, indent=2))
print(f"\n=== Test-set results (2019-2022), horizon {h}, features: {args.features} ===")
print(cmp_df.round(3).to_string(index=False))
print("\nCV deviance for default settings, for reference:", round(cv_score({"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8}), 4))
