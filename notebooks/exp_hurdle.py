"""Experiment 7: hurdle model vs the single Poisson model.

Hurdle = P(any case this week) x E[count | at least one case].
  step 1: XGBoost classifier for 'any case?'
  step 2: XGBoost Poisson regressor trained only on weeks with cases (target - 1)
Usage:  python exp_hurdle.py
Outputs: results/exp_hurdle.csv (+ printed table to paste)
"""
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from common import FEATURE_SETS, RESULTS_DIR, city_tiers, evaluate, load_data, make_xgb, split_by_year, target_col

cols = FEATURE_SETS["case_weather"]
df = load_data()
tiers = city_tiers(df)
rows = []
for h in [0, 1, 2, 3]:
    ycol = target_col(h)
    train, test = split_by_year(df, h)
    y_tr, y_te = train[ycol].values, test[ycol].values
    tier1 = (test["adm3_en"].map(tiers) == "Tier 1").values

    single = make_xgb("count:poisson")
    single.fit(train[cols], y_tr)
    p_single = single.predict(test[cols])

    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                        colsample_bytree=0.8, random_state=42, n_jobs=-1)
    clf.fit(train[cols], (y_tr > 0).astype(int))
    pos = y_tr > 0
    reg = make_xgb("count:poisson")
    reg.fit(train.loc[pos, cols], y_tr[pos] - 1)
    p_hurdle = clf.predict_proba(test[cols])[:, 1] * (np.clip(reg.predict(test[cols]), 0, None) + 1)

    for name, p in [("single_poisson", p_single), ("hurdle", p_hurdle)]:
        for scope, mask in [("All cities", np.ones(len(test), bool)), ("Tier 1", tier1)]:
            rows.append({"horizon": h, "model": name, "scope": scope, **evaluate(y_te[mask], p[mask])})
    print(f"horizon {h} done")

res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "exp_hurdle.csv", index=False)
for metric in ["Poisson_deviance", "RMSE", "MAE"]:
    print(f"\n=== {metric} ===")
    print(res.pivot_table(index=["scope", "model"], columns="horizon", values=metric).round(3).to_string())
