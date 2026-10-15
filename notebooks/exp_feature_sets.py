"""Experiment 1: which feature set should the app ship? (also tests seasonality and seed noise)

Compares case-only, +season, +weather (with/without NDVI), +season, and the full set.
Each model is trained with 3 random seeds; results are averaged so one lucky seed
cannot decide the ranking.

Usage:  python exp_feature_sets.py
Outputs: results/exp_feature_sets.csv  (+ printed tables to paste)
"""
import numpy as np
import pandas as pd

from common import (CASE_FEATURES, FEATURE_SETS, RESULTS_DIR, SEASON_FEATURES, STATIC_FEATURES,
                    WEATHER_FEATURES, WEATHER_NO_NDVI, add_season, city_tiers, evaluate,
                    load_data, make_xgb, split_by_year, target_col)

SEEDS = [42, 7, 2024]
SETS = {
    "case_only": CASE_FEATURES,
    "case_season": CASE_FEATURES + SEASON_FEATURES,
    "case_weather_noNDVI": CASE_FEATURES + WEATHER_NO_NDVI,
    "case_weather": CASE_FEATURES + WEATHER_FEATURES,
    "case_weather_noNDVI_season": CASE_FEATURES + WEATHER_NO_NDVI + SEASON_FEATURES,
    "case_weather_season": CASE_FEATURES + WEATHER_FEATURES + SEASON_FEATURES,
    "full": FEATURE_SETS["full"],
}

df = load_data()
tiers = city_tiers(df)
rows = []
for h in [0, 1, 2, 3]:
    train, test = split_by_year(df, h)
    train, test = add_season(train, h), add_season(test, h)
    y_tr, y_te = train[target_col(h)].values, test[target_col(h)].values
    tier1 = (test["adm3_en"].map(tiers) == "Tier 1").values
    for name, cols in SETS.items():
        for seed in SEEDS:
            m = make_xgb("count:poisson", seed=seed)
            m.fit(train[cols], y_tr)
            p = np.clip(m.predict(test[cols]), 0, None)
            for scope, mask in [("All cities", np.ones(len(test), bool)), ("Tier 1", tier1)]:
                rows.append({"horizon": h, "feature_set": name, "seed": seed, "scope": scope,
                             **evaluate(y_te[mask], p[mask])})
    print(f"horizon {h} done")

res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "exp_feature_sets.csv", index=False)
order = list(SETS)
for scope in ["All cities", "Tier 1"]:
    d = res[res.scope == scope]
    for metric in ["Poisson_deviance", "RMSE", "MAE"]:
        mean = d.groupby(["feature_set", "horizon"])[metric].mean().unstack().reindex(order)
        print(f"\n=== {metric} (mean of {len(SEEDS)} seeds) | {scope} ===")
        print(mean.round(3).to_string())
sd = res[res.scope == "All cities"].groupby(["feature_set", "horizon"])["Poisson_deviance"].std().unstack().reindex(order)
print("\n=== Seed-to-seed standard deviation of deviance (how much is just noise) ===")
print(sd.round(3).to_string())
