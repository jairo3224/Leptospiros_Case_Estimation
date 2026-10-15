"""Experiment 5: rolling-origin evaluation (is the result stable across years?)

For each test year Y (2014-2022): train on all earlier years only, test on year Y.
Models: all-zero, persistence, XGBoost (Poisson) with case-only / case+weather / full features.

Usage:  python exp_rolling_origin.py --horizons 2        (add more: --horizons 0 1 2 3)
Outputs: results/exp_rolling_origin.csv (+ printed tables to paste)
"""
import argparse

import numpy as np
import pandas as pd

from common import FEATURE_SETS, RESULTS_DIR, evaluate, load_data, make_xgb, target_col

ap = argparse.ArgumentParser()
ap.add_argument("--horizons", type=int, nargs="+", default=[2])
args = ap.parse_args()
EMBARGO = pd.Timedelta(weeks=5)
SETS = {k: FEATURE_SETS[k] for k in ["case_only", "case_weather", "full"]}

df = load_data()
rows = []
for h in args.horizons:
    ycol = target_col(h)
    d = df.dropna(subset=[ycol]).copy()
    d["target_date"] = d["date"] + pd.Timedelta(weeks=h)
    for year in range(2014, 2023):
        start, end = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year}-12-31")
        tr = d[d["target_date"] < start - EMBARGO]
        te = d[(d["date"] >= start + EMBARGO) & (d["date"] <= end)]
        y = te[ycol].values
        preds = {"A_all_zero": np.zeros(len(te)), "B_persistence": te["cases_lag1"].values}
        for name, cols in SETS.items():
            m = make_xgb("count:poisson")
            m.fit(tr[cols], tr[ycol])
            preds["XGB_" + name] = m.predict(te[cols])
        for model, p in preds.items():
            rows.append({"horizon": h, "test_year": year, "model": model, **evaluate(y, p)})
        print(f"h={h} year {year} done")

res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "exp_rolling_origin.csv", index=False)
for h in args.horizons:
    r = res[res.horizon == h]
    for metric in ["Poisson_deviance", "RMSE", "MAE"]:
        print(f"\n=== {metric} by test year | horizon {h} ===")
        print(r.pivot(index="model", columns="test_year", values=metric).round(2).to_string())
    dv = r.pivot(index="model", columns="test_year", values="Poisson_deviance")
    weather = (dv.loc["XGB_case_only"] - dv.loc["XGB_case_weather"]) / dv.loc["XGB_case_only"] * 100
    built = (dv.loc["XGB_case_weather"] - dv.loc["XGB_full"]) / dv.loc["XGB_case_weather"] * 100
    print(f"\n=== % deviance reduction by year | horizon {h} (positive = better) ===")
    print(pd.DataFrame({"weather_added": weather, "built_env_added": built}).round(1).T.to_string())
    print(f"Weather helped in {(weather > 0).sum()} of {len(weather)} years; "
          f"built environment helped in {(built > 0).sum()} of {len(built)} years.")
