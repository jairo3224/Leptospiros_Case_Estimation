"""Experiment 6: how much does the Cagayan de Oro 2011-12 surge (554 cases) drive the results?

Variants (training data only; the test years are never changed):
  original        - as is
  drop_cdo_surge  - remove Cagayan de Oro weeks 2011-12-19 .. 2012-01-30 from training
  capped_targets  - cap training targets at the 99.5th percentile
Usage:  python exp_outlier_sensitivity.py --horizon 2
Outputs: results/exp_outlier_sensitivity_h{H}.csv (+ printed table to paste)
"""
import argparse

import numpy as np
import pandas as pd

from common import FEATURE_SETS, RESULTS_DIR, evaluate, load_data, make_xgb, split_by_year, target_col

ap = argparse.ArgumentParser()
ap.add_argument("--horizon", type=int, default=2)
h = ap.parse_args().horizon
ycol = target_col(h)

df = load_data()
train, test = split_by_year(df, h)
target_dates = train["date"] + pd.Timedelta(weeks=h)
surge = (
    train["adm3_en"].eq("Cagayan de Oro City")
    & target_dates.between("2011-12-19", "2012-01-30")
)
cap = float(np.percentile(train[ycol], 99.5))
variants = {
    "original": (train, train[ycol].values),
    "drop_cdo_surge": (train[~surge], train.loc[~surge, ycol].values),
    "capped_targets": (train, np.minimum(train[ycol].values, cap)),
}
print(f"Rows removed in drop_cdo_surge: {int(surge.sum())}; cap value: {cap:.1f} cases")

rows = []
for vname, (tr, y_tr) in variants.items():
    for fname in ["case_weather", "full"]:
        cols = FEATURE_SETS[fname]
        m = make_xgb("count:poisson")
        m.fit(tr[cols], y_tr)
        rows.append({"variant": vname, "features": fname, **evaluate(test[ycol], m.predict(test[cols]))})
res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / f"exp_outlier_sensitivity_h{h}.csv", index=False)
print(f"\n=== Test results (2019-2022), horizon {h} ===")
print(res.round(3).to_string(index=False))
