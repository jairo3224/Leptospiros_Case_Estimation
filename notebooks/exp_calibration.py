"""Experiment 3: are the probabilities and ranges the app will show honest?

Uses results/predictions.csv. For each prediction mu (expected cases):
  * P(at least one case) = 1 - exp(-mu)    -> compared with how often a case really occurred
  * a Poisson-assumption 80% range -> how often the real count fell inside it

The probability and range are diagnostics under a Poisson assumption, not a
calibrated uncertainty guarantee for the fitted model.

Usage:  python exp_calibration.py
Outputs: results/calibration_bins.csv, results/calibration_coverage.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import poisson

from common import RESULTS_DIR, city_tiers, load_data

MODEL = "XGB_case_weather_pois"
pred = pd.read_csv(RESULTS_DIR / "predictions.csv", parse_dates=["date"])
pred = pred[pred.model == MODEL].copy()
tiers = city_tiers(load_data())
pred["tier"] = pred["city"].map(tiers)
pred["p_any"] = 1 - np.exp(-pred["y_pred"])
pred["any_case"] = (pred["y_true"] > 0).astype(int)
bins = [0, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.0001]
pred["bin"] = pd.cut(pred["p_any"], bins, right=False)

out = []
for h, g in pred.groupby("horizon"):
    t = g.groupby("bin", observed=True).agg(n=("any_case", "size"), predicted=("p_any", "mean"),
                                           observed=("any_case", "mean")).reset_index()
    t.insert(0, "horizon", h)
    out.append(t)
bins_df = pd.concat(out)
bins_df.to_csv(RESULTS_DIR / "calibration_bins.csv", index=False)
print(f"=== Calibration of 'chance of at least one case' ({MODEL}) ===")
print("predicted vs observed should match in every row\n")
print(bins_df.round(3).to_string(index=False))

rows = []
for (h, tier), g in list(pred.groupby(["horizon", "tier"])) + [((h, "All"), g) for h, g in pred.groupby("horizon")]:
    lo, hi = poisson.ppf(0.10, g["y_pred"]), poisson.ppf(0.90, g["y_pred"])
    inside = ((g["y_true"] >= lo) & (g["y_true"] <= hi)).mean()
    brier = ((g["p_any"] - g["any_case"]) ** 2).mean()
    base = ((g["any_case"].mean() - g["any_case"]) ** 2).mean()
    rows.append({"horizon": h, "scope": tier, "n": len(g), "coverage_of_80pct_range": inside,
                 "brier": brier, "brier_of_constant_rate": base,
                 "mean_predicted": g["y_pred"].mean(), "mean_actual": g["y_true"].mean()})
cov = pd.DataFrame(rows).sort_values(["horizon", "scope"])
cov.to_csv(RESULTS_DIR / "calibration_coverage.csv", index=False)
print("\n=== Does the 80% range contain the real count ~80% of the time? ===")
print(cov.round(3).to_string(index=False))
