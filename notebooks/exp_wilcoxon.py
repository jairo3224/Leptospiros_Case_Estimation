"""Experiment 8: Wilcoxon signed-rank tests on paired model errors.

Reads results/predictions.csv (made by run_experiments.py). For each comparison
"reference model -> candidate model" it pairs the two models' errors on the SAME units
and tests whether the candidate's errors are systematically smaller.

Units (levels of pairing):
  week       - average error across the 12 cities for each test week   (main analysis)
  city       - average error across test weeks for each of the 12 cities (secondary, n = 12)
  city-week  - every city-week separately (shown for reference only: observations are
               correlated, so p-values here are too optimistic)
Losses:  abs = absolute error (MAE-style), sq = squared error (RMSE-style),
         dev = Poisson unit deviance (only for model-vs-model, not vs the zero baselines).

Multiple comparisons: Holm correction within each (level, loss, horizon) family.

Usage:  python exp_wilcoxon.py
Outputs: results/wilcoxon_results.csv  (+ printed tables to paste)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon

from common import RESULTS_DIR

COMPARISONS = [
    ("B_persistence", "XGB_case_weather_pois"),
    ("A_all_zero", "XGB_case_weather_pois"),
    ("A_all_zero", "XGB_full_pois"),
    ("C_poisson_regression", "XGB_case_weather_pois"),
    ("XGB_case_only_pois", "XGB_case_weather_pois"),   # does weather help?
    ("XGB_case_weather_pois", "XGB_full_pois"),        # does built environment help?
    ("XGB_full_pois", "CAT_full_pois"),
    ("XGB_full_pois", "RF_full"),
]
BASELINES = {"A_all_zero", "B_persistence"}

pred = pd.read_csv(RESULTS_DIR / "predictions.csv", parse_dates=["date"])
pred["y_pred"] = pred["y_pred"].clip(lower=0)
y, mu = pred["y_true"].values, np.clip(pred["y_pred"].values, 1e-3, None)
pred["abs"] = np.abs(pred["y_true"] - pred["y_pred"])
pred["sq"] = (pred["y_true"] - pred["y_pred"]) ** 2
pred["dev"] = 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0) - (y - mu))


def holm(p):
    order = np.argsort(p)
    adj = np.empty(len(p))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


rows = []
for h, ph in pred.groupby("horizon"):
    for level, keys in [("week", ["date"]), ("city", ["city"]), ("city-week", ["date", "city"])]:
        for loss in ["abs", "sq", "dev"]:
            wide = ph.pivot_table(index=keys, columns="model", values=loss, aggfunc="mean")
            fam = []
            for ref, cand in COMPARISONS:
                if loss == "dev" and ({ref, cand} & BASELINES):
                    continue
                d = (wide[ref] - wide[cand]).dropna()
                nz = d[d != 0]
                if len(nz) < 6:
                    continue
                p = wilcoxon(nz).pvalue
                ranks = rankdata(nz.abs())
                wp, wm = ranks[(nz > 0).values].sum(), ranks[(nz < 0).values].sum()
                fam.append({"horizon": h, "level": level, "loss": loss, "comparison": f"{ref} -> {cand}",
                            "n_pairs": len(nz), "median_diff": d.median(), "mean_diff": d.mean(),
                            "rank_biserial": (wp - wm) / (wp + wm), "p_value": p})
            if fam:
                adj = holm(np.array([f["p_value"] for f in fam]))
                for f, a in zip(fam, adj):
                    f["p_holm"] = a
                rows.extend(fam)

res = pd.DataFrame(rows)
res["candidate_better"] = res["rank_biserial"] > 0
res["significant_holm_0.05"] = res["p_holm"] < 0.05
res.to_csv(RESULTS_DIR / "wilcoxon_results.csv", index=False)


def cell(r):
    sign = "+" if r.rank_biserial > 0 else "-"
    star = "*" if r.p_holm < 0.05 else " "
    return f"{sign}{r.p_holm:.3f}{star}"


print("Legend: '+' candidate has smaller errors, '-' reference has smaller errors; value = Holm-adjusted p;")
print("        '*' = significant at 0.05. Reference -> candidate.\n")
for level in ["week", "city"]:
    for loss, name in [("abs", "ABSOLUTE error (MAE-style)"), ("sq", "SQUARED error (RMSE-style)"), ("dev", "POISSON DEVIANCE")]:
        sub = res[(res.level == level) & (res.loss == loss)].copy()
        if sub.empty:
            continue
        sub["cell"] = sub.apply(cell, axis=1)
        print(f"=== {name} | pairing level: {level} (n pairs ~ {int(sub.n_pairs.median())}) ===")
        print(sub.pivot(index="comparison", columns="horizon", values="cell").reindex(
            [f"{a} -> {b}" for a, b in COMPARISONS if f"{a} -> {b}" in set(sub.comparison)]).to_string())
        print()
print(f"Full results saved to {RESULTS_DIR / 'wilcoxon_results.csv'}")
