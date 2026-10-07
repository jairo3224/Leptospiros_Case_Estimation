"""Step 5: block-bootstrap confidence intervals for the key model comparisons.

Usage:  python bootstrap_ci.py
Reads  results/predictions.csv  (created by run_experiments.py)
Writes results/bootstrap_ci.csv

Method: test weeks are resampled in blocks of 8 consecutive weeks (all cities in a
week stay together), so correlation across cities and over time is respected.
Each comparison reports the metric for model A, model B, and the % reduction
(A - B) / A with a 95% interval. If the interval excludes 0, the difference is
unlikely to be noise.
"""
import numpy as np
import pandas as pd

from common import RANDOM_STATE, RESULTS_DIR

N_BOOT = 2000
BLOCK = 8

# (metric, model A = reference, model B = candidate)  ->  % reduction going from A to B
COMPARISONS = [
    ("RMSE", "B_persistence", "XGB_case_weather_pois"),
    ("RMSE", "A_all_zero", "XGB_case_weather_pois"),
    ("MAE", "B_persistence", "XGB_case_weather_pois"),
    ("MAE", "A_all_zero", "XGB_case_weather_pois"),
    ("MAE", "A_all_zero", "XGB_full_pois"),
    ("RMSE", "C_poisson_regression", "XGB_case_weather_pois"),
    ("Deviance", "XGB_case_only_pois", "XGB_case_weather_pois"),   # weather gain
    ("Deviance", "XGB_case_weather_pois", "XGB_full_pois"),        # built-environment gain
    ("Deviance", "XGB_full_pois", "CAT_full_pois"),                # XGBoost vs CatBoost
    ("RMSE", "XGB_case_weather_pois", "XGB_full_pois"),
]


def poisson_dev(y, mu):
    mu = np.clip(mu, 1e-3, None)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0.0)
    return 2 * (term - (y - mu))


def per_date_sums(g):
    y, p = g["y_true"].values, np.clip(g["y_pred"].values, 0, None)
    return pd.Series({"sq": ((y - p) ** 2).sum(), "ab": np.abs(y - p).sum(),
                      "dv": poisson_dev(y, p).sum(), "n": len(y)})


def metric_from(sums, idx, which):
    s = {k: v[idx].sum(axis=-1) for k, v in sums.items()}
    if which == "RMSE":
        return np.sqrt(s["sq"] / s["n"])
    if which == "MAE":
        return s["ab"] / s["n"]
    return s["dv"] / s["n"]


def main():
    pred = pd.read_csv(RESULTS_DIR / "predictions.csv", parse_dates=["date"])
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for h, ph in pred.groupby("horizon"):
        dates = np.sort(ph["date"].unique())
        d = len(dates)
        n_blocks = int(np.ceil(d / BLOCK))
        starts = rng.integers(0, d - BLOCK + 1, size=(N_BOOT, n_blocks))
        idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_BOOT, -1)[:, :d]
        full_idx = np.arange(d)

        sums = {}
        for model, pm in ph.groupby("model"):
            agg = pm.groupby("date").apply(per_date_sums, include_groups=False).reindex(dates)
            sums[model] = {k: agg[k].values for k in ["sq", "ab", "dv", "n"]}

        for which, a, b in COMPARISONS:
            if a not in sums or b not in sums:
                continue
            va_all, vb_all = metric_from(sums[a], idx, which), metric_from(sums[b], idx, which)
            red = (va_all - vb_all) / va_all * 100
            va = float(metric_from(sums[a], full_idx, which))
            vb = float(metric_from(sums[b], full_idx, which))
            lo, hi = np.percentile(red, [2.5, 97.5])
            rows.append({"horizon": h, "metric": which, "reference_A": a, "candidate_B": b,
                         "value_A": va, "value_B": vb, "pct_reduction": (va - vb) / va * 100,
                         "ci_low": lo, "ci_high": hi, "ci_excludes_0": bool(lo > 0 or hi < 0)})

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "bootstrap_ci.csv", index=False)
    show = out.assign(result=lambda x: x.pct_reduction.round(1).astype(str) + "  [" +
                      x.ci_low.round(1).astype(str) + ", " + x.ci_high.round(1).astype(str) + "]")
    for (which, a, b), g in show.groupby(["metric", "reference_A", "candidate_B"], sort=False):
        print(f"\n{which}: % reduction from {a} to {b}  (95% CI; positive = B is better)")
        print(g.set_index("horizon")["result"].to_string())
    print(f"\nSaved to {RESULTS_DIR / 'bootstrap_ci.csv'}")


if __name__ == "__main__":
    main()
