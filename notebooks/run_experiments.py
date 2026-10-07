"""Step 2: baselines + ablation models, chronological split, all horizons.

Usage:
    python run_experiments.py            # main experiments (split by year)
    python run_experiments.py --loco     # extra: leave-one-city-out test

Outputs (in results/): predictions.csv, metrics_overall.csv, metrics_by_city.csv
(and metrics_loco.csv with --loco).
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (FEATURE_SETS, HORIZONS, RANDOM_STATE, RESULTS_DIR, city_tiers,
                    evaluate, load_data, make_catboost, make_xgb, split_by_year, target_col)


def run_main(df, tiers):
    pred_rows, metric_rows, city_rows = [], [], []

    for h in HORIZONS:
        train, test = split_by_year(df, h)
        y_col = target_col(h)
        y_tr, y_te = train[y_col].values, test[y_col].values
        print(f"\nHorizon h={h}: train {len(train):,} rows, test {len(test):,} rows")

        preds = {
            "A_all_zero": np.zeros(len(test)),
            "B_persistence": test["cases_lag1"].values,  # last known weekly count
        }

        full = FEATURE_SETS["full"]
        poisson = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1.0, max_iter=2000))
        poisson.fit(train[full], y_tr)
        preds["C_poisson_regression"] = poisson.predict(test[full])

        for set_name, cols in FEATURE_SETS.items():
            for obj, short in [("count:poisson", "pois"), ("reg:squarederror", "sq")]:
                model = make_xgb(obj)
                model.fit(train[cols], y_tr)
                preds[f"XGB_{set_name}_{short}"] = model.predict(test[cols])

        for set_name, cols in FEATURE_SETS.items():
            for loss, short in [("Poisson", "pois"), ("RMSE", "rmse")]:
                cat = make_catboost(loss)
                cat.fit(train[cols], y_tr)
                preds[f"CAT_{set_name}_{short}"] = cat.predict(test[cols])

        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                   n_jobs=-1, random_state=RANDOM_STATE)
        rf.fit(train[full], y_tr)
        preds["RF_full"] = rf.predict(test[full])

        for name, p in preds.items():
            p = np.clip(p, 0, None)
            tier_series = test["adm3_en"].map(tiers)
            for scope, mask in [("All cities", np.ones(len(test), bool)),
                                ("Tier 1", (tier_series == "Tier 1").values),
                                ("Tier 2", (tier_series == "Tier 2").values)]:
                if mask.sum():
                    metric_rows.append({"horizon": h, "model": name, "scope": scope,
                                        **evaluate(y_te[mask], p[mask])})
            for city, g in test.assign(_p=p, _y=y_te).groupby("adm3_en"):
                city_rows.append({"horizon": h, "model": name, "city": city,
                                  "tier": tiers[city], **evaluate(g["_y"], g["_p"])})
            pred_rows.append(pd.DataFrame({"date": test["date"].values, "city": test["adm3_en"].values,
                                           "horizon": h, "model": name, "y_true": y_te, "y_pred": p}))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(pred_rows).to_csv(RESULTS_DIR / "predictions.csv", index=False)
    overall = pd.DataFrame(metric_rows)
    overall.to_csv(RESULTS_DIR / "metrics_overall.csv", index=False)
    pd.DataFrame(city_rows).to_csv(RESULTS_DIR / "metrics_by_city.csv", index=False)

    show = overall[overall["scope"] == "Tier 1"].pivot(index="model", columns="horizon", values="MAE_nonzero")
    print("\nMAE on NON-ZERO weeks, Tier 1 cities (lower is better):")
    print(show.round(3).to_string())
    show = overall[overall["scope"] == "All cities"].pivot(index="model", columns="horizon", values="Poisson_deviance")
    print("\nPoisson deviance, all cities (lower is better):")
    print(show.round(3).to_string())


def run_loco(df, tiers, horizon=2):
    """Leave-one-city-out: train on 11 cities (training years), test on the held-out city
    (test years). Tests whether the model works for a city it has never seen."""
    rows = []
    train, test = split_by_year(df, horizon)
    y_col = target_col(horizon)
    for city in sorted(df["adm3_en"].unique()):
        tr, te = train[train["adm3_en"] != city], test[test["adm3_en"] == city]
        preds = {"A_all_zero": np.zeros(len(te)), "B_persistence": te["cases_lag1"].values}
        for set_name in ("case_weather", "full"):
            cols = FEATURE_SETS[set_name]
            model = make_xgb("count:poisson")
            model.fit(tr[cols], tr[y_col])
            preds[f"XGB_{set_name}_pois"] = model.predict(te[cols])
            cat = make_catboost("Poisson")
            cat.fit(tr[cols], tr[y_col])
            preds[f"CAT_{set_name}_pois"] = cat.predict(te[cols])
        for name, p in preds.items():
            rows.append({"held_out_city": city, "tier": tiers[city], "horizon": horizon,
                         "model": name, **evaluate(te[y_col], p)})
    out = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(RESULTS_DIR / "metrics_loco.csv", index=False)
    print(f"\nLeave-one-city-out, horizon h={horizon}: MAE on non-zero weeks")
    print(out.pivot(index="held_out_city", columns="model", values="MAE_nonzero").round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--loco", action="store_true", help="run leave-one-city-out test")
    args = ap.parse_args()
    data = load_data()
    tier_map = city_tiers(data)
    print("City tiers (from training years):", tier_map)
    if args.loco:
        run_loco(data, tier_map)
    else:
        run_main(data, tier_map)
