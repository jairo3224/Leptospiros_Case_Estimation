"""Step 4: print compact result tables to paste into the chat / the paper.

Usage:  python summarize_results.py
Reads results/metrics_overall.csv and results/metrics_by_city.csv
Writes results/summary_tables.txt
"""
import pandas as pd

from common import RESULTS_DIR

overall = pd.read_csv(RESULTS_DIR / "metrics_overall.csv")
by_city = pd.read_csv(RESULTS_DIR / "metrics_by_city.csv")
out = []


def show(title, df):
    text = f"\n=== {title} ===\n{df.round(3).to_string()}"
    print(text)
    out.append(text)


KEY = ["A_all_zero", "B_persistence", "C_poisson_regression", "XGB_case_only_pois",
       "XGB_case_weather_pois", "XGB_full_pois", "CAT_full_pois", "RF_full"]

# 1. Overall error by horizon (RMSE and MAE are the fairest headline numbers)
for scope in ["All cities", "Tier 1"]:
    d = overall[(overall["scope"] == scope) & overall["model"].isin(KEY)]
    for metric in ["RMSE", "MAE", "Poisson_deviance"]:
        show(f"{metric} | {scope}", d.pivot(index="model", columns="horizon", values=metric).reindex(KEY))

# 2. Ablation: percent improvement in Poisson deviance (all cities)
d = overall[overall["scope"] == "All cities"].pivot(index="model", columns="horizon", values="Poisson_deviance")
steps = {
    "Weather added (case_only -> case_weather)": ("XGB_case_only_pois", "XGB_case_weather_pois"),
    "Built environment added (case_weather -> full)": ("XGB_case_weather_pois", "XGB_full_pois"),
}
abl = pd.DataFrame({name: (d.loc[a] - d.loc[b]) / d.loc[a] * 100 for name, (a, b) in steps.items()}).T
show("Ablation: % reduction in Poisson deviance (XGBoost, Poisson loss)", abl)

# 3. Per-city results at horizon 2
c = by_city[(by_city["horizon"] == 2) & by_city["model"].isin(
    ["A_all_zero", "B_persistence", "XGB_case_weather_pois", "XGB_full_pois"])]
show("Per-city MAE at horizon 2", c.pivot(index="city", columns="model", values="MAE"))
show("Per-city Poisson deviance at horizon 2", c.pivot(index="city", columns="model", values="Poisson_deviance"))

(RESULTS_DIR / "summary_tables.txt").write_text("\n".join(out))
print(f"\nSaved to {RESULTS_DIR / 'summary_tables.txt'}")
