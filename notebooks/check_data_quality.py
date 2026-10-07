from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "final_training_data.csv"
df = pd.read_csv(data_path, parse_dates=["date"])

# which weeks are missing, across all cities
all_weeks = pd.date_range(df["date"].min(), df["date"].max(), freq="W-MON")
missing = sorted(set(all_weeks) - set(df["date"]))
print(len(missing), "missing weeks, e.g.:", [d.date() for d in missing[:12]])

# share of zero-case weeks per city, and mean weekly cases
summary = df.groupby("adm3_en")["weekly_cases"].agg(
    zero_share=lambda s: (s == 0).mean(), mean_cases="mean", max_cases="max"
)
print(summary.sort_values("zero_share", ascending=False))