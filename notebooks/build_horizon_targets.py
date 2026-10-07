"""Step 1: add 1-, 2-, and 3-week-ahead targets to final_training_data.csv.

Targets are matched by exact date within each city, so missing weeks never
pair a row with the wrong week. Output: data/final_training_data_horizons.csv
"""
import pandas as pd

from common import BASE_CSV, HORIZON_CSV

df = pd.read_csv(BASE_CSV, parse_dates=["date"])
base = df[["adm3_pcode", "date", "weekly_cases"]]

for h in (1, 2, 3):
    future = base.copy()
    # the row dated t+h supplies the target for the row dated t
    future["date"] = future["date"] - pd.Timedelta(weeks=h)
    future = future.rename(columns={"weekly_cases": f"target_h{h}"})
    df = df.merge(future, on=["adm3_pcode", "date"], how="left", validate="one_to_one")

df.to_csv(HORIZON_CSV, index=False)
print(f"Saved {len(df):,} rows to {HORIZON_CSV}")
print("Rows with no target (end of each city's series):")
print(df[["weekly_cases", "target_h1", "target_h2", "target_h3"]].isna().sum())
