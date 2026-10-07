from pathlib import Path

import pandas as pd

project_dir = Path(__file__).resolve().parent.parent
data_path = project_dir / "data" / "final_training_data.csv"
df = pd.read_csv(data_path, parse_dates=["date"])

# 1) rows per city, and how many weeks are missing between first and last date
for city, g in df.groupby("adm3_en"):
    expected = (g["date"].max() - g["date"].min()).days // 7 + 1
    print(city, len(g), "of", expected, "expected weeks")

# 2) how many target weeks are zero
print((df["weekly_cases"] == 0).mean())