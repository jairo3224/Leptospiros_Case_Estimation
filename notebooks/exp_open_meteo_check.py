"""Experiment 2: training-vs-serving check.

Your model was trained on CCHAIN weather (barangay values averaged to the city).
The app will fetch Open-Meteo at one point per city. This script fetches Open-Meteo
for the same cities and dates, builds the SAME lag features, and compares them
with the values in final_training_data.csv.

Needs internet. Fetched data is cached in data/openmeteo_cache_*.csv, so rerunning is fast.
Usage:  python exp_open_meteo_check.py
Outputs: results/openmeteo_vs_training.csv (+ printed table to paste)
"""
import time

import numpy as np
import pandas as pd
import requests

from common import BASE_CSV, DATA_DIR, RESULTS_DIR

# Approximate city-centre coordinates (edit if you prefer other points)
CITIES = {
    "Cagayan de Oro City": (8.4542, 124.6319), "City of Mandaluyong": (14.5794, 121.0359),
    "City of Muntinlupa": (14.4081, 121.0415), "City of Navotas": (14.6667, 120.9420),
    "Dagupan City": (16.0430, 120.3333), "Davao City": (7.0731, 125.6128),
    "Iloilo City": (10.7202, 122.5621), "Legazpi City": (13.1391, 123.7438),
    "Mandaue City": (10.3236, 123.9223), "Palayan City": (15.5436, 121.0846),
    "Tacloban City": (11.2543, 124.9632), "Zamboanga City": (6.9214, 122.0790),
}
URL = "https://archive-api.open-meteo.com/v1/archive"
CHUNKS = [("2007-11-01", "2012-12-31"), ("2013-01-01", "2017-12-31"), ("2018-01-01", "2022-12-31")]


def fetch_city(city, lat, lon):
    cache = DATA_DIR / f"openmeteo_cache_{city.replace(' ', '_')}.csv"
    if cache.is_file():
        return pd.read_csv(cache, parse_dates=["date"])
    parts = []
    for start, end in CHUNKS:
        params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
                  "daily": "precipitation_sum,temperature_2m_mean", "hourly": "relative_humidity_2m",
                  "timezone": "Asia/Manila"}
        for attempt in range(5):
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 200:
                break
            print(f"  {city} {start}: HTTP {r.status_code}, retrying...")
            time.sleep(10 * (attempt + 1))
        r.raise_for_status()
        js = r.json()
        daily = pd.DataFrame(js["daily"]).rename(columns={"time": "date"})
        hourly = pd.DataFrame(js["hourly"])
        hourly["date"] = pd.to_datetime(hourly["time"]).dt.normalize()
        rh = hourly.groupby("date")["relative_humidity_2m"].mean().rename("rh").reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        parts.append(daily.merge(rh, on="date", how="left"))
        time.sleep(3)
    out = pd.concat(parts).rename(columns={"precipitation_sum": "pr", "temperature_2m_mean": "tave"})
    out.to_csv(cache, index=False)
    return out


def lag_features(daily):
    d = daily.set_index("date").sort_index()
    f = pd.DataFrame(index=d.index)
    f["rain_sum_7d_lag"] = d["pr"].rolling(7, min_periods=7).sum().shift(1)
    f["rain_sum_14d_lag"] = d["pr"].rolling(14, min_periods=14).sum().shift(1)
    f["rain_sum_30d_lag"] = d["pr"].rolling(30, min_periods=30).sum().shift(1)
    f["humidity_mean_14d_lag"] = d["rh"].rolling(14, min_periods=14).mean().shift(1)
    f["temperature_mean_14d_lag"] = d["tave"].rolling(14, min_periods=14).mean().shift(1)
    return f


train = pd.read_csv(BASE_CSV, parse_dates=["date"])
cols = ["rain_sum_7d_lag", "rain_sum_14d_lag", "rain_sum_30d_lag", "humidity_mean_14d_lag", "temperature_mean_14d_lag"]
rows = []
for city, (lat, lon) in CITIES.items():
    print("fetching", city)
    om = lag_features(fetch_city(city, lat, lon))
    tr = train[train.adm3_en == city].set_index("date")
    j = tr[cols].join(om[cols], lsuffix="_cchain", rsuffix="_om", how="inner").dropna()
    for c in cols:
        a, b = j[c + "_cchain"], j[c + "_om"]
        rows.append({"city": city, "feature": c, "n_weeks": len(j), "correlation": a.corr(b),
                     "mean_cchain": a.mean(), "mean_openmeteo": b.mean(),
                     "bias_om_minus_cchain": (b - a).mean(), "MAE": (b - a).abs().mean(),
                     "ratio_of_means": b.mean() / a.mean() if a.mean() else np.nan})
res = pd.DataFrame(rows)
res.to_csv(RESULTS_DIR / "openmeteo_vs_training.csv", index=False)

print("\n=== Correlation (1.0 = identical pattern) ===")
print(res.pivot(index="city", columns="feature", values="correlation").round(2).to_string())
print("\n=== Ratio of means: Open-Meteo / CCHAIN (1.0 = same scale) ===")
print(res.pivot(index="city", columns="feature", values="ratio_of_means").round(2).to_string())
print("\n=== Average across cities ===")
print(res.groupby("feature")[["correlation", "ratio_of_means", "bias_om_minus_cchain", "MAE"]].mean().round(2).to_string())
