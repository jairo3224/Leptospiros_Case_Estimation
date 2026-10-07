"""Shared settings and helper functions for the leptospirosis ML experiments."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, mean_squared_error

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
BASE_CSV = DATA_DIR / "final_training_data.csv"
HORIZON_CSV = DATA_DIR / "final_training_data_horizons.csv"

CASE_FEATURES = ["cases_lag1", "cases_lag2", "case_velocity"]
WEATHER_FEATURES = [
    "rain_sum_7d_lag", "rain_sum_14d_lag", "rain_sum_30d_lag",
    "humidity_mean_14d_lag", "temperature_mean_14d_lag", "ndvi_mean_14d_lag",
]
STATIC_FEATURES = [
    "google_bldgs_density", "google_bldgs_pct_built_up_area",
    "pct_area_flood_hazard_100yr_high", "brgy_distance_to_coast", "brgy_is_coastal",
    "pop_density_mean", "toilet_count", "pop_count_total",
    "population_per_km2_2020", "toilet_count_per_10000_pop",
]
FEATURE_SETS = {
    "case_only": CASE_FEATURES,
    "case_weather": CASE_FEATURES + WEATHER_FEATURES,
    "full": CASE_FEATURES + WEATHER_FEATURES + STATIC_FEATURES,
}
# Used to group SHAP values by category
FEATURE_GROUPS = {
    "Rainfall": ["rain_sum_7d_lag", "rain_sum_14d_lag", "rain_sum_30d_lag"],
    "Environmental persistence": ["humidity_mean_14d_lag", "temperature_mean_14d_lag", "ndvi_mean_14d_lag"],
    "Case momentum": CASE_FEATURES,
    "Built environment": STATIC_FEATURES,
}

TRAIN_END = pd.Timestamp("2018-12-31")
TEST_START = pd.Timestamp("2019-02-04")  # ~5-week embargo after the training period
HORIZONS = [0, 1, 2, 3]  # weeks after the week the features describe (0 = the week itself)
RANDOM_STATE = 42


def target_col(h: int) -> str:
    return "weekly_cases" if h == 0 else f"target_h{h}"


def load_data() -> pd.DataFrame:
    if not HORIZON_CSV.is_file():
        raise FileNotFoundError(f"{HORIZON_CSV} not found. Run build_horizon_targets.py first.")
    df = pd.read_csv(HORIZON_CSV, parse_dates=["date"])
    return df.sort_values(["adm3_en", "date"]).reset_index(drop=True)


def split_by_year(df: pd.DataFrame, h: int):
    """Chronological split. Training rows must have their TARGET week inside the
    training years, so no 2019+ information leaks into training."""
    d = df.dropna(subset=[target_col(h)])
    target_date = d["date"] + pd.Timedelta(weeks=h)
    train = d[target_date <= TRAIN_END]
    test = d[d["date"] >= TEST_START]
    return train, test


def city_tiers(df: pd.DataFrame, zero_cutoff: float = 0.80) -> dict:
    """Tier 1 = fewer than 80% zero-case weeks in the TRAINING period."""
    tr = df[df["date"] <= TRAIN_END]
    zero_share = tr.groupby("adm3_en")["weekly_cases"].apply(lambda s: (s == 0).mean())
    return {c: ("Tier 1" if z < zero_cutoff else "Tier 2") for c, z in zero_share.items()}


def make_xgb(objective: str = "count:poisson"):
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective=objective, n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1,
    )


def make_catboost(loss: str = "Poisson"):
    """loss: "Poisson" (count data) or "RMSE" (squared error)."""
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function=loss, iterations=300, depth=4, learning_rate=0.05,
        random_seed=RANDOM_STATE, verbose=0, thread_count=-1,
    )


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    nz = y_true > 0
    out = {
        "n": len(y_true),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAE_nonzero": float(mean_absolute_error(y_true[nz], y_pred[nz])) if nz.any() else np.nan,
        "RMSE_nonzero": float(np.sqrt(mean_squared_error(y_true[nz], y_pred[nz]))) if nz.any() else np.nan,
        "Poisson_deviance": float(mean_poisson_deviance(y_true, np.clip(y_pred, 1e-3, None))),
    }
    return out
