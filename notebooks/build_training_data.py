from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_PATH = DATA_DIR / "final_training_data.csv"
CHUNK_SIZE = 250_000

ATMOSPHERE_FEATURES = ["pr", "rh", "tave"]
LAND_FEATURES = ["ndvi"]
SANITATION_FEATURES = ["toilet_count"]
STATIC_WEIGHTED_FEATURES = [
    "google_bldgs_density",
    "google_bldgs_pct_built_up_area",
    "pct_area_flood_hazard_100yr_high",
    "brgy_distance_to_coast",
    "brgy_is_coastal",
    "pop_density_mean",
]
WEATHER_FEATURES = [
    "rain_sum_7d_lag",
    "rain_sum_14d_lag",
    "rain_sum_30d_lag",
    "humidity_mean_14d_lag",
    "temperature_mean_14d_lag",
    "ndvi_mean_14d_lag",
]


def require_file(filename: str) -> Path:
    path = DATA_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Required dataset not found: {path}")
    return path


def require_columns(path: Path, columns: list[str]) -> None:
    available = set(pd.read_csv(path, nrows=0).columns)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")


def convert_numeric(data: pd.DataFrame, features: list[str], filename: str) -> None:
    for feature in features:
        raw = data[feature]
        numeric = pd.to_numeric(raw, errors="coerce")
        invalid = raw.notna() & numeric.isna()
        if invalid.any():
            examples = raw.loc[invalid].astype(str).unique()[:5].tolist()
            raise ValueError(
                f"{filename} has non-numeric values in {feature}: {examples}"
            )
        data[feature] = numeric


def load_location() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = require_file("location.csv")
    columns = ["adm3_en", "adm3_pcode", "adm4_pcode", "brgy_total_area"]
    require_columns(path, columns)
    location = pd.read_csv(path, usecols=columns)

    if location[["adm3_en", "adm3_pcode", "adm4_pcode"]].isna().any().any():
        raise ValueError("location.csv contains missing administrative mapping keys.")
    if location["adm4_pcode"].duplicated().any():
        raise ValueError("location.csv must contain one row per adm4_pcode.")
    if location.groupby("adm3_pcode")["adm3_en"].nunique().gt(1).any():
        raise ValueError("An adm3_pcode maps to more than one city name.")

    city_codes = location[["adm3_pcode", "adm3_en"]].drop_duplicates()
    return location, city_codes


def load_calendar() -> pd.DatetimeIndex:
    path = require_file("calendar.csv")
    require_columns(path, ["date"])
    calendar = pd.read_csv(path, usecols=["date"], parse_dates=["date"])
    if calendar["date"].isna().any() or calendar["date"].duplicated().any():
        raise ValueError("calendar.csv must contain unique, valid dates.")

    dates = pd.DatetimeIndex(calendar["date"].sort_values())
    if len(dates) > 1 and not dates.to_series().diff().dropna().eq(
        pd.Timedelta(days=1)
    ).all():
        raise ValueError("calendar.csv must contain a continuous daily date range.")
    return dates


def aggregate_daily_to_city(
    filename: str,
    features: list[str],
    city_by_barangay: pd.Series,
) -> pd.DataFrame:
    path = require_file(filename)
    required = ["adm4_pcode", "date", *features]
    require_columns(path, required)

    partial_sums: list[pd.DataFrame] = []
    partial_counts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(path, usecols=required, chunksize=CHUNK_SIZE):
        chunk["date"] = pd.to_datetime(chunk["date"], errors="coerce")
        if chunk["date"].isna().any():
            raise ValueError(f"{filename} contains invalid dates.")

        chunk["adm3_en"] = chunk["adm4_pcode"].map(city_by_barangay)
        if chunk["adm3_en"].isna().any():
            unknown = chunk.loc[chunk["adm3_en"].isna(), "adm4_pcode"].unique()
            raise ValueError(
                f"{filename} contains barangay codes missing from location.csv: "
                f"{unknown[:10].tolist()}"
            )

        convert_numeric(chunk, features, filename)

        keys = ["adm3_en", "date"]
        grouped = chunk.groupby(keys, sort=False)[features]
        partial_sums.append(grouped.sum(min_count=1))
        partial_counts.append(grouped.count())

    if not partial_sums:
        raise ValueError(f"{filename} contains no data rows.")

    sums = pd.concat(partial_sums).groupby(level=[0, 1], sort=False).sum(min_count=1)
    counts = pd.concat(partial_counts).groupby(level=[0, 1], sort=False).sum()
    means = sums.div(counts.where(counts.gt(0)))
    means.index = means.index.set_names(["adm3_en", "date"])
    return means


def make_lagged_weather_features(
    daily_weather: pd.DataFrame,
    city_names: list[str],
    calendar_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    complete_index = pd.MultiIndex.from_product(
        [city_names, calendar_dates], names=["adm3_en", "date"]
    )
    daily_weather = daily_weather.reindex(complete_index)
    city_features: list[pd.DataFrame] = []

    for city in city_names:
        daily = daily_weather.xs(city, level="adm3_en")
        features = pd.DataFrame(index=daily.index)
        features["rain_sum_7d_lag"] = (
            daily["pr"].rolling(7, min_periods=7).sum().shift(1)
        )
        features["rain_sum_14d_lag"] = (
            daily["pr"].rolling(14, min_periods=14).sum().shift(1)
        )
        features["rain_sum_30d_lag"] = (
            daily["pr"].rolling(30, min_periods=30).sum().shift(1)
        )
        features["humidity_mean_14d_lag"] = (
            daily["rh"].rolling(14, min_periods=14).mean().shift(1)
        )
        features["temperature_mean_14d_lag"] = (
            daily["tave"].rolling(14, min_periods=14).mean().shift(1)
        )
        features["ndvi_mean_14d_lag"] = (
            daily["ndvi"].rolling(14, min_periods=14).mean().shift(1)
        )
        features = features.loc[features.index.dayofweek == 0].copy()
        features["adm3_en"] = city
        city_features.append(features.reset_index())

    return pd.concat(city_features, ignore_index=True)


def latest_barangay_values(filename: str, features: list[str]) -> pd.DataFrame:
    path = require_file(filename)
    required = ["adm4_pcode", "date", *features]
    require_columns(path, required)
    data = pd.read_csv(path, usecols=required)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    if data["date"].isna().any():
        raise ValueError(f"{filename} contains invalid dates.")

    data = data.sort_values("date").drop_duplicates("adm4_pcode", keep="last")
    if data["adm4_pcode"].duplicated().any():
        raise ValueError(f"{filename} has duplicate latest rows per barangay.")
    convert_numeric(data, features, filename)
    return data.drop(columns="date")


def make_city_static_features(location: pd.DataFrame) -> pd.DataFrame:
    location_columns = ["adm3_en", "adm4_pcode", "brgy_total_area"]
    static = location[location_columns].copy()
    sources = [
        (
            "google_open_buildings.csv",
            ["google_bldgs_density", "google_bldgs_pct_built_up_area"],
        ),
        (
            "project_noah_hazards.csv",
            ["pct_area_flood_hazard_100yr_high"],
        ),
        (
            "brgy_geography.csv",
            ["brgy_distance_to_coast", "brgy_is_coastal"],
        ),
        (
            "worldpop_population.csv",
            ["pop_density_mean", "pop_count_total"],
        ),
        ("osm_poi_sanitation.csv", SANITATION_FEATURES),
    ]

    for filename, features in sources:
        latest = latest_barangay_values(filename, features)
        static = static.merge(
            latest, on="adm4_pcode", how="left", validate="one_to_one"
        )

    if static["brgy_total_area"].isna().any() or static["brgy_total_area"].le(0).any():
        raise ValueError("location.csv must provide positive area for every barangay.")

    grouped = static.groupby("adm3_en", sort=False)
    city = pd.DataFrame(index=grouped.size().index)
    city.index.name = "adm3_en"
    for feature in STATIC_WEIGHTED_FEATURES:
        valid = static[feature].notna() & static["brgy_total_area"].gt(0)
        weighted_sum = (
            static.loc[valid]
            .assign(_weighted=lambda frame: frame[feature] * frame["brgy_total_area"])
            .groupby("adm3_en")["_weighted"]
            .sum()
        )
        area_sum = (
            static.loc[valid]
            .groupby("adm3_en")["brgy_total_area"]
            .sum()
        )
        city[feature] = weighted_sum.div(area_sum)

    for feature in SANITATION_FEATURES:
        city[feature] = grouped[feature].sum(min_count=1)

    city["pop_count_total"] = grouped["pop_count_total"].sum(min_count=1)
    city_area = grouped["brgy_total_area"].sum()
    city["population_per_km2_2020"] = city["pop_count_total"].div(city_area)
    city["toilet_count_per_10000_pop"] = (
        city["toilet_count"].mul(10_000).div(city["pop_count_total"])
    )
    city = city.reset_index()
    return city


def main() -> None:
    print("Loading location and calendar...")
    location, city_codes = load_location()
    calendar_dates = load_calendar()
    city_by_barangay = location.set_index("adm4_pcode")["adm3_en"]
    city_names = sorted(location["adm3_en"].unique())

    disease_path = require_file("disease_pidsr_totals.csv")
    disease_columns = [
        "date",
        "adm3_pcode",
        "disease_common_name",
        "case_total",
    ]
    require_columns(disease_path, disease_columns)
    disease = pd.read_csv(disease_path, usecols=disease_columns)
    lepto = disease.loc[
        disease["disease_common_name"]
        .astype("string")
        .str.strip()
        .str.casefold()
        .eq("leptospirosis")
    ].copy()
    if lepto.empty:
        raise ValueError("No leptospirosis rows found in disease_pidsr_totals.csv.")

    lepto["date"] = pd.to_datetime(lepto["date"], errors="coerce")
    lepto["weekly_cases"] = pd.to_numeric(lepto.pop("case_total"), errors="coerce")
    if lepto[["date", "adm3_pcode", "weekly_cases"]].isna().any().any():
        raise ValueError("Leptospirosis rows contain missing or invalid key/target values.")
    if not lepto["date"].isin(calendar_dates).all():
        raise ValueError("Some disease dates are not present in calendar.csv.")

    lepto = lepto.merge(
        city_codes, on="adm3_pcode", how="left", validate="many_to_one"
    )
    if lepto["adm3_en"].isna().any():
        raise ValueError("Some disease adm3_pcode values are missing from location.csv.")
    lepto = lepto[["date", "adm3_pcode", "weekly_cases", "adm3_en"]]
    if not lepto["date"].dt.dayofweek.eq(0).all():
        raise ValueError("Disease dates are expected to be Mondays in the supplied data.")
    if lepto.duplicated(["adm3_en", "date"]).any():
        raise ValueError("Leptospirosis data has duplicate city-week rows.")

    lepto = lepto.sort_values(["adm3_en", "date"]).reset_index(drop=True)
    gaps = lepto.groupby("adm3_en", sort=False)["date"].diff().dropna()
    missing_week_gaps = int(gaps.gt(pd.Timedelta(days=7)).sum())
    if missing_week_gaps:
        print(
            f"Found {missing_week_gaps} gaps between reported case weeks; "
            "case lags will match exact dates, not treat the prior row as one week ago."
        )

    for lag_days, feature_name in [
        (7, "cases_lag1"),
        (14, "cases_lag2"),
    ]:
        lagged_cases = lepto[["adm3_en", "date", "weekly_cases"]].copy()
        lagged_cases["date"] += pd.Timedelta(days=lag_days)
        lagged_cases = lagged_cases.rename(columns={"weekly_cases": feature_name})
        lepto = lepto.merge(
            lagged_cases,
            on=["adm3_en", "date"],
            how="left",
            validate="one_to_one",
        )
    lepto["case_velocity"] = lepto["cases_lag1"] - lepto["cases_lag2"]

    print("Aggregating daily climate to city means in chunks...")
    atmosphere = aggregate_daily_to_city(
        "climate_atmosphere.csv", ATMOSPHERE_FEATURES, city_by_barangay
    )
    land = aggregate_daily_to_city(
        "climate_land.csv", LAND_FEATURES, city_by_barangay
    )
    daily_weather = atmosphere.join(land, how="outer", validate="one_to_one")
    weather_features = make_lagged_weather_features(
        daily_weather, city_names, calendar_dates
    )

    print("Aggregating static barangay features to city level...")
    static_city = make_city_static_features(location)

    print("Joining target, lagged weather, and static features...")
    final_training_data = lepto.merge(
        weather_features,
        on=["adm3_en", "date"],
        how="left",
        validate="one_to_one",
    ).merge(
        static_city,
        on="adm3_en",
        how="left",
        validate="many_to_one",
    )

    required_lags = [
        *WEATHER_FEATURES,
        "cases_lag1",
        "cases_lag2",
        "case_velocity",
    ]
    missing_by_feature = final_training_data[required_lags].isna().sum()
    missing_by_feature = missing_by_feature[missing_by_feature.gt(0)]
    if not missing_by_feature.empty:
        print("Rows with incomplete lag features by column:")
        print(missing_by_feature.to_string())
    original_rows = len(final_training_data)
    final_training_data = final_training_data.dropna(subset=required_lags).copy()
    dropped_rows = original_rows - len(final_training_data)
    if final_training_data.empty:
        raise ValueError("No training rows remain after requiring complete lag features.")

    final_training_data = final_training_data.sort_values(
        ["adm3_en", "date"]
    ).reset_index(drop=True)
    final_training_data.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(final_training_data):,} rows to {OUTPUT_PATH}")
    print(f"Removed {dropped_rows:,} rows with incomplete lag features.")
    print(
        "Note: calendar.csv contains dates only (no epi_week field); "
        "disease dates are retained as supplied."
    )
    optional_missing = final_training_data["pop_density_mean"].isna().sum()
    if optional_missing:
        print(
            f"Warning: pop_density_mean is missing for {optional_missing:,} rows; "
            "population_per_km2_2020 is also included as a complete alternative."
        )


if __name__ == "__main__":
    main()
