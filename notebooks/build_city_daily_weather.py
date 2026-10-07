"""Build daily city-level weather summaries for the historical exhibit replay."""

from build_training_data import aggregate_daily_to_city, load_location
from common import DATA_DIR


def main() -> None:
    location, _ = load_location()
    city_by_barangay = location.set_index("adm4_pcode")["adm3_en"]
    atmosphere = aggregate_daily_to_city(
        "climate_atmosphere.csv",
        ["pr", "rh", "tave"],
        city_by_barangay,
    )
    daily_weather = (
        atmosphere.rename(
            columns={
                "pr": "rainfall_mm",
                "rh": "relative_humidity_pct",
                "tave": "temperature_c",
            }
        )
        .reset_index()
        .sort_values(["adm3_en", "date"])
        .rename(columns={"adm3_en": "city"})
    )
    output_path = DATA_DIR / "city_daily_weather.csv"
    daily_weather.to_csv(output_path, index=False)
    print(f"Saved {len(daily_weather):,} city-day weather rows to {output_path}")


if __name__ == "__main__":
    main()
