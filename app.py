from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
TRAIN_END = pd.Timestamp("2018-12-31")
HORIZONS = (0, 1, 2, 3)
MODEL_LABELS = {
    "XGB_full_pois": "XGBoost — cases, weather, and city features",
    "XGB_case_weather_pois": "XGBoost — cases and weather",
    "CAT_full_pois": "CatBoost — cases, weather, and city features",
    "C_poisson_regression": "Poisson regression",
    "B_persistence": "Persistence baseline — last week's cases",
    "A_all_zero": "Zero-case baseline",
}
MODEL_INPUTS = {
    "XGB_full_pois": "recent case counts, lagged weather summaries, and city-level features",
    "XGB_case_weather_pois": "recent case counts and lagged weather summaries",
    "CAT_full_pois": "recent case counts, lagged weather summaries, and city-level features",
    "C_poisson_regression": "recent case counts, lagged weather summaries, and city-level features",
}


def target_week(anchor_date: pd.Timestamp, horizon: int) -> pd.Timestamp:
    if horizon not in HORIZONS:
        raise ValueError(f"Unsupported forecast horizon: {horizon}")
    return pd.Timestamp(anchor_date) + pd.Timedelta(weeks=horizon)


def get_saved_prediction(
    predictions: pd.DataFrame,
    city: str,
    anchor_date: pd.Timestamp,
    horizon: int,
    model: str,
) -> pd.Series:
    matches = predictions.loc[
        predictions["city"].eq(city)
        & predictions["date"].eq(pd.Timestamp(anchor_date))
        & predictions["horizon"].eq(horizon)
        & predictions["model"].eq(model)
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one saved prediction for the selected city, "
            f"week, horizon, and model; found {len(matches)}."
        )
    return matches.iloc[0]


def get_weather_week(
    daily_weather: pd.DataFrame,
    city: str,
    week_start: pd.Timestamp,
) -> pd.DataFrame:
    week_start = pd.Timestamp(week_start)
    week_end = week_start + pd.Timedelta(days=7)
    return (
        daily_weather.loc[
            daily_weather["city"].eq(city)
            & daily_weather["date"].ge(week_start)
            & daily_weather["date"].lt(week_end)
        ]
        .sort_values("date")
        .copy()
    )


def explain_replay_difference(
    model: str,
    estimate: float,
    actual: float,
) -> str:
    difference = actual - estimate
    if difference > 0:
        comparison = f"The estimate was {difference:.2f} cases lower than the recorded count."
    elif difference < 0:
        comparison = f"The estimate was {abs(difference):.2f} cases higher than the recorded count."
    else:
        comparison = "The estimate matched the recorded count."

    if model == "A_all_zero":
        model_note = (
            "The zero-case baseline always predicts zero and ignores the case and "
            "weather data, so it will miss any week with reported cases."
        )
    elif model == "B_persistence":
        model_note = (
            "The persistence baseline repeats the previous week's case count, "
            "so a sudden increase or decrease can lead to a miss."
        )
    else:
        inputs = MODEL_INPUTS.get(model)
        if inputs is None:
            raise ValueError(f"No explanation is available for model {model!r}.")
        model_note = (
            f"This model uses {inputs}. Those features do not capture every "
            "factor affecting reported cases, and this replay cannot identify "
            "which factor caused the difference."
        )

    return (
        f"{comparison} {model_note} This is one historical test week; the "
        "held-out MAE shown above summarizes performance across many weeks "
        "for this city and horizon."
    )


@st.cache_data
def load_replay_data() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    prediction_path = RESULTS_DIR / "predictions.csv"
    training_path = DATA_DIR / "final_training_data.csv"
    metrics_path = RESULTS_DIR / "metrics_by_city.csv"
    weather_path = DATA_DIR / "city_daily_weather.csv"
    for path in (prediction_path, training_path, metrics_path, weather_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required exhibit data not found: {path}")

    predictions = pd.read_csv(prediction_path, parse_dates=["date"])
    training = pd.read_csv(training_path, parse_dates=["date"])
    metrics = pd.read_csv(metrics_path)
    weather = pd.read_csv(weather_path, parse_dates=["date"])
    required_predictions = {"date", "city", "horizon", "model", "y_true", "y_pred"}
    missing = required_predictions - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions.csv is missing columns: {sorted(missing)}")
    if predictions.duplicated(["date", "city", "horizon", "model"]).any():
        raise ValueError("predictions.csv contains duplicate prediction keys.")
    required_training = {
        "date",
        "adm3_en",
        "weekly_cases",
        "pop_count_total",
        "pct_area_flood_hazard_100yr_high",
    }
    missing = required_training - set(training.columns)
    if missing:
        raise ValueError(f"final_training_data.csv is missing columns: {sorted(missing)}")
    required_weather = {
        "date",
        "city",
        "rainfall_mm",
        "temperature_c",
        "relative_humidity_pct",
    }
    missing = required_weather - set(weather.columns)
    if missing:
        raise ValueError(f"city_daily_weather.csv is missing columns: {sorted(missing)}")
    if weather.duplicated(["city", "date"]).any():
        raise ValueError("city_daily_weather.csv contains duplicate city-day rows.")

    training_period = training.loc[training["date"].le(TRAIN_END)]
    profiles = (
        training_period.sort_values("date")
        .groupby("adm3_en")
        .agg(
            population=("pop_count_total", "first"),
            flood_share=("pct_area_flood_hazard_100yr_high", "first"),
            usual_weekly_cases=("weekly_cases", "mean"),
        )
    )
    if profiles.empty:
        raise ValueError("No city profiles are available from the model training period.")
    weekly_cases = (
        training[["date", "adm3_en", "weekly_cases"]]
        .rename(columns={"adm3_en": "city"})
        .sort_values(["city", "date"])
    )
    return predictions, profiles, metrics, weather, weekly_cases


def render_header() -> None:
    st.markdown(
        """
        <style>
        :root {font-family: Inter, "Segoe UI", Arial, sans-serif;}
        [data-testid="stAppViewContainer"] {background: #f5f7fa;}
        [data-testid="stHeader"] {background: rgba(245, 247, 250, .9);}
        [data-testid="stMainBlockContainer"] {max-width: 1260px; padding-top: 2rem;}
        h1, h2, h3 {color: #142c46; letter-spacing: -.025em;}
        h1 {font-size: 2.35rem !important; font-weight: 720 !important;}
        h2, h3 {font-weight: 650 !important;}
        p, label, [data-testid="stCaptionContainer"] {color: #425166;}
        [data-testid="stMetric"] {
          background: #fff; border: 1px solid #e4eaf1; border-radius: 12px;
          padding: 1rem 1.1rem; box-shadow: 0 2px 8px rgba(20, 44, 70, .035);
        }
        [data-testid="stMetricLabel"] {color: #586a7f; font-weight: 600;}
        [data-testid="stMetricValue"] {color: #142c46; font-weight: 700;}
        [data-testid="stTabs"] button[role="tab"] {font-weight: 650;}
        [data-testid="stDataFrame"], [data-testid="stVegaLiteChart"] {
          border: 1px solid #e4eaf1; border-radius: 12px; overflow: hidden;
        }
        .top-stripe {height: 5px; margin: -1rem -1rem 1.2rem;
          background: linear-gradient(to right, #1769aa 0 33.33%, #bf3030 33.33% 66.66%, #efc431 66.66%);}
        .prototype {display: inline-block; padding: .28rem .7rem; border-radius: 999px;
          background: #e7eef6; color: #17324d; font-size: .78rem; font-weight: 750; letter-spacing: .06em;}
        .subtitle {color: #526276; margin-top: -.6rem; font-size: 1.05rem;}
        .section-intro {color: #586a7f; margin-top: -.7rem; margin-bottom: 1rem;}
        </style>
        <div class="top-stripe"></div>
        <span class="prototype">RESEARCH PROTOTYPE</span>
        """,
        unsafe_allow_html=True,
    )
    st.title("Leptospirosis Weekly Case Estimates")
    st.markdown(
        '<p class="subtitle">Choose a city, explore a historical estimate, '
        "and see how model estimates compare with recorded cases.</p>",
        unsafe_allow_html=True,
    )


def render_replay(
    predictions: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    daily_weather: pd.DataFrame,
    weekly_cases: pd.DataFrame,
) -> None:
    city_names = sorted(predictions["city"].dropna().unique().tolist())
    city = st.selectbox("Choose a study city", city_names)
    city_predictions = predictions.loc[predictions["city"].eq(city)]
    city_metrics = metrics.loc[metrics["city"].eq(city)]

    left, right = st.columns([1, 1])
    with left:
        horizon = st.selectbox(
            "How far ahead?",
            HORIZONS,
            format_func=lambda value: (
                "Same-week estimate (horizon 0)"
                if value == 0
                else f"{value} week{'s' if value != 1 else ''} ahead"
            ),
        )
    horizon_predictions = city_predictions.loc[
        city_predictions["horizon"].eq(horizon)
    ]
    anchor_dates = sorted(horizon_predictions["date"].drop_duplicates())
    if not anchor_dates:
        st.error("No historical replay weeks are available for this selection.")
        return
    with right:
        anchor_date = st.selectbox(
            "Historical input week",
            anchor_dates,
            index=len(anchor_dates) - 1,
            format_func=lambda value: pd.Timestamp(value).strftime("%B %d, %Y"),
        )
    anchor_date = pd.Timestamp(anchor_date)
    outcome_date = target_week(anchor_date, int(horizon))

    profile = profiles.loc[city]
    st.subheader(city)
    profile_columns = st.columns(3)
    profile_columns[0].metric(
        "Population in study data", f"{profile['population']:,.0f}"
    )
    profile_columns[1].metric(
        "High flood-hazard area", f"{profile['flood_share']:.1f}%"
    )
    profile_columns[2].metric(
        "Usual weekly cases",
        f"{profile['usual_weekly_cases']:.2f}",
        help="Average during the model training period, through December 2018.",
    )

    st.markdown("#### Weather during the selected input week")
    st.markdown(
        '<p class="section-intro">Observed daily citywide weather for the '
        "selected historical week.</p>",
        unsafe_allow_html=True,
    )
    weather_week = get_weather_week(daily_weather, city, anchor_date)
    if weather_week.empty:
        st.info("No daily weather observations are available for this city and week.")
    else:
        weather_display = weather_week[
            ["date", "rainfall_mm", "temperature_c", "relative_humidity_pct"]
        ].rename(
            columns={
                "date": "Date",
                "rainfall_mm": "Rainfall (mm/day)",
                "temperature_c": "Mean temperature (°C)",
                "relative_humidity_pct": "Mean relative humidity (%)",
            }
        )
        weather_display.insert(
            1,
            "Rain recorded",
            weather_display["Rainfall (mm/day)"].map(
                lambda amount: (
                    "Unavailable"
                    if pd.isna(amount)
                    else "Yes"
                    if amount > 0
                    else "No"
                )
            ),
        )
        weather_display["Date"] = weather_display["Date"].dt.strftime("%b %d, %Y")
        st.dataframe(weather_display, hide_index=True, width="stretch")
        st.caption(
            f"Showing {len(weather_display)} of 7 days in this Monday-starting week. "
            "Values are daily citywide averages from the study weather data. "
            "“Rain recorded” reports whether precipitation was greater than zero; "
            "it does not classify rain intensity or imply sunny conditions."
        )

    model_options = [
        model
        for model in MODEL_LABELS
        if model in horizon_predictions["model"].unique()
    ]
    if not model_options:
        st.error("No supported comparison models are available for this selection.")
        return
    selected_label = st.selectbox(
        "Model to replay",
        [MODEL_LABELS[model] for model in model_options],
        index=0,
    )
    selected_model = next(
        model for model in model_options if MODEL_LABELS[model] == selected_label
    )

    prediction = get_saved_prediction(
        predictions, city, anchor_date, int(horizon), selected_model
    )
    actual = float(prediction["y_true"])
    estimate = max(0.0, float(prediction["y_pred"]))
    baseline_values: dict[str, float] = {}
    for baseline in ("A_all_zero", "B_persistence"):
        if baseline in horizon_predictions["model"].values:
            baseline_row = get_saved_prediction(
                predictions, city, anchor_date, int(horizon), baseline
            )
            baseline_values[baseline] = max(0.0, float(baseline_row["y_pred"]))

    metric_match = city_metrics.loc[
        city_metrics["horizon"].eq(horizon)
        & city_metrics["model"].eq(selected_model)
    ]
    if len(metric_match) != 1:
        raise ValueError(
            "Expected exactly one held-out metric for the selected city, "
            "horizon, and model."
        )

    st.divider()
    st.markdown(
        f"### Estimate for the week of {outcome_date.strftime('%B %d, %Y')}"
    )
    st.markdown(
        '<p class="section-intro">Historical model estimate and reference '
        "values. The visualization controls below do not change model results.</p>",
        unsafe_allow_html=True,
    )
    result_columns = st.columns(3)
    result_columns[0].metric("Estimated weekly cases", f"{estimate:.2f}")
    result_columns[1].metric(
        "All-zero baseline", f"{baseline_values.get('A_all_zero', 0.0):.2f}"
    )
    result_columns[2].metric(
        "Persistence baseline",
        f"{baseline_values.get('B_persistence', float('nan')):.2f}",
    )
    st.caption(
        f"Selected model: {selected_label}. Its mean absolute error for this "
        f"city and horizon across the held-out test period was "
        f"{float(metric_match.iloc[0]['MAE']):.2f} cases "
        f"(n={int(metric_match.iloc[0]['n']):,} city-weeks)."
    )

    usual_cases = float(profile["usual_weekly_cases"])
    meter_scale = max(usual_cases * 2, estimate, 1.0)
    meter_ratio = estimate / meter_scale
    if usual_cases > 0:
        usual_ratio = estimate / usual_cases
        usual_text = f"{usual_ratio:.1f}× the training-period average"
    else:
        usual_text = "No non-zero training-period average to compare"
    st.markdown("#### Estimate compared with a usual week")
    st.progress(
        meter_ratio,
        text=(
            f"{estimate:.2f} estimated cases on a visual scale of "
            f"0–{meter_scale:.2f} cases"
        ),
    )
    st.caption(
        f"That is {usual_text}. The meter is a visual reference, not a "
        "probability, risk score, or uncertainty interval."
    )

    with st.expander("Visualization controls — display only"):
        st.caption(
            "These controls change the charts only. They do not adjust the "
            "saved prediction or simulate a new model result."
        )
        history_weeks = st.slider(
            "Weeks of case history to show",
            min_value=8,
            max_value=52,
            value=26,
            step=2,
        )
        rain_threshold = st.slider(
            "Rainfall reference line (mm/day)",
            min_value=0.0,
            max_value=10.0,
            value=1.0,
            step=0.5,
            help="Display reference only; does not classify rain intensity.",
        )

    # End before the selected input week so this context chart cannot reveal
    # the selected target outcome in same-week mode.
    case_history = weekly_cases.loc[
        weekly_cases["city"].eq(city)
        & weekly_cases["date"].lt(anchor_date)
    ].tail(history_weeks)
    if not case_history.empty:
        case_chart = case_history.set_index("date")[["weekly_cases"]].rename(
            columns={"weekly_cases": "Recorded cases"}
        )
        case_chart["Usual weekly average"] = usual_cases
        st.markdown("#### Recent case history")
        st.markdown(
            '<p class="section-intro">Recorded weekly cases before the selected '
            "input week; the selected week’s outcome remains hidden until "
            "revealed.</p>",
            unsafe_allow_html=True,
        )
        st.line_chart(
            case_chart,
            y=["Recorded cases", "Usual weekly average"],
            color=["#1769aa", "#bf3030"],
            x_label="Week",
            y_label="Reported cases",
        )

    if not weather_week.empty:
        st.markdown("#### Daily weather visualization")
        rain_chart = weather_week.set_index("date")[["rainfall_mm"]].rename(
            columns={"rainfall_mm": "Rainfall (mm/day)"}
        )
        rain_chart["Visual reference"] = rain_threshold
        weather_chart_columns = st.columns(3)
        with weather_chart_columns[0]:
            st.markdown("**Rainfall**")
            st.line_chart(
                rain_chart,
                y=["Rainfall (mm/day)", "Visual reference"],
                color=["#1769aa", "#bf3030"],
                x_label="Day",
                y_label="Rainfall (mm/day)",
            )
        with weather_chart_columns[1]:
            st.markdown("**Mean temperature**")
            temperature_chart = weather_week.set_index("date")[
                ["temperature_c"]
            ].rename(columns={"temperature_c": "Temperature (°C)"})
            st.line_chart(
                temperature_chart,
                y=["Temperature (°C)"],
                color="#bf3030",
                x_label="Day",
                y_label="Temperature (°C)",
            )
        with weather_chart_columns[2]:
            st.markdown("**Relative humidity**")
            humidity_chart = weather_week.set_index("date")[
                ["relative_humidity_pct"]
            ].rename(columns={"relative_humidity_pct": "Relative humidity (%)"})
            st.line_chart(
                humidity_chart,
                y=["Relative humidity (%)"],
                color="#1769aa",
                x_label="Day",
                y_label="Relative humidity (%)",
            )

    reveal_key = (
        f"reveal-{city}-{anchor_date.date()}-{horizon}-{selected_model}"
    )
    if st.button("Reveal the recorded outcome", key=f"button-{reveal_key}"):
        st.metric("Recorded cases for the target week", f"{actual:.0f}")
        st.info(
            "#### Why might the estimate differ?\n\n"
            + explain_replay_difference(selected_model, estimate, actual)
        )
    else:
        st.caption("The recorded outcome stays hidden until you choose to reveal it.")
    st.caption(
        "Replay uses saved predictions for historical test weeks "
        "(February 2019–December 2022). Horizon 0 is a same-week estimate; "
        "horizons 1–3 predict later weeks."
    )


def main() -> None:
    st.set_page_config(
        page_title="Leptospirosis Research Prototype",
        page_icon="📊",
        layout="wide",
    )
    render_header()
    predictions, profiles, metrics, weather, weekly_cases = load_replay_data()
    replay_tab, future_tab = st.tabs(
        ["Compare with a past week", "Explore a future estimate"]
    )
    with replay_tab:
        render_replay(predictions, profiles, metrics, weather, weekly_cases)
    with future_tab:
        st.subheader("Future estimate")
        st.info(
            "This mode is not available yet. The case-count data in this study "
            "end on December 26, 2022. Open-Meteo can provide weather data, but "
            "weather alone is not enough: the model also needs recent case "
            "counts. Update the case data, retrain the horizon-specific models, "
            "and evaluate them on later weeks before presenting a current "
            "forecast."
        )
        st.caption(
            "The historical replay remains available and does not make a "
            "current prediction."
        )
    st.divider()
    st.caption(
        "Research demonstration only — not an official Department of Health "
        "product and not for public-health decision-making."
    )


if __name__ == "__main__":
    main()
