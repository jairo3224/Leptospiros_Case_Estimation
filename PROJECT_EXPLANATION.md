# Leptospirosis Weekly Case Estimation Project

## 1. Project in plain language

This project studies whether recent leptospirosis reports, weather, and selected city characteristics can help estimate weekly reported case counts in 12 Philippine cities. The analysis brings together weekly disease reports, daily climate records, and barangay-level geographic and population information. It then compares several statistical and machine-learning models on later historical weeks that were held out from training.

The final modelling table contains **9,238 city-week rows**, covering **January 21, 2008 to December 26, 2022**. It represents 12 cities and is assembled from information for 879 barangays. Most model inputs describe conditions before the week being estimated: earlier case counts, recent rainfall and other environmental measurements, and relatively stable city characteristics.

The Streamlit application is a **historical demonstration**, not a live forecasting service. It replays saved test-period predictions and compares them with recorded outcomes. It does not currently produce estimates for future weeks.

### A short presentation summary

> We built a city-week dataset by combining leptospirosis reports with lagged climate indicators and barangay-derived city characteristics. We trained and compared baseline, regression, and tree-based count-estimation models using a chronological training/test split. The evaluation shows that weather features can improve some measures—especially Poisson deviance—but the gains are not consistent across every metric or forecast horizon. The dashboard demonstrates historical held-out predictions; it is not a current public-health forecast.

## 2. How the project works

The main data flow is:

1. **Prepare the weekly case target and geographic links.** Filter the disease reports to leptospirosis and map administrative codes to city names.
2. **Build prior case and weather features.** Match earlier case counts by exact dates, aggregate daily weather from barangays to cities, and calculate lagged rolling summaries.
3. **Add city characteristics.** Combine static barangay indicators into city-level features, using area-weighted averages for many measures and sums for counts.
4. **Add future targets.** For each anchor week, attach the reported case count one, two, or three weeks later.
5. **Train and evaluate models.** Train on earlier years and test on later years, then save predictions and performance metrics.
6. **Explain and present the results.** Produce bootstrap intervals, SHAP feature summaries, tables, and a historical-replay dashboard.

The scripts are in [`notebooks/`](./notebooks/). Despite the folder name, these are Python scripts rather than Jupyter notebook files.

## 3. Explanation of the code

### Data preparation

- [`notebooks/build_training_data.py`](./notebooks/build_training_data.py) is the main data-construction script. It checks required files and columns, validates dates and geographic identifiers, filters `disease_pidsr_totals.csv` to leptospirosis, and joins city, weather, and static data. It calculates one- and two-week case lags, case velocity, rolling lagged weather summaries, and derived population/sanitation measures. It drops rows that lack required case or weather lag features and writes `data/final_training_data.csv`.
- [`notebooks/build_horizon_targets.py`](./notebooks/build_horizon_targets.py) adds `target_h1`, `target_h2`, and `target_h3`: the same city's observed weekly case count one, two, or three weeks after the row's date. It joins by exact city code and date, so a missing week is not mistakenly treated as the next week. Its output is `data/final_training_data_horizons.csv`.
- [`notebooks/build_city_daily_weather.py`](./notebooks/build_city_daily_weather.py) builds `data/city_daily_weather.csv` for the dashboard. It aggregates daily precipitation, relative humidity, and temperature to city level. This is a display dataset; it is separate from the lagged weather summaries used as model features.
- [`notebooks/common.py`](./notebooks/common.py) centralizes paths, feature lists, model settings, forecast horizons, the time split, city-tier definitions, data loading, and evaluation metrics. In particular, it defines the three feature sets used for model comparisons and constructs XGBoost and CatBoost models.

### Modelling and analysis

- [`notebooks/run_experiments.py`](./notebooks/run_experiments.py) trains the baseline and machine-learning models for horizons 0, 1, 2, and 3. It saves individual predictions and metrics overall, by city, and by city tier. With `--loco`, it also runs a leave-one-city-out analysis at horizon 2: each city is held out in turn while the model trains on the other cities.
- [`notebooks/bootstrap_ci.py`](./notebooks/bootstrap_ci.py) estimates 95% confidence intervals for selected model comparisons. It resamples test dates in blocks of eight consecutive weeks, keeping all cities in a sampled week together. This helps preserve some of the time dependence and cross-city correlation in the evaluation data. It uses 2,000 bootstrap samples and a fixed random seed.
- [`notebooks/shap_analysis.py`](./notebooks/shap_analysis.py) fits the full-feature XGBoost or CatBoost model for a selected horizon and calculates TreeSHAP feature contributions. It saves a beeswarm plot, grouped feature importance, and a CSV of mean absolute SHAP values. For the Poisson objective, the script notes that the SHAP values are on a log-count scale. These are explanations of model behaviour, not evidence that a feature causes cases.
- [`notebooks/summarize_results.py`](./notebooks/summarize_results.py) creates compact tables from the saved overall and per-city metrics. It includes an ablation comparison showing how scores change when weather or static city features are added.

### Checks, application, and tests

- [`notebooks/check_data_quality.py`](./notebooks/check_data_quality.py) reports weeks absent from the combined date set and summarizes zero-case shares, average cases, and maximum cases by city.
- [`notebooks/check_weeks.py`](./notebooks/check_weeks.py) checks how many weekly rows each city has compared with the span between its first and last dates, and reports the overall proportion of zero-case rows.
- [`app.py`](./app.py) is the Streamlit historical-replay interface. A user selects a city, forecast horizon, historical input week, and model. The app displays saved model estimates, reference baselines, held-out error for that city and horizon, weather during the input week, and prior case history. The recorded target outcome is hidden until the user chooses to reveal it. Its “future estimate” tab explicitly says that live forecasting is not available.
- [`tests/test_app.py`](./tests/test_app.py) contains unit tests for dashboard helper behaviour, including date horizons, selection of exactly one saved prediction, the seven-day weather window, and careful wording when explaining an estimate's difference from the recorded count.
- [`requirements.txt`](./requirements.txt) currently declares pandas and Streamlit. The experiment scripts also import NumPy, scikit-learn, XGBoost, CatBoost, SHAP, and Matplotlib, so the environment used to run the full analysis needs those packages as well.

## 4. Datasets

### Source and intermediate data

| File | What it contains and how the project uses it |
|---|---|
| [`data/disease_pidsr_totals.csv`](./data/disease_pidsr_totals.csv) | Weekly disease reports for 12 administrative city codes. It has 74,880 rows across multiple diseases; the pipeline filters the records to leptospirosis and uses `case_total` as the weekly case count. The source dates run from January 7, 2008 to December 26, 2022. |
| [`data/location.csv`](./data/location.csv) | 879 barangay records with administrative names/codes at several levels and barangay area. It links barangays (`adm4_pcode`) to cities (`adm3_en`, `adm3_pcode`) and provides areas used when aggregating features. |
| [`data/calendar.csv`](./data/calendar.csv) | A continuous daily calendar of 7,305 dates from January 1, 2003 through December 31, 2022. It is used to align daily climate observations and construct rolling windows. It contains dates only; it does not provide an epidemiological-week field. |
| [`data/climate_atmosphere.csv`](./data/climate_atmosphere.csv) | 6,421,095 barangay-day rows with atmospheric measurements, including precipitation (`pr`), relative humidity (`rh`), and average temperature (`tave`). The pipeline uses those three fields and aggregates them to city-day means. |
| [`data/climate_land.csv`](./data/climate_land.csv) | 6,421,095 barangay-day rows containing the land indicator `ndvi`. The pipeline aggregates it to city-day values and uses a prior 14-day mean in the model table. |
| [`data/brgy_geography.csv`](./data/brgy_geography.csv) | 879 barangay records containing area, distance to coast, coastal status, and geometry. The pipeline uses distance and coastal status as static city features; the geometry is not used as a model input. |
| [`data/google_open_buildings.csv`](./data/google_open_buildings.csv) | 879 barangay records with building counts and area indicators. The model table uses building density and percentage of built-up area. |
| [`data/project_noah_hazards.csv`](./data/project_noah_hazards.csv) | 879 barangay records with hazard-area proportions. The model table uses the percentage of area in the high 100-year flood-hazard category. |
| [`data/worldpop_population.csv`](./data/worldpop_population.csv) | 18,459 barangay-year records spanning 2000–2020 with population counts and density summaries. The pipeline takes the latest available barangay record, then derives city-level population totals and density measures. |
| [`data/osm_poi_sanitation.csv`](./data/osm_poi_sanitation.csv) | 7,911 dated barangay records (2014–2022) containing point-of-interest counts and distances. The model construction uses the latest available `toilet_count` per barangay and aggregates counts to cities. |

### Constructed modelling and dashboard tables

| File | Contents |
|---|---|
| [`data/final_training_data.csv`](./data/final_training_data.csv) | The main modelling table: 9,238 city-week rows and 23 columns, covering January 21, 2008 through December 26, 2022. It contains the observed count, city identifiers, case lags, lagged weather, and static city characteristics. |
| [`data/final_training_data_horizons.csv`](./data/final_training_data_horizons.csv) | The same 9,238 rows with three additional target columns (`target_h1`, `target_h2`, `target_h3`) for outcomes one to three weeks later. This is the table loaded for model experiments. |
| [`data/city_daily_weather.csv`](./data/city_daily_weather.csv) | 87,660 city-day rows (12 cities across the 2003–2022 calendar) with citywide mean rainfall, relative humidity, and temperature. The app uses it to display the weather for a selected historical input week. |
| [`final_training_data.csv`](./final_training_data.csv) | A byte-identical copy of `data/final_training_data.csv` at the project root. The scripts use the copy inside `data/`. |

### Model-result files

Files under [`results/`](./results/) are generated analysis outputs, not raw input datasets:

- `predictions.csv` stores one row per test prediction, with the anchor date, city, horizon, model, recorded value (`y_true`), and estimate (`y_pred`). The saved file has 151,408 rows.
- `metrics_overall.csv` reports performance by horizon, model, and scope (all cities, Tier 1, or Tier 2).
- `metrics_by_city.csv` reports the same types of metrics by individual city.
- `metrics_loco.csv` stores the optional leave-one-city-out results.
- `bootstrap_ci.csv` stores confidence intervals for selected pairwise comparisons.
- `shap_importance_xgboost_h2.csv` stores per-feature mean absolute SHAP values for a horizon-2 XGBoost run.
- `shap_beeswarm_*.png` and `shap_groups_*.png` are feature-explanation plots.
- `summary_tables.txt` is the formatted summary generated from saved metric tables.

The `catboost_info/` folders contain CatBoost training logs and progress information, rather than inputs to the modelling pipeline.

## 5. What the model table's features mean

Each row represents one **city and anchor week**. `weekly_cases` is the observed case count for that week. The model inputs are grouped as follows:

- **Recent cases:** `cases_lag1` is the previous week's count; `cases_lag2` is the count two weeks earlier; `case_velocity` is `cases_lag1 - cases_lag2`.
- **Weather and environment:** rainfall sums over the previous 7, 14, and 30 days; 14-day means of relative humidity, temperature, and NDVI. The rolling calculations are shifted back one day so they do not include observations from the anchor date or later.
- **Built environment and city characteristics:** building density and built-up percentage, high flood-hazard share, distance to coast and coastal indicator, population density, toilet count, population total, derived population per area, and toilets per 10,000 people.

The source weather is daily and barangay-based. For these weather fields, the pipeline takes city-level means across available barangay observations. For many static characteristics it calculates an area-weighted city average; count variables such as total population and toilet count are summed. The scripts use the most recent record for each barangay in the static input files.

The target columns describe future outcomes relative to the anchor row: `target_h1` is one week later, `target_h2` two weeks later, and `target_h3` three weeks later. Horizon 0 instead predicts `weekly_cases` for the anchor week itself. Thus, horizon 0 is a same-week estimate, not a forecast made one or more weeks in advance.

## 6. Models and evaluation design

### Feature sets

The experiments compare nested groups of predictors:

1. **Case only:** previous one- and two-week case counts plus case velocity.
2. **Case + weather:** the case features plus the six lagged weather/environment fields.
3. **Full:** case and weather features plus the ten built-environment/city-characteristic fields.

The ablation comparison asks whether adding weather, and then city characteristics, changes predictive performance.

### Models

The saved experiment results cover 16 model configurations for each of four horizons:

- **All-zero baseline:** always estimates zero cases.
- **Persistence baseline:** repeats the previous week's case count.
- **Poisson regression:** a scaled linear regression model for count-valued outcomes.
- **XGBoost:** trains with either a Poisson count objective or squared-error objective on each of the three feature sets.
- **CatBoost:** trains with Poisson or RMSE loss on each of the three feature sets.
- **Random forest:** a tree-ensemble model trained on the full feature set.

The common tree-model settings use 300 boosting iterations/trees, depth 4, a learning rate of 0.05 for XGBoost and CatBoost, and random seed 42. The random forest uses 300 trees and a minimum leaf size of 2.

### Time split and measures

The evaluation is chronological rather than a random row split. Training targets must be dated on or before **December 31, 2018**. Test anchor dates start on **February 4, 2019**, leaving an approximately five-week gap between the training cutoff and the start of the test anchors. For horizons 1–3, training eligibility is checked using the future target date, not only the anchor date.

Cities are also grouped using their share of zero-case weeks in the training period: **Tier 1** has fewer than 80% zero-case weeks; **Tier 2** has at least 80%.

The main reported metrics are:

- **MAE (mean absolute error):** average absolute difference in cases; easier to interpret in case-count units.
- **RMSE (root mean squared error):** similar to MAE but penalizes large misses more strongly.
- **Poisson deviance:** a count-oriented measure of how well estimates align with observed counts; lower is better.
- **MAE/RMSE on non-zero weeks:** errors calculated only for weeks when cases were recorded.

Predictions are clipped at zero before evaluation because negative case counts are not meaningful. Poisson deviance also requires strictly positive estimates, so its calculation clips predictions to a small positive floor.

## 7. What the saved results say

The current `summary_tables.txt` and bootstrap output show that results depend on the metric and horizon; there is no single model that is best in every comparison.

- For **all cities at horizon 0**, the full XGBoost model has MAE 0.540 cases, compared with 0.549 for the all-zero baseline—a small difference.
- At **horizon 2**, the full XGBoost model has Poisson deviance 1.356, compared with 1.426 for the case-and-weather XGBoost model. The block-bootstrap estimated deviance reduction from adding the static city features is about 4.95%, with a 95% interval of about 0.11% to 11.38%.
- Also at **horizon 2**, adding weather to the case-only XGBoost model reduces Poisson deviance by about 13.98%; the bootstrap 95% interval is about 5.46% to 18.97%.
- However, at horizon 2 the all-zero baseline's MAE is 0.550, while the full XGBoost MAE is 0.586. In the saved bootstrap comparison, the all-zero baseline has a lower MAE than full XGBoost for that horizon. A strong zero-case baseline can perform well on average because many city-weeks have no reported cases.
- Errors and case frequency differ substantially by city. For example, the final table's zero-case share ranges from about 28% in Iloilo City to about 96% in Mandaue City. It is therefore useful to show city-level and non-zero-week results as well as overall averages.

The careful conclusion is that recent weather contains useful predictive information for some evaluation measures, and static city features can add a smaller improvement in some comparisons. The results do **not** establish a consistent improvement in every metric or horizon, and they do not establish a causal relationship between weather or city characteristics and leptospirosis.

## 8. Important limitations to explain

1. **Selected study cities:** the modelling table covers 12 cities, not all Philippine cities or the whole country. Results should not be generalized beyond the study sample without further evaluation.
2. **Historical rather than live predictions:** the app replays saved estimates. The data end in December 2022 and no future case data are used to create a current forecast.
3. **Aggregated observations:** the model predicts city-level reported case counts. It does not estimate individual risk or barangay-level risk.
4. **Many zero weeks and uneven case counts:** an all-zero model is a meaningful benchmark, and averages can hide performance on weeks with cases or in cities with higher counts.
5. **Association is not causation:** predictive importance or SHAP values describe patterns the model used; they do not show that changing rainfall, sanitation, or another feature would change disease incidence.
6. **Reported counts are not necessarily all infections:** the target is the weekly count in the supplied reporting data, not a direct measure of every infection in the population.
7. **Not decision support:** the application labels itself a research prototype and says it is not an official Department of Health product or a tool for public-health decision-making.

## 9. Running the project

From the project root, the intended sequence is:

```powershell
python notebooks\build_training_data.py
python notebooks\build_horizon_targets.py
python notebooks\build_city_daily_weather.py
python notebooks\run_experiments.py
python notebooks\bootstrap_ci.py
python notebooks\shap_analysis.py --horizon 2 --model xgboost
python notebooks\summarize_results.py
streamlit run app.py
```

Use `python notebooks\run_experiments.py --loco` to run the additional leave-one-city-out evaluation. The full modelling and analysis commands require the relevant scientific Python packages; the current `requirements.txt` does not list all of them.

To run the dashboard helper tests:

```powershell
python -m unittest discover -s tests
```

## 10. Questions a professor may ask

**Why use a chronological split?**  
Because the intended use is to estimate later outcomes from earlier information. A random split could mix later and earlier observations and give an overly optimistic test.

**Why compare with an all-zero model?**  
Many city-weeks have zero reported cases. The baseline checks whether a more complex model improves on the simple strategy of always predicting zero.

**What does a two-week horizon mean?**  
The model uses features attached to an anchor week and predicts the reported case count two weeks after that anchor. Weather features are lagged so they describe prior conditions.

**Does a high SHAP value mean a cause?**  
No. It means that a feature contributed strongly to the fitted model's prediction in the analyzed data. It is not causal evidence.

**Can the dashboard forecast next week's cases?**  
Not currently. It shows saved predictions for historical held-out weeks. A future forecast would require updated case data, retraining/evaluation, and appropriate recent model inputs.
