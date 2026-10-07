# Leptospirosis Case Estimation

This project combines weekly leptospirosis case reports with weather, geographic, population, building, flood-hazard, and sanitation data to estimate weekly case counts for 12 selected Philippine cities. It builds a city-week modelling dataset, compares statistical and machine-learning models using a chronological holdout period, and provides a Streamlit app for exploring saved historical predictions.

**This is a research prototype, not a live forecasting service or an official public-health decision tool.** The dashboard replays historical test predictions; it does not currently generate predictions for future weeks.

For a detailed explanation of the code, datasets, evaluation, findings, and limitations, see [PROJECT_EXPLANATION.md](PROJECT_EXPLANATION.md).

## What we did

1. Joined weekly disease reports to city identifiers and retained leptospirosis case counts.
2. Aggregated daily weather measurements from barangays to cities and created lagged rainfall, humidity, temperature, and vegetation-index features.
3. Combined barangay-level building, flood-hazard, coastal, population, and sanitation information into city-level features.
4. Created targets for the same week and for one, two, and three weeks later.
5. Compared zero and persistence baselines with Poisson regression, XGBoost, CatBoost, and random forest models.
6. Evaluated predictions on later historical weeks, summarized model errors, calculated block-bootstrap confidence intervals, and produced SHAP feature explanations.
7. Built a Streamlit application that lets users inspect a saved estimate for a city and historical week, compare it with baselines, and reveal the recorded outcome.

The modelling table contains 9,238 city-week rows from January 2008 through December 2022. The evaluation is time-ordered: training targets end by December 31, 2018, and test anchor weeks begin February 4, 2019. Results vary by metric and forecast horizon; model performance is not consistently better than the simple all-zero baseline on every measure.

## Repository contents

- `app.py` — Streamlit historical-prediction explorer.
- `notebooks/` — data-building, modelling, evaluation, diagnostic, and analysis scripts.
- `data/` — raw input datasets and constructed modelling/weather tables.
- `results/` — saved predictions, metrics, confidence intervals, and SHAP outputs.
- `tests/` — unit tests for dashboard helper functions.
- `PROJECT_EXPLANATION.md` — detailed project and dataset guide.

The climate files `data/climate_atmosphere.csv` and `data/climate_land.csv` are stored with Git LFS because each is larger than GitHub's regular-file limit. Install Git LFS before cloning so those datasets are downloaded correctly.

## Clone on another computer

You need Git, Git LFS, and Python installed. If the GitHub repository is private, your GitHub account must also have access.

### Windows (PowerShell)

```powershell
git lfs install
git clone https://github.com/jairo3224/Leptospiros_Case_Estimation.git
cd Leptospiros_Case_Estimation
git lfs pull

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If PowerShell blocks virtual-environment activation, either follow your organization's Python/PowerShell policy or run the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS or Linux

```bash
git lfs install
git clone https://github.com/jairo3224/Leptospiros_Case_Estimation.git
cd Leptospiros_Case_Estimation
git lfs pull

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app reads its saved estimates and metrics from `results/` and displays historical data only.

## Run the analysis

The commands below rebuild the derived datasets, run the main model comparisons, calculate confidence intervals, and create the result summary. Run them from the repository root after installing the extra analysis packages listed below.

```bash
python notebooks/build_training_data.py
python notebooks/build_horizon_targets.py
python notebooks/build_city_daily_weather.py
python notebooks/run_experiments.py
python notebooks/bootstrap_ci.py
python notebooks/shap_analysis.py --horizon 2 --model xgboost
python notebooks/summarize_results.py
```

Optional leave-one-city-out evaluation:

```bash
python notebooks/run_experiments.py --loco
```

Run the dashboard tests with:

```bash
python -m unittest discover -s tests
```

## Dependencies

`requirements.txt` lists the app's core dependencies: pandas and Streamlit. To run the modelling and analysis scripts, install their additional packages in the activated environment:

```bash
python -m pip install numpy scikit-learn xgboost catboost shap matplotlib
```

The Git LFS climate datasets are large, so cloning them requires sufficient Git LFS storage/bandwidth quota and may take time on a slower connection.

## Interpretation and limitations

- Predictions are city-level reported case counts, not individual or barangay-level risk estimates.
- Many city-weeks have zero reported cases, so the all-zero baseline is an important comparison.
- SHAP values describe how features contribute to a fitted model; they do not establish causation.
- The study covers selected cities and historical data only. It should not be generalized to other locations or used for public-health decisions without further validation.
