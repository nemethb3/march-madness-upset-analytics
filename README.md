# march-madness-upset-analytics

Resume-quality, reproducible NCAA March Madness upset prediction pipeline using Kaggle March Machine Learning Mania men's compact CSVs.

## Overview
This project builds a modeling dataset where each row is a men's NCAA tournament game matchup, then trains baseline classifiers to predict whether the lower-ID team (`Team1ID`) wins.

It uses only local Kaggle files already downloaded to `data/raw`:
- `MRegularSeasonCompactResults.csv`
- `MNCAATourneyCompactResults.csv`
- `MNCAATourneySeeds.csv`
- `MTeams.csv`

No web scraping or external APIs are used.

## Engineered Features
Per `(Season, TeamID)` regular-season features:
- `games_played`, `wins`, `losses`, `win_pct`
- `points_for`, `points_against`
- `avg_points_for`, `avg_points_against`
- `avg_margin`
- `strength_proxy` (equal to `avg_margin`)

Tournament matchup features:
- Deterministic team ordering: `Team1ID < Team2ID`
- Seeds: `Team1Seed`, `Team2Seed`, `SeedDiff`
- Team stats for both teams
- Difference features (`Team1 - Team2`) for key performance stats
- Label: `Team1Win`
- Team names for convenience: `Team1Name`, `Team2Name`

## Models
- Logistic Regression (with `StandardScaler`)
- Random Forest Classifier

Time-based split:
- Train seasons: `<= max_season - 1`
- Test season: `max_season`

Evaluation:
- Accuracy
- ROC AUC (when possible)
- Confusion matrix
- Top 10 logistic coefficients and top 10 RF feature importances

## Run
1. Create a Python 3.11+ environment.
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Execute the full pipeline:
```bash
python run_pipeline.py --data_dir data --output_dir outputs
```

## Output Files
Generated/overwritten on each run:
- `data/processed/team_season_features.csv`
- `data/processed/tourney_seeds_clean.csv`
- `data/processed/tourney_matchups_model.csv`
- `outputs/models/logistic_regression_pipeline.joblib`
- `outputs/models/random_forest.joblib`
- `outputs/reports/model_report.md`

Optional directory for plots:
- `outputs/figures/`

## Streamlit Dashboard
This repo includes a multi-page Streamlit dashboard at `app/streamlit_app.py` with:
- Upset alerts
- Bracket builder
- Tournament simulations

### Run Locally
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

### Streamlit Community Cloud Deployment (Free)
1. Push this repo to GitHub.
2. In Streamlit Community Cloud, create a new app from the repo.
3. Set the main file path to: `app/streamlit_app.py`
4. Deploy.

The app supports:
- **Bundle Mode**: automatic loading from `data/app/{season}/`.
- **Demo fallback**: if selected season bundle is incomplete.

## Season Bundle Workflow (Maintainer)
To add a new season for the dashboard:
1. Download Kaggle data for that season.
2. Run the feature pipeline locally so `team_season_features.csv` is generated.
3. Create bundle folder:
   - `data/app/{season}/seeds.csv`
   - `data/app/{season}/slots.csv`
   - `data/app/{season}/team_features.csv`
4. Ensure `data/app/team_id_map.csv` includes `TeamID,TeamName` for teams in your bundles.
5. Push to GitHub.

Streamlit Community Cloud redeploys automatically after push.

### Synthetic Demo Season
- `data/app/2026/` is synthetic demo data for UI testing.
- It is not real tournament data and should not be used for real forecasting decisions.

Regenerate synthetic 2026 demo bundle:
```bash
python tools/generate_demo_season.py --source_season 2025 --target_season 2026 --seed 42
```
