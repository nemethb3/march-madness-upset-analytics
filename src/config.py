"""Project configuration constants."""

from __future__ import annotations

from pathlib import Path

RAW_DIRNAME = "raw"
PROCESSED_DIRNAME = "processed"
MODELS_DIRNAME = "models"
REPORTS_DIRNAME = "reports"
FIGURES_DIRNAME = "figures"

TEAM_SEASON_FEATURES_FILENAME = "team_season_features.csv"
TOURNEY_SEEDS_CLEAN_FILENAME = "tourney_seeds_clean.csv"
TOURNEY_MATCHUPS_FILENAME = "tourney_matchups_model.csv"

MODEL_LR_FILENAME = "logistic_regression_pipeline.joblib"
MODEL_LR_CALIBRATED_FILENAME = "logistic_regression_calibrated.joblib"
MODEL_RF_FILENAME = "random_forest.joblib"
REPORT_FILENAME = "model_report.md"


def ensure_directories(data_dir: Path, output_dir: Path) -> None:
    """Create processed and output directories if they do not exist."""
    (data_dir / PROCESSED_DIRNAME).mkdir(parents=True, exist_ok=True)
    (output_dir / MODELS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (output_dir / REPORTS_DIRNAME).mkdir(parents=True, exist_ok=True)
    (output_dir / FIGURES_DIRNAME).mkdir(parents=True, exist_ok=True)
