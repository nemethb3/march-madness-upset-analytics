"""Train and evaluate baseline models for March Madness upset prediction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config
from src.evaluate import EvaluationResult, evaluate_classifier
from src.io_utils import models_path, processed_path, reports_path

LABEL_COLUMN = "Team1Win"


def _get_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Perform season-based split: train <= max_season - 1, test == max_season."""
    if df.empty:
        raise ValueError("Matchup dataframe is empty.")

    max_season = int(df["Season"].max())
    train_df = df[df["Season"] <= max_season - 1].copy()
    test_df = df[df["Season"] == max_season].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Time-based split failed: train/test split is empty.")
    return train_df, test_df, max_season


def _get_train_cal_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
    """Perform time-aware split for calibration."""
    if df.empty:
        raise ValueError("Matchup dataframe is empty.")
    max_season = int(df["Season"].max())
    train_df = df[df["Season"] <= max_season - 2].copy()
    cal_df = df[df["Season"] == max_season - 1].copy()
    test_df = df[df["Season"] == max_season].copy()
    if train_df.empty or cal_df.empty or test_df.empty:
        raise ValueError("Calibration split failed: one or more splits are empty.")
    return train_df, cal_df, test_df, max_season


def _get_model_features(df: pd.DataFrame) -> list[str]:
    """Select numeric model features while excluding identifiers, labels, and names."""
    excluded_exact = {"Season", "DayNum", "Team1ID", "Team2ID", LABEL_COLUMN}
    candidate_cols = [c for c in df.columns if c not in excluded_exact]
    numeric_cols = [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]
    feature_cols = [c for c in numeric_cols if "Name" not in c]
    if not feature_cols:
        raise ValueError("No numeric feature columns available for training.")
    return feature_cols


def _format_cm(cm: np.ndarray) -> str:
    """Format confusion matrix for markdown report."""
    return (
        "| Actual \\ Pred | 0 | 1 |\n"
        "|---|---:|---:|\n"
        f"| 0 | {int(cm[0, 0])} | {int(cm[0, 1])} |\n"
        f"| 1 | {int(cm[1, 0])} | {int(cm[1, 1])} |"
    )


def _top_lr_coefficients(model: Pipeline, feature_names: list[str], top_k: int = 10) -> pd.DataFrame:
    """Return top logistic coefficients by absolute magnitude."""
    lr = model.named_steps["logreg"]
    coefs = lr.coef_.ravel()
    coef_df = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    return coef_df.reindex(coef_df["coefficient"].abs().sort_values(ascending=False).index).head(top_k)


def _top_rf_importances(model: RandomForestClassifier, feature_names: list[str], top_k: int = 10) -> pd.DataFrame:
    """Return top random forest feature importances."""
    imp_df = pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
    return imp_df.sort_values("importance", ascending=False).head(top_k)


def _format_metrics(result: EvaluationResult) -> str:
    """Format evaluation metrics block for markdown."""
    roc_text = "N/A" if result.roc_auc is None else f"{result.roc_auc:.4f}"
    return (
        f"- Accuracy: {result.accuracy:.4f}\n"
        f"- ROC AUC: {roc_text}\n"
        f"- Confusion Matrix:\n\n{_format_cm(result.confusion_matrix)}\n"
    )


def _format_top_table(df: pd.DataFrame, value_col: str) -> str:
    """Render a simple markdown table for top features."""
    lines = [
        f"| feature | {value_col} |",
        "|---|---:|",
    ]
    for _, row in df.iterrows():
        lines.append(f"| {row['feature']} | {row[value_col]:.6f} |")
    return "\n".join(lines)


def train_and_evaluate_models(
    data_dir: Path,
    output_dir: Path,
    calibrate: bool = False,
    calibration_method: str = "sigmoid",
) -> dict[str, Any]:
    """Train baseline models, save artifacts, and write markdown report."""
    matchup_path = processed_path(data_dir, config.TOURNEY_MATCHUPS_FILENAME)
    if not matchup_path.exists():
        raise FileNotFoundError(f"Processed matchup file not found: {matchup_path}")

    df = pd.read_csv(matchup_path)
    model_features = _get_model_features(df)
    required_cols = {"Season", LABEL_COLUMN, *model_features}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"Matchup data missing required columns: {missing}")

    if calibrate:
        train_df, cal_df, test_df, test_season = _get_train_cal_test(df)
        x_train = train_df[model_features].copy()
        y_train = train_df[LABEL_COLUMN].to_numpy()
        x_cal = cal_df[model_features].copy()
        y_cal = cal_df[LABEL_COLUMN].to_numpy()
    else:
        train_df, test_df, test_season = _get_train_test(df)
        x_train = train_df[model_features].copy()
        y_train = train_df[LABEL_COLUMN].to_numpy()
        x_cal = None
        y_cal = None
    x_test = test_df[model_features].copy()
    y_test = test_df[LABEL_COLUMN].to_numpy()

    lr_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )
    lr_pipeline.fit(x_train, y_train)

    if calibrate:
        lr_calibrated = CalibratedClassifierCV(FrozenEstimator(lr_pipeline), method=calibration_method)
        lr_calibrated.fit(x_cal, y_cal)
        lr_primary = lr_calibrated
    else:
        lr_primary = lr_pipeline

    rf_model = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(x_train, y_train)

    lr_eval = evaluate_classifier(lr_primary, x_test, y_test)
    rf_eval = evaluate_classifier(rf_model, x_test, y_test)

    top_lr = _top_lr_coefficients(lr_pipeline, model_features, top_k=10)
    top_rf = _top_rf_importances(rf_model, model_features, top_k=10)

    joblib.dump(lr_pipeline, models_path(output_dir, config.MODEL_LR_FILENAME))
    if calibrate:
        joblib.dump(lr_primary, models_path(output_dir, config.MODEL_LR_CALIBRATED_FILENAME))
    joblib.dump(rf_model, models_path(output_dir, config.MODEL_RF_FILENAME))

    lr_prob = lr_primary.predict_proba(x_test)[:, 1]
    rf_prob = rf_model.predict_proba(x_test)[:, 1]
    lr_brier = float(brier_score_loss(y_test, lr_prob))
    rf_brier = float(brier_score_loss(y_test, rf_prob))

    report_lines = [
        "# March Madness Upset Analytics - Model Report",
        "",
        f"- Train seasons: <= {test_season - (2 if calibrate else 1)}",
        *([f"- Calibration season: {test_season - 1}"] if calibrate else []),
        f"- Test season: {test_season}",
        f"- Training rows: {len(train_df)}",
        *([f"- Calibration rows: {len(cal_df)}"] if calibrate else []),
        f"- Test rows: {len(test_df)}",
        f"- Feature count: {len(model_features)}",
        "",
        "## Logistic Regression (Scaled)",
        *([f"- Calibration: enabled (`{calibration_method}` via `CalibratedClassifierCV`)\n"] if calibrate else ["- Calibration: disabled\n"]),
        _format_metrics(lr_eval),
        f"- Brier Score: {lr_brier:.4f}\n",
        "### Top 10 Coefficients (absolute magnitude)",
        _format_top_table(top_lr, "coefficient"),
        "",
        "## Random Forest",
        _format_metrics(rf_eval),
        f"- Brier Score: {rf_brier:.4f}\n",
        "### Top 10 Feature Importances",
        _format_top_table(top_rf, "importance"),
        "",
    ]
    reports_path(output_dir, config.REPORT_FILENAME).write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "test_season": test_season,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "lr_accuracy": lr_eval.accuracy,
        "lr_roc_auc": lr_eval.roc_auc,
        "lr_brier": lr_brier,
        "rf_accuracy": rf_eval.accuracy,
        "rf_roc_auc": rf_eval.roc_auc,
        "rf_brier": rf_brier,
        "feature_count": len(model_features),
        "calibrate": calibrate,
    }
