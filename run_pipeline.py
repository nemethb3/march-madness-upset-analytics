"""CLI entrypoint to run full March Madness upset analytics pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src import config
from src.build_advanced_team_season_features import build_and_save_advanced_team_season_features
from src.build_conference_features import build_and_save_conference_features
from src.build_massey_features import build_and_save_massey_features
from src.build_team_season_features import add_giant_killer_features
from src.build_team_season_features import build_and_save_team_season_features
from src.build_team_season_features import merge_team_season_feature_tables
from src.build_tourney_matchups import build_and_save_tourney_data
from src.io_utils import processed_path
from src.train_models import train_and_evaluate_models


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run March Madness upset analytics pipeline.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Base data directory (default: data)")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs"),
        help="Base output directory (default: outputs)",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable probability calibration for logistic regression using prior season as calibration holdout.",
    )
    parser.add_argument(
        "--calibration_method",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "isotonic"],
        help="Calibration method used when --calibrate is set.",
    )
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def main() -> None:
    """Execute feature engineering, matchup building, and model training."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    config.ensure_directories(args.data_dir, args.output_dir)

    compact_features_df = build_and_save_team_season_features(args.data_dir)
    advanced_features_df = build_and_save_advanced_team_season_features(args.data_dir)
    massey_features_df = build_and_save_massey_features(args.data_dir)
    conference_features_df = build_and_save_conference_features(args.data_dir)

    team_features_df = merge_team_season_feature_tables(
        compact_features_df,
        [advanced_features_df, massey_features_df, conference_features_df],
    )
    team_features_df = add_giant_killer_features(team_features_df)
    team_features_df.to_csv(processed_path(args.data_dir, config.TEAM_SEASON_FEATURES_FILENAME), index=False)

    _, _ = build_and_save_tourney_data(args.data_dir, team_features_df=team_features_df)
    summary = train_and_evaluate_models(
        args.data_dir,
        args.output_dir,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
    )

    logging.info("Pipeline complete.")
    logging.info("Training rows: %d | Test rows: %d", summary["train_rows"], summary["test_rows"])
    logging.info(
        "LR accuracy: %.4f | LR ROC AUC: %s",
        summary["lr_accuracy"],
        "N/A" if summary["lr_roc_auc"] is None else f"{summary['lr_roc_auc']:.4f}",
    )
    logging.info(
        "RF accuracy: %.4f | RF ROC AUC: %s",
        summary["rf_accuracy"],
        "N/A" if summary["rf_roc_auc"] is None else f"{summary['rf_roc_auc']:.4f}",
    )


if __name__ == "__main__":
    main()
