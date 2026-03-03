"""Build team-season aggregate features from Massey ordinals."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io_utils import processed_path, raw_path, read_csv_checked

MASSEY_REQUIRED_COLUMNS = ["Season", "RankingDayNum", "SystemName", "TeamID", "OrdinalRank"]


def build_massey_features(massey_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate latest-per-system Massey rankings into team-season summary features."""
    sort_cols = ["Season", "TeamID", "SystemName", "RankingDayNum"]
    latest_per_system = massey_df.sort_values(sort_cols).drop_duplicates(
        subset=["Season", "TeamID", "SystemName"], keep="last"
    )

    agg = (
        latest_per_system.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            massey_rank_mean=("OrdinalRank", "mean"),
            massey_rank_median=("OrdinalRank", "median"),
            massey_system_count=("SystemName", "nunique"),
        )
        .sort_values(["Season", "TeamID"])
        .reset_index(drop=True)
    )
    return agg


def build_and_save_massey_features(data_dir: Path) -> pd.DataFrame:
    """Load Massey ordinals, build features, and save to processed CSV."""
    massey_path = raw_path(data_dir, "MMasseyOrdinals.csv")
    massey_df = read_csv_checked(massey_path, MASSEY_REQUIRED_COLUMNS, name="MMasseyOrdinals")
    out_df = build_massey_features(massey_df)
    out_df.to_csv(processed_path(data_dir, "team_season_massey_features.csv"), index=False)
    return out_df

