"""Build team-season features from regular season compact results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.io_utils import processed_path, raw_path, read_csv_checked

REGULAR_REQUIRED_COLUMNS = [
    "Season",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
]


def build_team_season_features(regular_season_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build one row per (Season, TeamID) from compact regular-season results.

    Uses both winner and loser perspectives for correct per-team aggregation.
    """
    winners = regular_season_df[["Season", "WTeamID", "WScore", "LScore"]].copy()
    winners.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
    winners["Win"] = 1

    losers = regular_season_df[["Season", "LTeamID", "LScore", "WScore"]].copy()
    losers.columns = ["Season", "TeamID", "PointsFor", "PointsAgainst"]
    losers["Win"] = 0

    long_df = pd.concat([winners, losers], ignore_index=True)
    long_df["GameCount"] = 1

    grouped = (
        long_df.groupby(["Season", "TeamID"], as_index=False)
        .agg(
            games_played=("GameCount", "sum"),
            wins=("Win", "sum"),
            points_for=("PointsFor", "sum"),
            points_against=("PointsAgainst", "sum"),
        )
        .sort_values(["Season", "TeamID"])
    )

    grouped["losses"] = grouped["games_played"] - grouped["wins"]
    grouped["win_pct"] = grouped["wins"] / grouped["games_played"]
    grouped["avg_points_for"] = grouped["points_for"] / grouped["games_played"]
    grouped["avg_points_against"] = grouped["points_against"] / grouped["games_played"]
    grouped["avg_margin"] = grouped["avg_points_for"] - grouped["avg_points_against"]
    grouped["strength_proxy"] = grouped["avg_margin"]

    ordered_cols = [
        "Season",
        "TeamID",
        "games_played",
        "wins",
        "losses",
        "win_pct",
        "points_for",
        "points_against",
        "avg_points_for",
        "avg_points_against",
        "avg_margin",
        "strength_proxy",
    ]
    return grouped[ordered_cols]


def build_and_save_team_season_features(data_dir: Path) -> pd.DataFrame:
    """Load raw regular-season results, build features, and save to processed CSV."""
    regular_path = raw_path(data_dir, "MRegularSeasonCompactResults.csv")
    regular_df = read_csv_checked(regular_path, REGULAR_REQUIRED_COLUMNS, name="MRegularSeasonCompactResults")

    features_df = build_team_season_features(regular_df)
    out_path = processed_path(data_dir, config.TEAM_SEASON_FEATURES_FILENAME)
    features_df.to_csv(out_path, index=False)
    return features_df


def merge_team_season_feature_tables(base_df: pd.DataFrame, extra_feature_tables: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge compact-derived features with additional team-season feature tables."""
    out = base_df.copy()
    for df in extra_feature_tables:
        if df is None or df.empty:
            continue
        expected = {"Season", "TeamID"}
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Feature table missing merge keys {missing}: {list(df.columns)}")
        extra_cols = [c for c in df.columns if c not in {"Season", "TeamID"}]
        dedup = df[["Season", "TeamID", *extra_cols]].drop_duplicates(subset=["Season", "TeamID"])
        out = out.merge(dedup, on=["Season", "TeamID"], how="left")
    return out.sort_values(["Season", "TeamID"]).reset_index(drop=True)


def add_giant_killer_features(team_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add fg3a_rate and normalized GiantKillerScore to team-season features.

    Score uses: fg3a_rate, orb_pct, 1/def_rtg, 1/pace and is min-max normalized to [0, 1].
    """
    out = team_features_df.copy()

    if "fg3a_rate" not in out.columns:
        if {"fg3a_pg", "fga_pg"}.issubset(out.columns):
            denom = out["fga_pg"].replace(0, np.nan)
            out["fg3a_rate"] = out["fg3a_pg"] / denom
        elif {"TeamFGA3", "TeamFGA"}.issubset(out.columns):
            denom = out["TeamFGA"].replace(0, np.nan)
            out["fg3a_rate"] = out["TeamFGA3"] / denom
        else:
            out["fg3a_rate"] = np.nan

    needed = ["fg3a_rate", "orb_pct", "def_rtg", "pace"]
    for c in needed:
        if c not in out.columns:
            out[c] = np.nan

    inv_def = 1.0 / out["def_rtg"].replace(0, np.nan)
    inv_pace = 1.0 / out["pace"].replace(0, np.nan)
    raw = 0.35 * out["fg3a_rate"] + 0.30 * out["orb_pct"] + 0.20 * inv_def + 0.15 * inv_pace

    valid = raw.dropna()
    if valid.empty:
        out["GiantKillerScore"] = np.nan
        return out

    rmin = float(valid.min())
    rmax = float(valid.max())
    if np.isclose(rmin, rmax):
        out["GiantKillerScore"] = 0.5
    else:
        out["GiantKillerScore"] = (raw - rmin) / (rmax - rmin)
    out["GiantKillerScore"] = out["GiantKillerScore"].clip(0, 1)
    return out
