"""Build cleaned tournament seeds and tournament matchup modeling dataset."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src import config
from src.io_utils import processed_path, raw_path, read_csv_checked

LOGGER = logging.getLogger(__name__)

SEEDS_REQUIRED_COLUMNS = ["Season", "Seed", "TeamID"]
TOURNEY_REQUIRED_COLUMNS = ["Season", "DayNum", "WTeamID", "LTeamID"]
TEAMS_REQUIRED_COLUMNS = ["TeamID", "TeamName"]

TEAM_STATS_COLUMNS = [
    "win_pct",
    "avg_margin",
    "avg_points_for",
    "avg_points_against",
]


def parse_seed_number(seed_str: str) -> int:
    """Extract numeric portion from seed strings like 'W01' or 'W16a'."""
    digits = re.findall(r"\d+", str(seed_str))
    if not digits:
        raise ValueError(f"Seed value has no digits: {seed_str}")
    return int(digits[0])


def build_clean_seeds(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned seeds dataframe with SeedNum and Region."""
    out = seeds_df[["Season", "Seed", "TeamID"]].copy()
    out = out.rename(columns={"Seed": "SeedStr"})
    out["Region"] = out["SeedStr"].astype(str).str[0]
    out["SeedNum"] = out["SeedStr"].map(parse_seed_number)
    return out[["Season", "TeamID", "SeedStr", "SeedNum", "Region"]].sort_values(["Season", "TeamID"])


def build_tourney_matchups(
    tourney_df: pd.DataFrame,
    clean_seeds_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build matchup-level dataframe with deterministic team ordering and label."""
    data = tourney_df[["Season", "DayNum", "WTeamID", "LTeamID"]].copy()
    data["Team1ID"] = data[["WTeamID", "LTeamID"]].min(axis=1)
    data["Team2ID"] = data[["WTeamID", "LTeamID"]].max(axis=1)
    data["Team1Win"] = (data["WTeamID"] == data["Team1ID"]).astype(int)
    data = data[["Season", "DayNum", "Team1ID", "Team2ID", "Team1Win"]]

    pre_merge_rows = len(data)

    seeds_lookup = clean_seeds_df[["Season", "TeamID", "SeedNum"]].copy()
    t1_seeds = seeds_lookup.rename(columns={"TeamID": "Team1ID", "SeedNum": "Team1Seed"})
    t2_seeds = seeds_lookup.rename(columns={"TeamID": "Team2ID", "SeedNum": "Team2Seed"})

    data = data.merge(t1_seeds, on=["Season", "Team1ID"], how="left")
    data = data.merge(t2_seeds, on=["Season", "Team2ID"], how="left")

    feature_cols = [c for c in team_features_df.columns if c not in {"Season", "TeamID"}]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(team_features_df[c])]

    features_lookup = team_features_df[["Season", "TeamID", *feature_cols]].copy()
    t1_features = features_lookup.rename(columns={"TeamID": "Team1ID", **{c: f"Team1_{c}" for c in feature_cols}})
    t2_features = features_lookup.rename(columns={"TeamID": "Team2ID", **{c: f"Team2_{c}" for c in feature_cols}})

    data = data.merge(t1_features, on=["Season", "Team1ID"], how="left")
    data = data.merge(t2_features, on=["Season", "Team2ID"], how="left")

    missing_seed_mask = data[["Team1Seed", "Team2Seed"]].isna().any(axis=1)
    missing_feature_cols = [f"Team1_{c}" for c in numeric_feature_cols] + [f"Team2_{c}" for c in numeric_feature_cols]
    missing_feature_mask = data[missing_feature_cols].isna().any(axis=1)

    LOGGER.info("Tournament rows before drops: %d", pre_merge_rows)
    LOGGER.info("Rows with missing seeds: %d", int(missing_seed_mask.sum()))
    LOGGER.info("Rows with missing team features: %d", int(missing_feature_mask.sum()))

    keep_mask = ~(missing_seed_mask | missing_feature_mask)
    data = data.loc[keep_mask].copy()
    LOGGER.info("Rows after drops: %d", len(data))

    data["SeedDiff"] = data["Team1Seed"] - data["Team2Seed"]
    for col in numeric_feature_cols:
        data[f"Diff_{col}"] = data[f"Team1_{col}"] - data[f"Team2_{col}"]

    names = teams_df[["TeamID", "TeamName"]].copy()
    t1_names = names.rename(columns={"TeamID": "Team1ID", "TeamName": "Team1Name"})
    t2_names = names.rename(columns={"TeamID": "Team2ID", "TeamName": "Team2Name"})
    data = data.merge(t1_names, on="Team1ID", how="left").merge(t2_names, on="Team2ID", how="left")

    ordered_cols = [
        "Season",
        "DayNum",
        "Team1ID",
        "Team2ID",
        "Team1Name",
        "Team2Name",
        "Team1Seed",
        "Team2Seed",
        "SeedDiff",
    ]
    t1_cols = [f"Team1_{c}" for c in feature_cols]
    t2_cols = [f"Team2_{c}" for c in feature_cols]
    diff_cols = [f"Diff_{c}" for c in numeric_feature_cols]
    ordered_cols.extend(t1_cols)
    ordered_cols.extend(t2_cols)
    ordered_cols.extend(diff_cols)
    ordered_cols.append("Team1Win")

    return data[ordered_cols].sort_values(["Season", "DayNum", "Team1ID", "Team2ID"]).reset_index(drop=True)


def build_and_save_tourney_data(data_dir: Path, team_features_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build and save clean seeds and tournament matchup modeling dataset."""
    seeds_df = read_csv_checked(
        raw_path(data_dir, "MNCAATourneySeeds.csv"),
        SEEDS_REQUIRED_COLUMNS,
        name="MNCAATourneySeeds",
    )
    tourney_df = read_csv_checked(
        raw_path(data_dir, "MNCAATourneyCompactResults.csv"),
        TOURNEY_REQUIRED_COLUMNS,
        name="MNCAATourneyCompactResults",
    )
    teams_df = read_csv_checked(
        raw_path(data_dir, "MTeams.csv"),
        TEAMS_REQUIRED_COLUMNS,
        name="MTeams",
    )

    clean_seeds_df = build_clean_seeds(seeds_df)
    clean_seeds_df.to_csv(processed_path(data_dir, config.TOURNEY_SEEDS_CLEAN_FILENAME), index=False)

    matchups_df = build_tourney_matchups(
        tourney_df=tourney_df,
        clean_seeds_df=clean_seeds_df,
        team_features_df=team_features_df,
        teams_df=teams_df,
    )
    for seed_col in ["Team1Seed", "Team2Seed"]:
        matchups_df[seed_col] = matchups_df[seed_col].astype(int)
    matchups_df.to_csv(processed_path(data_dir, config.TOURNEY_MATCHUPS_FILENAME), index=False)

    return clean_seeds_df, matchups_df
