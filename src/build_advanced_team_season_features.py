"""Build advanced team-season features from detailed regular-season results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.io_utils import processed_path, raw_path, read_csv_checked

DETAILED_REQUIRED_COLUMNS = [
    "Season",
    "WTeamID",
    "WScore",
    "LTeamID",
    "LScore",
    "WFGM",
    "WFGA",
    "WFGM3",
    "WFGA3",
    "WFTM",
    "WFTA",
    "WOR",
    "WDR",
    "WAst",
    "WTO",
    "LFGM",
    "LFGA",
    "LFGM3",
    "LFGA3",
    "LFTM",
    "LFTA",
    "LOR",
    "LDR",
    "LAst",
    "LTO",
]


def _safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safely divide two series and return NaN when denominator is zero."""
    den_clean = den.replace(0, np.nan)
    return num / den_clean


def build_advanced_team_season_features(detailed_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build advanced one-row-per-(Season, TeamID) features from detailed results.

    Aggregates both winner and loser perspectives, then computes season-level rates.
    """
    winners = pd.DataFrame(
        {
            "Season": detailed_df["Season"],
            "TeamID": detailed_df["WTeamID"],
            "points_for": detailed_df["WScore"],
            "points_against": detailed_df["LScore"],
            "FGM": detailed_df["WFGM"],
            "FGA": detailed_df["WFGA"],
            "FGM3": detailed_df["WFGM3"],
            "FGA3": detailed_df["WFGA3"],
            "FTM": detailed_df["WFTM"],
            "FTA": detailed_df["WFTA"],
            "OR": detailed_df["WOR"],
            "DR": detailed_df["WDR"],
            "Ast": detailed_df["WAst"],
            "TO": detailed_df["WTO"],
            "oppFGA": detailed_df["LFGA"],
            "oppFTA": detailed_df["LFTA"],
            "oppOR": detailed_df["LOR"],
            "oppDR": detailed_df["LDR"],
            "oppTO": detailed_df["LTO"],
        }
    )
    losers = pd.DataFrame(
        {
            "Season": detailed_df["Season"],
            "TeamID": detailed_df["LTeamID"],
            "points_for": detailed_df["LScore"],
            "points_against": detailed_df["WScore"],
            "FGM": detailed_df["LFGM"],
            "FGA": detailed_df["LFGA"],
            "FGM3": detailed_df["LFGM3"],
            "FGA3": detailed_df["LFGA3"],
            "FTM": detailed_df["LFTM"],
            "FTA": detailed_df["LFTA"],
            "OR": detailed_df["LOR"],
            "DR": detailed_df["LDR"],
            "Ast": detailed_df["LAst"],
            "TO": detailed_df["LTO"],
            "oppFGA": detailed_df["WFGA"],
            "oppFTA": detailed_df["WFTA"],
            "oppOR": detailed_df["WOR"],
            "oppDR": detailed_df["WDR"],
            "oppTO": detailed_df["WTO"],
        }
    )

    long_df = pd.concat([winners, losers], ignore_index=True)
    long_df["games_played"] = 1
    long_df["poss_game"] = 0.5 * (
        (long_df["FGA"] + 0.44 * long_df["FTA"] - long_df["OR"] + long_df["TO"])
        + (long_df["oppFGA"] + 0.44 * long_df["oppFTA"] - long_df["oppOR"] + long_df["oppTO"])
    )

    sum_cols = [
        "games_played",
        "points_for",
        "points_against",
        "FGM",
        "FGA",
        "FGM3",
        "FGA3",
        "FTM",
        "FTA",
        "OR",
        "DR",
        "Ast",
        "TO",
        "oppOR",
        "oppDR",
        "poss_game",
    ]
    grouped = long_df.groupby(["Season", "TeamID"], as_index=False)[sum_cols].sum()
    grouped = grouped.rename(columns={"poss_game": "poss"})

    grouped["fg_pct"] = _safe_divide(grouped["FGM"], grouped["FGA"])
    grouped["fg3_pct"] = _safe_divide(grouped["FGM3"], grouped["FGA3"])
    grouped["ft_pct"] = _safe_divide(grouped["FTM"], grouped["FTA"])
    grouped["efg_pct"] = _safe_divide(grouped["FGM"] + 0.5 * grouped["FGM3"], grouped["FGA"])
    grouped["ts_pct"] = _safe_divide(grouped["points_for"], 2 * (grouped["FGA"] + 0.44 * grouped["FTA"]))

    grouped["off_rtg"] = 100 * _safe_divide(grouped["points_for"], grouped["poss"])
    grouped["def_rtg"] = 100 * _safe_divide(grouped["points_against"], grouped["poss"])
    grouped["net_rtg"] = grouped["off_rtg"] - grouped["def_rtg"]
    grouped["pace"] = _safe_divide(grouped["poss"], grouped["games_played"])

    grouped["tov_rate"] = _safe_divide(grouped["TO"], grouped["FGA"] + 0.44 * grouped["FTA"] + grouped["TO"])
    grouped["orb_pct"] = _safe_divide(grouped["OR"], grouped["OR"] + grouped["oppDR"])
    grouped["drb_pct"] = _safe_divide(grouped["DR"], grouped["DR"] + grouped["oppOR"])
    grouped["ast_rate"] = _safe_divide(grouped["Ast"], grouped["FGM"])

    grouped["fga_pg"] = _safe_divide(grouped["FGA"], grouped["games_played"])
    grouped["fta_pg"] = _safe_divide(grouped["FTA"], grouped["games_played"])
    grouped["fg3a_pg"] = _safe_divide(grouped["FGA3"], grouped["games_played"])
    grouped["or_pg"] = _safe_divide(grouped["OR"], grouped["games_played"])
    grouped["dr_pg"] = _safe_divide(grouped["DR"], grouped["games_played"])
    grouped["to_pg"] = _safe_divide(grouped["TO"], grouped["games_played"])
    grouped["ast_pg"] = _safe_divide(grouped["Ast"], grouped["games_played"])

    out_cols = [
        "Season",
        "TeamID",
        "fg_pct",
        "fg3_pct",
        "ft_pct",
        "efg_pct",
        "ts_pct",
        "poss",
        "off_rtg",
        "def_rtg",
        "net_rtg",
        "pace",
        "tov_rate",
        "orb_pct",
        "drb_pct",
        "ast_rate",
        "fga_pg",
        "fta_pg",
        "fg3a_pg",
        "or_pg",
        "dr_pg",
        "to_pg",
        "ast_pg",
    ]
    return grouped[out_cols].sort_values(["Season", "TeamID"]).reset_index(drop=True)


def build_and_save_advanced_team_season_features(data_dir: Path) -> pd.DataFrame:
    """Load detailed regular-season data, build advanced features, and save to processed CSV."""
    detailed_path = raw_path(data_dir, "MRegularSeasonDetailedResults.csv")
    detailed_df = read_csv_checked(detailed_path, DETAILED_REQUIRED_COLUMNS, name="MRegularSeasonDetailedResults")
    adv_df = build_advanced_team_season_features(detailed_df)
    adv_df.to_csv(processed_path(data_dir, "team_season_advanced_features.csv"), index=False)
    return adv_df

