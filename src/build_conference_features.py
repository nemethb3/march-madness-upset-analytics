"""Build team-season conference metadata features."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.io_utils import processed_path, raw_path, read_csv_checked

TEAM_CONF_REQUIRED_COLUMNS = ["Season", "TeamID", "ConfAbbrev"]

# We expect `Conferences.csv` to expose conference code/name style fields.
# Join key is inferred from common candidates and defaults to ConfAbbrev.
CONF_JOIN_KEY_CANDIDATES = ["ConfAbbrev", "Conf", "Abbrev", "ConferenceAbbrev"]
CONF_NAME_CANDIDATES = ["Description", "ConfDescription", "ConfName", "ConferenceName", "Name"]

POWER_CONFS = {
    "acc",
    "big_ten",
    "big_twelve",
    "big_east",
    "sec",
    "pac_ten",
    "pac_twelve",
    "aac",
}


def build_conference_features(team_conf_df: pd.DataFrame, conferences_df: pd.DataFrame) -> pd.DataFrame:
    """Return per-(Season, TeamID) conference metadata and a simple power-conference flag."""
    join_key = next((c for c in CONF_JOIN_KEY_CANDIDATES if c in conferences_df.columns), None)
    if join_key is None:
        raise ValueError(f"Could not infer conference join key from columns: {list(conferences_df.columns)}")

    conf_name_col = next((c for c in CONF_NAME_CANDIDATES if c in conferences_df.columns), None)
    if conf_name_col is None:
        conf_name_col = join_key

    conf_lookup = conferences_df[[join_key, conf_name_col]].copy().drop_duplicates(subset=[join_key])
    if join_key != "ConfAbbrev":
        conf_lookup = conf_lookup.rename(columns={join_key: "ConfAbbrev"})
    if conf_name_col != "ConfName":
        conf_lookup = conf_lookup.rename(columns={conf_name_col: "ConfName"})

    out = team_conf_df[["Season", "TeamID", "ConfAbbrev"]].copy()
    out["ConfAbbrev"] = out["ConfAbbrev"].astype(str).str.lower()
    conf_lookup["ConfAbbrev"] = conf_lookup["ConfAbbrev"].astype(str).str.lower()
    out = out.merge(conf_lookup[["ConfAbbrev", "ConfName"]], on="ConfAbbrev", how="left")
    out["IsPowerConf"] = out["ConfAbbrev"].isin(POWER_CONFS).astype(int)
    return out.sort_values(["Season", "TeamID"]).reset_index(drop=True)


def build_and_save_conference_features(data_dir: Path) -> pd.DataFrame:
    """Load conference assignment files, build features, and save to processed CSV."""
    team_conf_df = read_csv_checked(
        raw_path(data_dir, "MTeamConferences.csv"),
        TEAM_CONF_REQUIRED_COLUMNS,
        name="MTeamConferences",
    )
    conferences_path = raw_path(data_dir, "Conferences.csv")
    if not conferences_path.exists():
        raise FileNotFoundError(f"Required file not found: {conferences_path}")
    conferences_df = pd.read_csv(conferences_path)

    out_df = build_conference_features(team_conf_df, conferences_df)
    out_df.to_csv(processed_path(data_dir, "team_season_conference_features.csv"), index=False)
    return out_df

