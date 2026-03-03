"""Build Round 1 matchups directly from tournament slots and seeds."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SEED_CODE_PATTERN = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build Round 1 matchup CSV from MNCAATourneySlots.")
    parser.add_argument("--season", type=int, required=True, help="Tournament season to build.")
    parser.add_argument(
        "--out_csv",
        type=Path,
        required=True,
        help="Output path for round1 matchup CSV.",
    )
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root (default: data).")
    return parser.parse_args()


def _seed_num(seed_str: str) -> int:
    """Extract numeric seed from a seed code like W01 or W16a."""
    digits = re.findall(r"\d+", str(seed_str))
    if not digits:
        raise ValueError(f"Seed has no digits: {seed_str}")
    return int(digits[0])


def build_round1_matchups_from_frames(
    slots_df: pd.DataFrame,
    seeds_df: pd.DataFrame,
    season: int,
    teams_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build Round 1 matchups from in-memory slots/seeds dataframes."""
    slots_s = slots_df[slots_df["Season"] == season].copy()
    mask_round1 = slots_s["StrongSeed"].astype(str).str.match(SEED_CODE_PATTERN) & slots_s["WeakSeed"].astype(str).str.match(
        SEED_CODE_PATTERN
    )
    round1 = slots_s.loc[mask_round1, ["Season", "Slot", "StrongSeed", "WeakSeed"]].copy()
    if round1.empty:
        return pd.DataFrame(
            columns=["Season", "Slot", "TeamAID", "TeamBID", "TeamAName", "TeamBName", "TeamASeedNum", "TeamBSeedNum"]
        )

    seed_col = "Seed" if "Seed" in seeds_df.columns else "SeedStr"
    seed_map = seeds_df[seeds_df["Season"] == season][[seed_col, "TeamID"]].rename(columns={seed_col: "SeedStr"})
    seed_map["SeedNum"] = seed_map["SeedStr"].map(_seed_num)
    round1 = round1.merge(
        seed_map.rename(columns={"SeedStr": "StrongSeed", "TeamID": "TeamAID", "SeedNum": "TeamASeedNum"}),
        on="StrongSeed",
        how="left",
    )
    round1 = round1.merge(
        seed_map.rename(columns={"SeedStr": "WeakSeed", "TeamID": "TeamBID", "SeedNum": "TeamBSeedNum"}),
        on="WeakSeed",
        how="left",
    )

    if teams_df is not None and {"TeamID", "TeamName"}.issubset(teams_df.columns):
        round1 = round1.merge(
            teams_df[["TeamID", "TeamName"]].rename(columns={"TeamID": "TeamAID", "TeamName": "TeamAName"}),
            on="TeamAID",
            how="left",
        )
        round1 = round1.merge(
            teams_df[["TeamID", "TeamName"]].rename(columns={"TeamID": "TeamBID", "TeamName": "TeamBName"}),
            on="TeamBID",
            how="left",
        )
    else:
        round1["TeamAName"] = round1["TeamAID"].map(lambda x: f"Team {int(x)}" if pd.notna(x) else "")
        round1["TeamBName"] = round1["TeamBID"].map(lambda x: f"Team {int(x)}" if pd.notna(x) else "")

    out_cols = ["Season", "Slot", "TeamAID", "TeamBID", "TeamAName", "TeamBName", "TeamASeedNum", "TeamBSeedNum"]
    return round1[out_cols].sort_values("Slot").reset_index(drop=True)


def build_round1_matchups(data_dir: Path, season: int) -> pd.DataFrame:
    """Build Round 1 slots matchup table for the requested season."""
    slots_path = data_dir / "raw" / "MNCAATourneySlots.csv"
    seeds_path = data_dir / "raw" / "MNCAATourneySeeds.csv"
    teams_path = data_dir / "raw" / "MTeams.csv"
    for p in [slots_path, seeds_path, teams_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    slots = pd.read_csv(slots_path)
    seeds = pd.read_csv(seeds_path)
    teams = pd.read_csv(teams_path, usecols=["TeamID", "TeamName"])

    for col in ["Season", "Slot", "StrongSeed", "WeakSeed"]:
        if col not in slots.columns:
            raise ValueError(f"MNCAATourneySlots missing column: {col}")

    return build_round1_matchups_from_frames(slots, seeds, season=season, teams_df=teams)


def main() -> None:
    """CLI entrypoint for Round 1 matchup generation."""
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df = build_round1_matchups(args.data_dir, args.season)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_df)} round1 matchups to {args.out_csv}")


if __name__ == "__main__":
    main()
