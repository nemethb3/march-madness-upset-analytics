"""Build historical upset-rate baselines by tournament seed pair."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

SEED_NUM_RE = re.compile(r"\d+")


def _seed_num(seed_str: str) -> int:
    """Extract numeric seed from seed code, e.g. W01 -> 1 and W16a -> 16."""
    match = SEED_NUM_RE.search(str(seed_str))
    if match is None:
        raise ValueError(f"Could not parse seed number from: {seed_str}")
    return int(match.group(0))


def build_historical_upset_rates(data_dir: Path) -> pd.DataFrame:
    """Compute historical upset rates by seed matchup from compact tournament results."""
    results_path = data_dir / "raw" / "MNCAATourneyCompactResults.csv"
    seeds_path = data_dir / "raw" / "MNCAATourneySeeds.csv"
    for p in [results_path, seeds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    results = pd.read_csv(results_path, usecols=["Season", "WTeamID", "LTeamID"])
    seeds = pd.read_csv(seeds_path, usecols=["Season", "Seed", "TeamID"]).rename(columns={"Seed": "SeedStr"})
    seeds["SeedNum"] = seeds["SeedStr"].map(_seed_num)

    winner_seed = seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "WTeamID", "SeedNum": "WSeed"})
    loser_seed = seeds[["Season", "TeamID", "SeedNum"]].rename(columns={"TeamID": "LTeamID", "SeedNum": "LSeed"})
    data = results.merge(winner_seed, on=["Season", "WTeamID"], how="left").merge(loser_seed, on=["Season", "LTeamID"], how="left")
    data = data.dropna(subset=["WSeed", "LSeed"]).copy()
    data["WSeed"] = data["WSeed"].astype(int)
    data["LSeed"] = data["LSeed"].astype(int)
    data["SeedA"] = data[["WSeed", "LSeed"]].min(axis=1)
    data["SeedB"] = data[["WSeed", "LSeed"]].max(axis=1)
    data["IsUpset"] = (data["WSeed"] > data["LSeed"]).astype(int)

    out = (
        data.groupby(["SeedA", "SeedB"], as_index=False)
        .agg(
            HistoricalUpsetRate=("IsUpset", "mean"),
            GameCount=("IsUpset", "count"),
        )
        .sort_values(["SeedA", "SeedB"])
        .reset_index(drop=True)
    )
    return out


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Build historical upset rates by seed pair.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root.")
    parser.add_argument(
        "--out_csv",
        type=Path,
        default=Path("data/processed/historical_upset_rates.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out = build_historical_upset_rates(args.data_dir)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote historical upset rates: {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()

