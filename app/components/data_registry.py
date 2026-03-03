"""Season bundle registry for Streamlit app data loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


class BundleMissingError(RuntimeError):
    """Raised when a season bundle is missing required files or schema."""


@dataclass
class SeasonBundle:
    """Container for a season bundle."""

    season: int
    seeds: pd.DataFrame
    slots: pd.DataFrame
    team_features: pd.DataFrame
    team_id_map: pd.DataFrame
    mode: str


def _app_data_root() -> Path:
    """Return data/app directory."""
    return Path("data/app")


def list_available_seasons() -> list[int]:
    """Return sorted seasons found in data/app/{season} directories."""
    root = _app_data_root()
    if not root.exists():
        return []
    seasons: list[int] = []
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            seasons.append(int(p.name))
    return sorted(seasons)


def _validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Validate dataframe required columns."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise BundleMissingError(f"{label} missing columns: {missing}")


def load_season_bundle(season: int) -> SeasonBundle:
    """Load season bundle and validate required schema."""
    root = _app_data_root()
    season_dir = root / str(season)
    team_map_path = root / "team_id_map.csv"

    required_paths = {
        "seeds": season_dir / "seeds.csv",
        "slots": season_dir / "slots.csv",
        "team_features": season_dir / "team_features.csv",
        "team_id_map": team_map_path,
    }
    missing_files = [name for name, p in required_paths.items() if not p.exists()]
    if missing_files:
        raise BundleMissingError(f"Missing bundle files for season {season}: {missing_files}")

    seeds = pd.read_csv(required_paths["seeds"])
    slots = pd.read_csv(required_paths["slots"])
    feats = pd.read_csv(required_paths["team_features"])
    team_map = pd.read_csv(required_paths["team_id_map"])

    _validate_columns(seeds, {"Season", "TeamID"}, "seeds.csv")
    if not ({"Seed"} <= set(seeds.columns) or {"SeedStr"} <= set(seeds.columns)):
        raise BundleMissingError("seeds.csv must include Seed or SeedStr column.")
    _validate_columns(slots, {"Season", "Slot", "StrongSeed", "WeakSeed"}, "slots.csv")
    _validate_columns(feats, {"Season", "TeamID"}, "team_features.csv")
    _validate_columns(team_map, {"TeamID", "TeamName"}, "team_id_map.csv")

    return SeasonBundle(
        season=season,
        seeds=seeds,
        slots=slots,
        team_features=feats,
        team_id_map=team_map,
        mode="Bundle Mode",
    )

