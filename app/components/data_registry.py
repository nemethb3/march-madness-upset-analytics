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
    paths: dict[str, Path]
    cache_token: str


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


def get_bundle_paths(season: int) -> dict[str, Path]:
    """Return required file paths for a season bundle."""
    root = _app_data_root()
    season_dir = root / str(season)
    return {
        "seeds": season_dir / "seeds.csv",
        "slots": season_dir / "slots.csv",
        "team_features": season_dir / "team_features.csv",
        "team_id_map": root / "team_id_map.csv",
    }


def bundle_cache_token(season: int) -> str:
    """Return cache token based on season and required file mtimes/sizes."""
    paths = get_bundle_paths(season)
    parts: list[str] = [str(season)]
    for key in ["seeds", "slots", "team_features", "team_id_map"]:
        p = paths[key]
        if not p.exists():
            parts.append(f"{key}:MISSING")
            continue
        st = p.stat()
        parts.append(f"{key}:{st.st_mtime_ns}:{st.st_size}")
    return "|".join(parts)


def _validate_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    """Validate dataframe required columns."""
    missing = sorted(required - set(df.columns))
    if missing:
        raise BundleMissingError(f"{label} missing columns: {missing}")


def _coerce_int_col(df: pd.DataFrame, col: str, label: str) -> pd.DataFrame:
    """Coerce integer-like ID columns safely."""
    out = df.copy()
    out[col] = pd.to_numeric(out[col], errors="coerce")
    bad = int(out[col].isna().sum())
    if bad:
        raise BundleMissingError(f"{label} has {bad} non-numeric values in {col}")
    out[col] = out[col].astype("Int64")
    return out


def _ensure_season(df: pd.DataFrame, season: int, label: str) -> pd.DataFrame:
    """Ensure dataframe has season column and only selected season rows."""
    out = df.copy()
    if "Season" not in out.columns:
        out["Season"] = season
    out["Season"] = pd.to_numeric(out["Season"], errors="coerce").astype("Int64")
    out = out[out["Season"] == season].copy()
    if out.empty:
        raise BundleMissingError(f"{label} has no rows for selected season {season}")
    return out


def load_season_bundle(season: int) -> SeasonBundle:
    """Load season bundle and validate required schema with season isolation."""
    required_paths = get_bundle_paths(season)
    missing_files = [name for name, p in required_paths.items() if not p.exists()]
    if missing_files:
        raise BundleMissingError(f"Missing bundle files for season {season}: {missing_files}")

    seeds = pd.read_csv(required_paths["seeds"])
    slots = pd.read_csv(required_paths["slots"])
    feats = pd.read_csv(required_paths["team_features"])
    team_map = pd.read_csv(required_paths["team_id_map"])

    _validate_columns(seeds, {"TeamID"}, "seeds.csv")
    if not ({"Seed"} <= set(seeds.columns) or {"SeedStr"} <= set(seeds.columns)):
        raise BundleMissingError("seeds.csv must include Seed or SeedStr column.")
    _validate_columns(slots, {"Slot", "StrongSeed", "WeakSeed"}, "slots.csv")
    _validate_columns(feats, {"TeamID"}, "team_features.csv")
    _validate_columns(team_map, {"TeamID", "TeamName"}, "team_id_map.csv")

    seeds = _ensure_season(seeds, season, "seeds.csv")
    slots = _ensure_season(slots, season, "slots.csv")
    feats = _ensure_season(feats, season, "team_features.csv")
    team_map = _coerce_int_col(team_map, "TeamID", "team_id_map.csv")

    seeds = _coerce_int_col(seeds, "TeamID", "seeds.csv")
    feats = _coerce_int_col(feats, "TeamID", "team_features.csv")
    seeds["TeamID"] = seeds["TeamID"].astype(int)
    feats["TeamID"] = feats["TeamID"].astype(int)
    team_map["TeamID"] = team_map["TeamID"].astype(int)

    slots["Slot"] = slots["Slot"].astype(str)
    slots["StrongSeed"] = slots["StrongSeed"].astype(str)
    slots["WeakSeed"] = slots["WeakSeed"].astype(str)

    seed_col = "Seed" if "Seed" in seeds.columns else "SeedStr"
    seeds[seed_col] = seeds[seed_col].astype(str)
    seeds = seeds.drop_duplicates(subset=[seed_col, "TeamID"]).copy()
    feats = feats.drop_duplicates(subset=["Season", "TeamID"], keep="first").copy()
    team_map = team_map.drop_duplicates(subset=["TeamID"], keep="last").copy()

    return SeasonBundle(
        season=season,
        seeds=seeds.reset_index(drop=True),
        slots=slots.reset_index(drop=True),
        team_features=feats.reset_index(drop=True),
        team_id_map=team_map.reset_index(drop=True),
        mode="Bundle Mode",
        paths=required_paths,
        cache_token=bundle_cache_token(season),
    )
