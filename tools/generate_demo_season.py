"""Generate a synthetic demo season bundle under data/app/2026."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic demo season bundle.")
    parser.add_argument("--source_season", type=int, default=None, help="Source season from data/app to copy/jitter.")
    parser.add_argument("--target_season", type=int, default=2026, help="Target synthetic season.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic generation.")
    return parser.parse_args()


def _list_bundle_seasons(root: Path) -> list[int]:
    seasons = []
    if not root.exists():
        return seasons
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            seasons.append(int(p.name))
    return sorted(seasons)


def _seed_num(seed_str: str) -> int:
    m = re.search(r"\d+", str(seed_str))
    if m is None:
        raise ValueError(f"Invalid seed code: {seed_str}")
    return int(m.group(0))


def _seed_suffix(seed_str: str) -> str:
    s = str(seed_str)
    m = re.search(r"\d+", s)
    if m is None:
        return ""
    return s[m.end() :]


def _rebuild_seed(region: str, num: int, suffix: str) -> str:
    return f"{region}{int(num):02d}{suffix}"


def _jitter_region_seeds(region_df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Jitter seed numbers while preserving unique 1..16 in each region."""
    out = region_df.copy()
    out["BaseNum"] = out["Seed"].map(_seed_num)
    out["Suffix"] = out["Seed"].map(_seed_suffix)
    out["Jittered"] = out["BaseNum"] + rng.integers(-2, 3, size=len(out))
    out["Jittered"] = out["Jittered"].clip(1, 16)

    # Enforce uniqueness deterministically using nearest available seed numbers.
    used: set[int] = set()
    assigned: list[int] = []
    available = set(range(1, 17))
    for _, r in out.sort_values("BaseNum").iterrows():
        target = int(r["Jittered"])
        if target not in used:
            pick = target
        else:
            pick = min(available, key=lambda x: (abs(x - target), x))
        used.add(pick)
        available.discard(pick)
        assigned.append(pick)
    out = out.sort_values("BaseNum").copy()
    out["NewSeedNum"] = assigned
    out["Seed"] = out.apply(lambda r: _rebuild_seed(str(r["Region"]), int(r["NewSeedNum"]), str(r["Suffix"])), axis=1)
    return out[["Season", "Seed", "TeamID"]]


def main() -> None:
    args = parse_args()
    root = Path("data/app")
    seasons = _list_bundle_seasons(root)
    if not seasons:
        raise RuntimeError("No source seasons found under data/app/")

    source = args.source_season if args.source_season is not None else seasons[-1]
    source_dir = root / str(source)
    target_dir = root / str(args.target_season)
    target_dir.mkdir(parents=True, exist_ok=True)

    seeds = pd.read_csv(source_dir / "seeds.csv")
    slots = pd.read_csv(source_dir / "slots.csv")
    feats = pd.read_csv(source_dir / "team_features.csv")

    if "Seed" not in seeds.columns:
        if "SeedStr" in seeds.columns:
            seeds = seeds.rename(columns={"SeedStr": "Seed"})
        else:
            raise ValueError("Source seeds.csv must contain Seed or SeedStr")

    rng = np.random.default_rng(args.seed)

    # Seeds: jitter within region with unique 1..16 preservation.
    seeds_out = seeds.copy()
    seeds_out["Region"] = seeds_out["Seed"].astype(str).str[0]
    parts = []
    for region, grp in seeds_out.groupby("Region", sort=True):
        g = grp.copy()
        g["Region"] = region
        parts.append(_jitter_region_seeds(g, rng))
    seeds_out = pd.concat(parts, ignore_index=True)
    seeds_out["Season"] = args.target_season

    # Slots: copy structure with target season.
    slots_out = slots.copy()
    slots_out["Season"] = args.target_season

    # Team features: add small noise to numeric columns.
    feats_out = feats.copy()
    num_cols = [c for c in feats_out.columns if pd.api.types.is_numeric_dtype(feats_out[c]) and c not in {"Season", "TeamID"}]
    for col in num_cols:
        s = feats_out[col].std(skipna=True)
        scale = 0.05 * float(s if pd.notna(s) and s > 0 else 1.0)
        noise = rng.normal(loc=0.0, scale=scale, size=len(feats_out))
        vals = feats_out[col].astype(float) + noise
        # Clip obvious bounded rates.
        if any(tok in col.lower() for tok in ["pct", "rate", "prob"]):
            vals = vals.clip(0.0, 1.0)
        # Keep non-negative for count-like features.
        if any(tok in col.lower() for tok in ["games", "wins", "loss", "points", "count", "num"]):
            vals = vals.clip(lower=0.0)
        feats_out[col] = vals
    feats_out["Season"] = args.target_season

    seeds_out.to_csv(target_dir / "seeds.csv", index=False)
    slots_out.to_csv(target_dir / "slots.csv", index=False)
    feats_out.to_csv(target_dir / "team_features.csv", index=False)

    print(f"Synthetic demo bundle created for {args.target_season}")
    print(f"Source season: {source}")
    print(f"Output: {target_dir}")


if __name__ == "__main__":
    main()

