"""Generate a synthetic, bracket-ready demo season bundle under data/app/{season}."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROUND1_TEMPLATE = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
REGIONS = ["W", "X", "Y", "Z"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic bracket-ready demo season.")
    parser.add_argument("--source_season", type=int, default=None, help="Source season (default: latest available).")
    parser.add_argument("--target_season", type=int, default=2026, help="Target synthetic season.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic RNG seed.")
    return parser.parse_args()


def _list_bundle_seasons(root: Path) -> list[int]:
    seasons: list[int] = []
    if not root.exists():
        return seasons
    for p in root.iterdir():
        if p.is_dir() and p.name.isdigit():
            seasons.append(int(p.name))
    return sorted(seasons)


def _seed_num(seed_str: str) -> int:
    m = re.search(r"\d+", str(seed_str))
    if m is None:
        raise ValueError(f"Invalid seed string: {seed_str}")
    return int(m.group(0))


def _seed_suffix(seed_str: str) -> str:
    s = str(seed_str)
    m = re.search(r"\d+", s)
    return "" if m is None else s[m.end() :]


def _canonical_64_seeds(raw_seeds: pd.DataFrame, source_season: int) -> pd.DataFrame:
    """Extract one team per (region, seed_num) from source season seeds."""
    seed_col = "Seed" if "Seed" in raw_seeds.columns else "SeedStr"
    s = raw_seeds[raw_seeds["Season"] == source_season][["Season", seed_col, "TeamID"]].copy()
    s = s.rename(columns={seed_col: "Seed"})
    if s.empty:
        raise RuntimeError(f"No seeds found for source season {source_season}.")
    s["Region"] = s["Seed"].astype(str).str[0]
    s["SeedNum"] = s["Seed"].map(_seed_num)
    s["SeedSuffix"] = s["Seed"].map(_seed_suffix)
    # Keep deterministic representative for play-ins (a/b): first by suffix then TeamID.
    s = s.sort_values(["Region", "SeedNum", "SeedSuffix", "TeamID"]).drop_duplicates(["Region", "SeedNum"], keep="first")
    # Require full 64 structure.
    required = {(r, n) for r in REGIONS for n in range(1, 17)}
    present = {(str(r["Region"]), int(r["SeedNum"])) for _, r in s.iterrows()}
    missing = sorted(required - present)
    if missing:
        raise RuntimeError(f"Source season missing required seeds for full 64-team bracket: {missing[:10]}")
    return s[["Region", "SeedNum", "TeamID"]].copy()


def _jitter_seed_numbers(df64: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Jitter seed numbers in each region while preserving unique 1..16."""
    rows: list[pd.DataFrame] = []
    for region, grp in df64.groupby("Region", sort=True):
        g = grp.copy()
        g["Jittered"] = (g["SeedNum"] + rng.integers(-2, 3, size=len(g))).clip(1, 16)
        used: set[int] = set()
        available = set(range(1, 17))
        assigned: list[int] = []
        for _, r in g.sort_values("SeedNum").iterrows():
            target = int(r["Jittered"])
            if target not in used:
                pick = target
            else:
                pick = min(available, key=lambda x: (abs(x - target), x))
            used.add(pick)
            available.discard(pick)
            assigned.append(pick)
        g = g.sort_values("SeedNum").copy()
        g["NewSeedNum"] = assigned
        g["Seed"] = g.apply(lambda r: f"{region}{int(r['NewSeedNum']):02d}", axis=1)
        rows.append(g[["Seed", "TeamID"]])
    out = pd.concat(rows, ignore_index=True)
    return out


def _build_full_slots(season: int) -> pd.DataFrame:
    """Build synthetic full bracket slots (63 games, no First Four)."""
    rows: list[dict[str, str | int]] = []
    for region in REGIONS:
        # Round 1
        for i, (a, b) in enumerate(ROUND1_TEMPLATE, start=1):
            rows.append({"Season": season, "Slot": f"R1{region}{i}", "StrongSeed": f"{region}{a:02d}", "WeakSeed": f"{region}{b:02d}"})
        # Round 2
        rows += [
            {"Season": season, "Slot": f"R2{region}1", "StrongSeed": f"R1{region}1", "WeakSeed": f"R1{region}2"},
            {"Season": season, "Slot": f"R2{region}2", "StrongSeed": f"R1{region}3", "WeakSeed": f"R1{region}4"},
            {"Season": season, "Slot": f"R2{region}3", "StrongSeed": f"R1{region}5", "WeakSeed": f"R1{region}6"},
            {"Season": season, "Slot": f"R2{region}4", "StrongSeed": f"R1{region}7", "WeakSeed": f"R1{region}8"},
        ]
        # Sweet 16
        rows += [
            {"Season": season, "Slot": f"R3{region}1", "StrongSeed": f"R2{region}1", "WeakSeed": f"R2{region}2"},
            {"Season": season, "Slot": f"R3{region}2", "StrongSeed": f"R2{region}3", "WeakSeed": f"R2{region}4"},
        ]
        # Elite 8 (regional final)
        rows.append({"Season": season, "Slot": f"R4{region}1", "StrongSeed": f"R3{region}1", "WeakSeed": f"R3{region}2"})

    # Final Four + Championship
    rows += [
        {"Season": season, "Slot": "R5WX", "StrongSeed": "R4W1", "WeakSeed": "R4X1"},
        {"Season": season, "Slot": "R5YZ", "StrongSeed": "R4Y1", "WeakSeed": "R4Z1"},
        {"Season": season, "Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"},
    ]
    return pd.DataFrame(rows)


def _build_round1_from_slots(seeds: pd.DataFrame, slots: pd.DataFrame) -> pd.DataFrame:
    """Quick internal round1 extraction used for validation."""
    seed_map = dict(zip(seeds["Seed"], seeds["TeamID"]))
    mask = slots["StrongSeed"].astype(str).str.match(r"^[WXYZ]\d{2}$") & slots["WeakSeed"].astype(str).str.match(r"^[WXYZ]\d{2}$")
    r1 = slots.loc[mask, ["Slot", "StrongSeed", "WeakSeed"]].copy()
    r1["TeamAID"] = r1["StrongSeed"].map(seed_map)
    r1["TeamBID"] = r1["WeakSeed"].map(seed_map)
    r1["SeedA"] = r1["StrongSeed"].map(_seed_num)
    r1["SeedB"] = r1["WeakSeed"].map(_seed_num)
    r1["SeedPair"] = r1.apply(lambda r: f"{min(int(r['SeedA']), int(r['SeedB']))} vs {max(int(r['SeedA']), int(r['SeedB']))}", axis=1)
    return r1


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    app_root = Path("data/app")
    app_root.mkdir(parents=True, exist_ok=True)

    raw_seeds_path = Path("data/raw/MNCAATourneySeeds.csv")
    raw_teams_path = Path("data/raw/MTeams.csv")
    processed_feats_path = Path("data/processed/team_season_features.csv")

    if not raw_seeds_path.exists():
        raise RuntimeError("Missing data/raw/MNCAATourneySeeds.csv")
    if not processed_feats_path.exists():
        raise RuntimeError("Missing data/processed/team_season_features.csv")

    raw_seeds = pd.read_csv(raw_seeds_path)
    seed_col = "Seed" if "Seed" in raw_seeds.columns else "SeedStr"
    available_raw = sorted(raw_seeds["Season"].dropna().astype(int).unique().tolist())
    bundle_seasons = _list_bundle_seasons(app_root)

    if args.source_season is not None:
        source_season = args.source_season
    elif available_raw:
        source_season = available_raw[-1]
    elif bundle_seasons:
        source_season = bundle_seasons[-1]
    else:
        raise RuntimeError("No source season available in raw seeds or app bundles.")

    seeds64 = _canonical_64_seeds(raw_seeds.rename(columns={seed_col: "Seed"}), source_season)
    seeds_out = _jitter_seed_numbers(seeds64, rng)
    seeds_out["Season"] = args.target_season
    seeds_out = seeds_out[["Season", "Seed", "TeamID"]].sort_values("Seed").reset_index(drop=True)

    slots_out = _build_full_slots(args.target_season)

    feats = pd.read_csv(processed_feats_path)
    source_feats = feats[feats["Season"] == source_season].copy()
    if source_feats.empty:
        raise RuntimeError(f"No team features found for source season {source_season}.")
    needed_teams = set(seeds_out["TeamID"].astype(int).tolist())
    feat_subset = source_feats[source_feats["TeamID"].isin(needed_teams)].copy()
    missing_feat = sorted(needed_teams - set(feat_subset["TeamID"].astype(int).tolist()))
    if missing_feat:
        raise RuntimeError(f"Missing team features for seeded teams: {missing_feat[:10]}")

    num_cols = [c for c in feat_subset.columns if pd.api.types.is_numeric_dtype(feat_subset[c]) and c not in {"Season", "TeamID"}]
    for col in num_cols:
        std = feat_subset[col].std(skipna=True)
        scale = 0.05 * float(std if pd.notna(std) and std > 0 else 1.0)
        noise = rng.normal(0.0, scale, len(feat_subset))
        vals = feat_subset[col].astype(float) + noise
        lower_name = col.lower()
        if any(tok in lower_name for tok in ["pct", "rate", "prob", "score"]) and "rank" not in lower_name:
            vals = vals.clip(0.0, 1.0)
        if any(tok in lower_name for tok in ["games", "wins", "loss", "points", "count", "num", "seed"]):
            vals = vals.clip(lower=0.0)
        feat_subset[col] = vals
    feat_subset["Season"] = args.target_season
    feat_subset = feat_subset.sort_values("TeamID").reset_index(drop=True)

    # Validation: 32 round1 games and 8 seed pair archetypes.
    r1 = _build_round1_from_slots(seeds_out, slots_out)
    if len(r1) != 32:
        raise RuntimeError(f"Validation failed: expected 32 Round 1 games, got {len(r1)}")
    expected_pairs = {f"{a} vs {b}" for a, b in ROUND1_TEMPLATE}
    found_pairs = set(r1["SeedPair"].unique().tolist())
    missing_pairs = expected_pairs - found_pairs
    if missing_pairs:
        raise RuntimeError(f"Validation failed: missing seed pair types: {sorted(missing_pairs)}")

    target_dir = app_root / str(args.target_season)
    target_dir.mkdir(parents=True, exist_ok=True)
    seeds_out.to_csv(target_dir / "seeds.csv", index=False)
    slots_out.to_csv(target_dir / "slots.csv", index=False)
    feat_subset.to_csv(target_dir / "team_features.csv", index=False)

    # Update team_id_map with all teams used by bundles.
    team_id_map_path = app_root / "team_id_map.csv"
    if raw_teams_path.exists():
        teams = pd.read_csv(raw_teams_path)[["TeamID", "TeamName"]]
    else:
        teams = pd.DataFrame({"TeamID": sorted(needed_teams), "TeamName": [f"Team {tid}" for tid in sorted(needed_teams)]})
    existing = pd.read_csv(team_id_map_path) if team_id_map_path.exists() else pd.DataFrame(columns=["TeamID", "TeamName"])
    merged = pd.concat([existing[["TeamID", "TeamName"]], teams[["TeamID", "TeamName"]]], ignore_index=True).drop_duplicates(
        subset=["TeamID"], keep="last"
    )
    merged = merged.sort_values("TeamID").reset_index(drop=True)
    merged.to_csv(team_id_map_path, index=False)

    print(f"Synthetic demo bundle created for {args.target_season}")
    print(f"Source season: {source_season}")
    print(f"Round1 games: {len(r1)}")
    print(f"Output dir: {target_dir}")


if __name__ == "__main__":
    main()

