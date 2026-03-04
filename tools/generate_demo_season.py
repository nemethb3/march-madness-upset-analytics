"""Generate a synthetic bracket-ready season bundle under data/app/{season}."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

REGIONS = ["W", "X", "Y", "Z"]
SEED_CODE_RE = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")
ROUND1_TEMPLATE = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Generate synthetic demo season bundle for Streamlit app.")
    parser.add_argument("--source_season", type=int, default=None, help="Source season to copy structure from.")
    parser.add_argument("--target_season", type=int, default=2026, help="Target synthetic season.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for deterministic synthetic noise.")
    return parser.parse_args()


def _seed_num(seed_code: str) -> int:
    m = re.search(r"\d+", str(seed_code))
    if m is None:
        raise ValueError(f"Invalid seed code: {seed_code}")
    return int(m.group(0))


def _load_source_slots(source_season: int) -> tuple[pd.DataFrame, str]:
    """Load source slots from raw first, then app bundle; return (slots, source_label)."""
    raw_path = Path("data/raw/MNCAATourneySlots.csv")
    if raw_path.exists():
        raw = pd.read_csv(raw_path)
        raw_s = raw[raw["Season"] == source_season].copy()
        if not raw_s.empty:
            return raw_s, "raw"

    bundle_path = Path("data/app") / str(source_season) / "slots.csv"
    if bundle_path.exists():
        bundled = pd.read_csv(bundle_path)
        bundled_s = bundled[bundled["Season"] == source_season].copy()
        if not bundled_s.empty:
            return bundled_s, "bundle"

    raise RuntimeError(f"No slot structure found for source season {source_season}.")


def _round1_slot_count(slots_df: pd.DataFrame) -> int:
    mask = slots_df["StrongSeed"].astype(str).str.match(SEED_CODE_RE) & slots_df["WeakSeed"].astype(str).str.match(SEED_CODE_RE)
    return int(mask.sum())


def _is_slots_full_bracket(slots_df: pd.DataFrame) -> bool:
    """Treat full bracket as at least 63 games and at least 32 seed-vs-seed Round 1 games."""
    return len(slots_df) >= 63 and _round1_slot_count(slots_df) >= 32


def _build_standard_slots(season: int) -> pd.DataFrame:
    """Build a 64-team slots table with full rounds (63 games)."""
    rows: list[dict[str, str | int]] = []
    for region in REGIONS:
        for i, (a, b) in enumerate(ROUND1_TEMPLATE, start=1):
            rows.append({"Season": season, "Slot": f"R1{region}{i}", "StrongSeed": f"{region}{a:02d}", "WeakSeed": f"{region}{b:02d}"})
        rows += [
            {"Season": season, "Slot": f"R2{region}1", "StrongSeed": f"R1{region}1", "WeakSeed": f"R1{region}2"},
            {"Season": season, "Slot": f"R2{region}2", "StrongSeed": f"R1{region}3", "WeakSeed": f"R1{region}4"},
            {"Season": season, "Slot": f"R2{region}3", "StrongSeed": f"R1{region}5", "WeakSeed": f"R1{region}6"},
            {"Season": season, "Slot": f"R2{region}4", "StrongSeed": f"R1{region}7", "WeakSeed": f"R1{region}8"},
            {"Season": season, "Slot": f"R3{region}1", "StrongSeed": f"R2{region}1", "WeakSeed": f"R2{region}2"},
            {"Season": season, "Slot": f"R3{region}2", "StrongSeed": f"R2{region}3", "WeakSeed": f"R2{region}4"},
            {"Season": season, "Slot": f"R4{region}1", "StrongSeed": f"R3{region}1", "WeakSeed": f"R3{region}2"},
        ]
    rows += [
        {"Season": season, "Slot": "R5WX", "StrongSeed": "R4W1", "WeakSeed": "R4X1"},
        {"Season": season, "Slot": "R5YZ", "StrongSeed": "R4Y1", "WeakSeed": "R4Z1"},
        {"Season": season, "Slot": "R6CH", "StrongSeed": "R5WX", "WeakSeed": "R5YZ"},
    ]
    return pd.DataFrame(rows)


def _required_seed_codes_from_slots(slots_df: pd.DataFrame) -> list[str]:
    """Return all seed codes required by slots (inputs that are direct seed references)."""
    codes: set[str] = set()
    for col in ["StrongSeed", "WeakSeed"]:
        vals = slots_df[col].astype(str)
        for v in vals[vals.str.match(SEED_CODE_RE)]:
            codes.add(v)
    return sorted(codes)


def _build_seed_assignments(
    raw_seeds: pd.DataFrame,
    source_season: int,
    required_seed_codes: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create seeds rows for required seed codes with deterministic randomized team assignment."""
    seed_col = "Seed" if "Seed" in raw_seeds.columns else "SeedStr"
    source = raw_seeds[raw_seeds["Season"] == source_season][[seed_col, "TeamID"]].copy().rename(columns={seed_col: "Seed"})
    if source.empty:
        raise RuntimeError(f"No source seeds found for season {source_season}.")

    source["Region"] = source["Seed"].astype(str).str[0]
    source["SeedNum"] = source["Seed"].map(_seed_num)
    source["BaseSeed"] = source["Region"] + source["SeedNum"].astype(str).str.zfill(2)

    required_by_region: dict[str, list[str]] = {r: [] for r in REGIONS}
    for code in required_seed_codes:
        required_by_region[code[0]].append(code)

    global_pool = source["TeamID"].drop_duplicates().astype(int).tolist()
    rng.shuffle(global_pool)
    used: set[int] = set()
    out_rows: list[dict[str, int | str]] = []

    for region in REGIONS:
        req_codes = sorted(required_by_region.get(region, []), key=lambda c: (int(re.search(r"\d+", c).group(0)), c))
        if not req_codes:
            continue

        src_reg = source[source["Region"] == region]["TeamID"].drop_duplicates().astype(int).tolist()
        rng.shuffle(src_reg)
        reg_iter = [tid for tid in src_reg if tid not in used]

        for code in req_codes:
            pick: int | None = None
            if reg_iter:
                pick = reg_iter.pop(0)
            else:
                while global_pool and global_pool[0] in used:
                    global_pool.pop(0)
                if global_pool:
                    pick = global_pool.pop(0)
            if pick is None:
                raise RuntimeError(f"Unable to assign unique TeamID for seed code {code}.")
            used.add(pick)
            out_rows.append({"Seed": code, "TeamID": int(pick)})

    out = pd.DataFrame(out_rows).drop_duplicates(subset=["Seed"], keep="first")
    if len(out) != len(required_seed_codes):
        missing = sorted(set(required_seed_codes) - set(out["Seed"].tolist()))
        raise RuntimeError(f"Failed to assign all required seed codes: missing {missing[:10]}")
    return out.sort_values("Seed").reset_index(drop=True)


def _build_team_features(source_season: int, target_season: int, team_ids: list[int], rng: np.random.Generator) -> pd.DataFrame:
    """Create synthetic team feature rows for selected TeamIDs."""
    feats_path = Path("data/processed/team_season_features.csv")
    if not feats_path.exists():
        raise RuntimeError("Missing data/processed/team_season_features.csv")

    feats = pd.read_csv(feats_path)
    src = feats[feats["Season"] == source_season].copy()
    if src.empty:
        raise RuntimeError(f"No team features found for source season {source_season}.")

    subset = src[src["TeamID"].isin(team_ids)].copy()
    missing = sorted(set(team_ids) - set(subset["TeamID"].astype(int).tolist()))
    if missing:
        raise RuntimeError(f"Missing source team features for TeamIDs: {missing[:10]}")

    numeric_cols = [c for c in subset.columns if pd.api.types.is_numeric_dtype(subset[c]) and c not in {"Season", "TeamID"}]
    for col in numeric_cols:
        std = subset[col].std(skipna=True)
        scale = 0.05 * float(std if pd.notna(std) and std > 0 else 1.0)
        noise = rng.normal(0.0, scale, len(subset))
        vals = subset[col].astype(float) + noise
        lname = col.lower()
        if any(tok in lname for tok in ["pct", "rate", "prob"]) and "rank" not in lname:
            vals = vals.clip(0.0, 1.0)
        if any(tok in lname for tok in ["games", "wins", "loss", "points", "count", "num", "seed"]):
            vals = vals.clip(lower=0.0)
        subset[col] = vals

    subset["Season"] = target_season
    subset = subset.drop_duplicates(subset=["TeamID"], keep="first").sort_values("TeamID").reset_index(drop=True)
    return subset


def _validate_round1(seeds_df: pd.DataFrame, slots_df: pd.DataFrame) -> tuple[int, list[str]]:
    """Return Round 1 game count and list of found seed pair labels."""
    seed_map = dict(zip(seeds_df["Seed"].astype(str), seeds_df["TeamID"]))
    mask = slots_df["StrongSeed"].astype(str).str.match(SEED_CODE_RE) & slots_df["WeakSeed"].astype(str).str.match(SEED_CODE_RE)
    r1 = slots_df.loc[mask, ["StrongSeed", "WeakSeed"]].copy()
    if r1.empty:
        return 0, []
    r1 = r1[r1["StrongSeed"].isin(seed_map) & r1["WeakSeed"].isin(seed_map)].copy()
    r1["Pair"] = r1.apply(lambda r: f"{min(_seed_num(r['StrongSeed']), _seed_num(r['WeakSeed']))} vs {max(_seed_num(r['StrongSeed']), _seed_num(r['WeakSeed']))}", axis=1)
    return len(r1), sorted(r1["Pair"].dropna().unique().tolist())


def _determine_source_season(raw_seeds: pd.DataFrame, explicit_source: int | None) -> int:
    if explicit_source is not None:
        return explicit_source
    seasons = sorted(raw_seeds["Season"].dropna().astype(int).unique().tolist())
    if not seasons:
        raise RuntimeError("No seasons found in data/raw/MNCAATourneySeeds.csv")
    return seasons[-1]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    app_root = Path("data/app")
    app_root.mkdir(parents=True, exist_ok=True)

    raw_seeds_path = Path("data/raw/MNCAATourneySeeds.csv")
    raw_teams_path = Path("data/raw/MTeams.csv")
    if not raw_seeds_path.exists():
        raise RuntimeError("Missing data/raw/MNCAATourneySeeds.csv")
    raw_seeds = pd.read_csv(raw_seeds_path)
    source_season = _determine_source_season(raw_seeds, args.source_season)

    source_slots, slots_source = _load_source_slots(source_season)
    used_fallback_64 = False
    if _is_slots_full_bracket(source_slots):
        slots_out = source_slots.copy()
        slots_out["Season"] = args.target_season
        required_seed_codes = _required_seed_codes_from_slots(slots_out)
    else:
        used_fallback_64 = True
        slots_out = _build_standard_slots(args.target_season)
        required_seed_codes = _required_seed_codes_from_slots(slots_out)

    seeds_out = _build_seed_assignments(raw_seeds, source_season, required_seed_codes, rng)
    seeds_out["Season"] = args.target_season
    seeds_out = seeds_out[["Season", "Seed", "TeamID"]]

    team_ids = seeds_out["TeamID"].astype(int).tolist()
    feats_out = _build_team_features(source_season, args.target_season, team_ids, rng)

    round1_count, seed_pairs = _validate_round1(seeds_out, slots_out)
    if round1_count < 32:
        raise RuntimeError(f"Validation failed: expected at least 32 Round 1 games, got {round1_count}")
    expected_pairs = {f"{a} vs {b}" for a, b in ROUND1_TEMPLATE}
    missing_pairs = expected_pairs - set(seed_pairs)
    if missing_pairs:
        raise RuntimeError(f"Validation failed: missing Round 1 seed pair types: {sorted(missing_pairs)}")

    target_dir = app_root / str(args.target_season)
    target_dir.mkdir(parents=True, exist_ok=True)
    seeds_out.to_csv(target_dir / "seeds.csv", index=False)
    slots_out.to_csv(target_dir / "slots.csv", index=False)
    feats_out.to_csv(target_dir / "team_features.csv", index=False)

    has_first_four = seeds_out["Seed"].astype(str).str.endswith(("a", "b")).any()
    note_lines = [
        f"Synthetic demo bundle for {args.target_season}",
        f"Source season: {source_season}",
        f"Slots source: {slots_source}",
        f"Fallback to 64-team synthetic slots: {used_fallback_64}",
        f"First Four included: {bool(has_first_four)}",
    ]
    (target_dir / "README_demo.txt").write_text("\n".join(note_lines) + "\n", encoding="utf-8")

    # Refresh team map with all known teams.
    team_map_path = app_root / "team_id_map.csv"
    if raw_teams_path.exists():
        teams = pd.read_csv(raw_teams_path)[["TeamID", "TeamName"]].copy()
    else:
        teams = pd.DataFrame({"TeamID": team_ids, "TeamName": [f"Team {tid}" for tid in team_ids]})
    existing = pd.read_csv(team_map_path) if team_map_path.exists() else pd.DataFrame(columns=["TeamID", "TeamName"])
    merged = (
        pd.concat([existing[["TeamID", "TeamName"]], teams[["TeamID", "TeamName"]]], ignore_index=True)
        .drop_duplicates(subset=["TeamID"], keep="last")
        .sort_values("TeamID")
        .reset_index(drop=True)
    )
    merged.to_csv(team_map_path, index=False)

    print(f"Synthetic demo bundle created for {args.target_season}")
    print(f"Source season: {source_season}")
    print(f"Slots source: {slots_source}")
    print(f"Fallback64: {used_fallback_64}")
    print(f"Seeds rows: {len(seeds_out)}")
    print(f"Unique seeded teams: {seeds_out['TeamID'].nunique()}")
    print(f"Team features rows: {len(feats_out)}")
    print(f"Round1 games: {round1_count}")
    print(f"FirstFourIncluded: {bool(has_first_four)}")
    print(f"Output dir: {target_dir}")


if __name__ == "__main__":
    main()
