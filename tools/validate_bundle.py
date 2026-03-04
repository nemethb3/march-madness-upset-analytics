"""Validate Streamlit season bundle completeness and bracket readiness."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

SEED_CODE_RE = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Validate data/app/{season} bundle.")
    parser.add_argument("--season", type=int, required=True, help="Season to validate.")
    parser.add_argument("--app_dir", type=Path, default=Path("data/app"), help="App bundle root directory.")
    return parser.parse_args()


def _seed_num(seed_code: str) -> int:
    match = re.search(r"\d+", str(seed_code))
    if match is None:
        raise ValueError(f"Invalid seed code: {seed_code}")
    return int(match.group(0))


def _print(line: str) -> None:
    print(line)


def main() -> None:
    args = parse_args()
    app_dir = args.app_dir
    season_dir = app_dir / str(args.season)

    required = {
        "seeds": season_dir / "seeds.csv",
        "slots": season_dir / "slots.csv",
        "team_features": season_dir / "team_features.csv",
        "team_id_map": app_dir / "team_id_map.csv",
    }

    failed = False
    _print(f"VALIDATE BUNDLE season={args.season}")
    _print(f"app_dir={app_dir}")
    _print("")

    for name, path in required.items():
        ok = path.exists()
        _print(f"[{'PASS' if ok else 'FAIL'}] file:{name} path={path}")
        failed = failed or not ok

    if failed:
        _print("")
        _print("RESULT: FAIL (missing required files)")
        sys.exit(1)

    seeds = pd.read_csv(required["seeds"])
    slots = pd.read_csv(required["slots"])
    feats = pd.read_csv(required["team_features"])
    team_map = pd.read_csv(required["team_id_map"])

    seed_col = "Seed" if "Seed" in seeds.columns else ("SeedStr" if "SeedStr" in seeds.columns else None)
    if seed_col is None:
        _print("[FAIL] seeds.csv missing Seed/SeedStr column")
        sys.exit(1)

    for label, df, cols in [
        ("seeds.csv", seeds, {"Season", "TeamID", seed_col}),
        ("slots.csv", slots, {"Season", "Slot", "StrongSeed", "WeakSeed"}),
        ("team_features.csv", feats, {"Season", "TeamID"}),
        ("team_id_map.csv", team_map, {"TeamID", "TeamName"}),
    ]:
        missing = sorted(cols - set(df.columns))
        ok = not missing
        _print(f"[{'PASS' if ok else 'FAIL'}] schema:{label} missing={missing}")
        failed = failed or not ok

    seeds_s = seeds[seeds["Season"] == args.season].copy()
    slots_s = slots[slots["Season"] == args.season].copy()
    feats_s = feats[feats["Season"] == args.season].copy()

    _print("")
    _print(f"counts.seeds_rows={len(seeds_s)}")
    _print(f"counts.seeds_unique_teams={seeds_s['TeamID'].nunique()}")
    _print(f"counts.team_features_rows={len(feats_s)}")
    _print(f"counts.team_features_unique_teams={feats_s['TeamID'].nunique()}")
    _print(f"counts.slots_rows={len(slots_s)}")
    _print(f"counts.team_id_map_rows={len(team_map)}")

    seed_codes = seeds_s[seed_col].astype(str).tolist()
    has_playin_seeds = any(code.endswith("a") or code.endswith("b") for code in seed_codes)
    slots_seed_refs = []
    for col in ["StrongSeed", "WeakSeed"]:
        vals = slots_s[col].astype(str)
        slots_seed_refs.extend(vals[vals.str.match(SEED_CODE_RE)].tolist())
    has_playin_slots = any(ref.endswith("a") or ref.endswith("b") for ref in slots_seed_refs)
    first_four_supported = bool(has_playin_seeds and has_playin_slots)

    seed_map = dict(zip(seeds_s[seed_col].astype(str), seeds_s["TeamID"]))
    r1_mask = slots_s["StrongSeed"].astype(str).str.match(SEED_CODE_RE) & slots_s["WeakSeed"].astype(str).str.match(SEED_CODE_RE)
    r1 = slots_s.loc[r1_mask, ["Slot", "StrongSeed", "WeakSeed"]].copy()
    r1["Resolvable"] = r1["StrongSeed"].isin(seed_map) & r1["WeakSeed"].isin(seed_map)
    r1_resolved = r1[r1["Resolvable"]].copy()
    r1_resolved["SeedPair"] = r1_resolved.apply(
        lambda row: f"{min(_seed_num(row['StrongSeed']), _seed_num(row['WeakSeed']))} vs {max(_seed_num(row['StrongSeed']), _seed_num(row['WeakSeed']))}",
        axis=1,
    )
    pair_types = sorted(r1_resolved["SeedPair"].dropna().unique().tolist()) if not r1_resolved.empty else []

    _print(f"round1.total_seed_seed_slots={len(r1)}")
    _print(f"round1.resolved_games={len(r1_resolved)}")
    _print(f"round1.seed_pair_types={pair_types}")
    _print(f"first_four_supported={first_four_supported}")

    seeded_teams = set(seeds_s["TeamID"].astype(int).tolist())
    featured_teams = set(feats_s["TeamID"].astype(int).tolist())
    missing_feats = sorted(seeded_teams - featured_teams)
    extra_feats = sorted(featured_teams - seeded_teams)

    cov_ok = len(missing_feats) == 0
    _print(f"[{'PASS' if cov_ok else 'FAIL'}] coverage.seeded_teams_in_features missing_count={len(missing_feats)}")
    if missing_feats:
        _print(f"coverage.missing_team_ids_sample={missing_feats[:10]}")
    _print(f"coverage.extra_feature_team_ids_count={len(extra_feats)}")

    if len(r1_resolved) < 32:
        _print("[FAIL] round1 resolved games < 32")
        failed = True
    else:
        _print("[PASS] round1 resolved games >= 32")

    if not cov_ok:
        failed = True

    _print("")
    _print(f"RESULT: {'FAIL' if failed else 'PASS'}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
