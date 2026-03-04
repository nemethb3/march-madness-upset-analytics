"""Validate season bundle and Upset Alerts state invariants."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "app") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "app"))

from app.components.data_registry import BundleMissingError, list_available_seasons, load_season_bundle
from app.components.io import build_round1_matchups_from_bracket, score_matchups_df
from src.inference_utils import build_season_context_from_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate app season state and alerts schema.")
    parser.add_argument("--season", type=int, default=2026, help="Season to validate strictly (default: 2026).")
    parser.add_argument("--check_secondary", action="store_true", help="Also check one additional season if present.")
    return parser.parse_args()


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def _pass(msg: str) -> None:
    print(f"[PASS] {msg}")


def validate_season(season: int, strict: bool) -> None:
    print(f"\n=== Validate Season {season} ===")
    try:
        bundle = load_season_bundle(season)
    except BundleMissingError as exc:
        _fail(f"Bundle load failed for season {season}: {exc}")

    seeds = bundle.seeds.copy()
    slots = bundle.slots.copy()
    feats = bundle.team_features.copy()
    teams = bundle.team_id_map[["TeamID", "TeamName"]].copy()

    unique_seed_teams = int(seeds["TeamID"].nunique())
    if strict and unique_seed_teams < 64:
        _fail(f"seed unique teams < 64 ({unique_seed_teams})")
    elif unique_seed_teams < 64:
        _warn(f"seed unique teams < 64 ({unique_seed_teams}); treated as partial bundle")
    else:
        _pass(f"seed unique teams >= 64 ({unique_seed_teams})")

    round1 = build_round1_matchups_from_bracket(season, seeds, slots, teams)
    round1_n = len(round1)
    if strict and round1_n < 32:
        _fail(f"round1 resolved games < 32 ({round1_n})")
    elif round1_n < 32:
        _warn(f"round1 resolved games < 32 ({round1_n}); partial season")
    else:
        _pass(f"round1 resolved games >= 32 ({round1_n})")

    season_ctx = build_season_context_from_frames(season, seeds, feats, teams)
    ctx = {
        "season": season,
        "seeds_df": seeds,
        "slots_df": slots,
        "team_features_df": feats,
        "teams_df": teams,
        "model": None,
        "required_features": [],
        "season_ctx": season_ctx,
    }
    scored = score_matchups_df(round1, ctx, top_k=5)
    scored = scored[scored["Error"] == ""].copy() if "Error" in scored.columns else scored.copy()

    required_cols = {
        "Season",
        "Favorite",
        "Underdog",
        "FavoriteSeed",
        "UnderdogSeed",
        "UpsetProb",
        "Reasons",
    }
    missing = sorted(required_cols - set(scored.columns))
    if missing:
        _fail(f"alerts dataframe missing columns: {missing}")
    _pass("alerts dataframe has required columns")

    scored["FavoriteSeed"] = pd.to_numeric(scored["FavoriteSeed"], errors="coerce")
    scored["UnderdogSeed"] = pd.to_numeric(scored["UnderdogSeed"], errors="coerce")
    scored["SeedPair"] = (
        scored[["FavoriteSeed", "UnderdogSeed"]].min(axis=1).astype("Int64").astype(str)
        + " vs "
        + scored[["FavoriteSeed", "UnderdogSeed"]].max(axis=1).astype("Int64").astype(str)
    )
    seed_pair_options = sorted(scored["SeedPair"].dropna().unique().tolist())
    if strict and len(seed_pair_options) != 8:
        _fail(f"seed pair options count != 8 ({len(seed_pair_options)}): {seed_pair_options}")
    elif len(seed_pair_options) < 4:
        _fail(f"seed pair options unexpectedly low ({len(seed_pair_options)}): {seed_pair_options}")
    else:
        _pass(f"seed pair options count OK ({len(seed_pair_options)}): {seed_pair_options}")

    if "Reasons" in scored.columns:
        reason_sets = scored["Reasons"].apply(lambda x: tuple(x) if isinstance(x, list) else (str(x),))
        unique_reason_sets = int(reason_sets.nunique())
        if strict and unique_reason_sets < 5:
            _fail(f"reasons appear too repetitive (unique sets={unique_reason_sets})")
        _pass(f"reason set diversity OK (unique sets={unique_reason_sets})")


def main() -> None:
    args = parse_args()
    seasons = list_available_seasons()
    if not seasons:
        _fail("No seasons found under data/app")

    validate_season(args.season, strict=True)
    if args.check_secondary:
        others = [s for s in seasons if s != args.season]
        if others:
            validate_season(others[-1], strict=False)

    print("\nValidation complete.")


if __name__ == "__main__":
    main()
