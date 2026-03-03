"""Full-tournament Monte Carlo simulation with matchup-specific probabilities."""

from __future__ import annotations

import argparse
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference_utils import (
    apply_temperature_scaling,
    build_feature_row,
    get_team_names,
    infer_required_features,
    load_model,
    load_season_context,
    map_pair_probs,
    predict_team1_win_prob,
)

SEED_CODE_PATTERN = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")
ADV_STAGES = ["R32", "S16", "E8", "F4", "TitleGame", "Champion"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Simulate tournament bracket via Monte Carlo.")
    parser.add_argument("--season", type=int, required=True, help="Tournament season to simulate.")
    parser.add_argument("--n_sims", type=int, default=50000, help="Number of Monte Carlo simulations.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.85,
        help="Probability temperature scaling for sampling (1.0 means no adjustment).",
    )
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/reports"), help="Output directory.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root.")
    return parser.parse_args()


def _load_slots_and_seeds(data_dir: Path, season: int) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load season slots and seed-to-team mapping."""
    slots_path = data_dir / "raw" / "MNCAATourneySlots.csv"
    seeds_path = data_dir / "raw" / "MNCAATourneySeeds.csv"
    for p in [slots_path, seeds_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    slots = pd.read_csv(slots_path)
    seeds = pd.read_csv(seeds_path)
    slots_s = slots[slots["Season"] == season][["Slot", "StrongSeed", "WeakSeed"]].copy()
    seeds_s = seeds[seeds["Season"] == season][["Seed", "TeamID"]].copy()
    seed_to_team = dict(zip(seeds_s["Seed"], seeds_s["TeamID"]))
    if slots_s.empty:
        raise ValueError(f"No slots found for season {season}.")
    if not seed_to_team:
        raise ValueError(f"No seeds found for season {season}.")
    return slots_s, seed_to_team


def _build_slot_order(slots_df: pd.DataFrame, seed_to_team: dict[str, int]) -> list[str]:
    """Topologically order slots so dependencies are always resolved first."""
    remaining = slots_df.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    available_tokens = set(seed_to_team.keys())
    ordered: list[str] = []

    while remaining:
        progress = False
        for slot in list(remaining.keys()):
            strong = str(remaining[slot]["StrongSeed"])
            weak = str(remaining[slot]["WeakSeed"])
            if strong in available_tokens and weak in available_tokens:
                ordered.append(slot)
                available_tokens.add(slot)
                del remaining[slot]
                progress = True
        if not progress:
            unresolved = ", ".join(sorted(remaining.keys())[:10])
            raise ValueError(f"Could not resolve slot dependency order. Unresolved sample: {unresolved}")
    return ordered


def _compute_slot_depths(slots_df: pd.DataFrame, ordered_slots: list[str], seed_to_team: dict[str, int]) -> dict[str, int]:
    """Compute dependency depth for each slot (seed-level games depth=1)."""
    slots_map = slots_df.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    depth: dict[str, int] = {}
    for slot in ordered_slots:
        strong = str(slots_map[slot]["StrongSeed"])
        weak = str(slots_map[slot]["WeakSeed"])
        strong_depth = 0 if strong in seed_to_team else depth[strong]
        weak_depth = 0 if weak in seed_to_team else depth[weak]
        depth[slot] = 1 + max(strong_depth, weak_depth)
    return depth


def _depth_to_roundapprox(depth_map: dict[str, int]) -> dict[str, str]:
    """Map slot depth to round labels; falls back gracefully when depth count differs."""
    unique_depths = sorted(set(depth_map.values()))
    if len(unique_depths) >= len(ADV_STAGES):
        selected = unique_depths[-len(ADV_STAGES) :]
        labels = ADV_STAGES
    else:
        selected = unique_depths
        labels = ADV_STAGES[: len(selected)]
    depth_to_label = {d: labels[i] for i, d in enumerate(selected)}
    out: dict[str, str] = {}
    for slot, dep in depth_map.items():
        out[slot] = depth_to_label.get(dep, f"PreRound{dep}")
    return out


def _is_round1_slot(strong_seed: str, weak_seed: str) -> bool:
    """Identify Round 1 games where both participants are explicit seeds."""
    return bool(SEED_CODE_PATTERN.match(strong_seed) and SEED_CODE_PATTERN.match(weak_seed))


def _precompute_pair_probabilities(
    model: Any,
    required_features: list[str],
    ctx: Any,
    team_ids: list[int],
) -> dict[tuple[int, int], dict[str, float]]:
    """Precompute probabilities for all seeded team pairings for fast simulation."""
    rows: list[dict[str, Any]] = []
    keys: list[tuple[int, int]] = []
    for a, b in itertools.combinations(sorted(team_ids), 2):
        feat_row, err = build_feature_row(a, b, ctx)
        if err is not None:
            continue
        if any((f not in feat_row) or pd.isna(feat_row[f]) for f in required_features):
            continue
        rows.append({f: feat_row[f] for f in required_features})
        keys.append((a, b))

    if not rows:
        return {}

    x_df = pd.DataFrame(rows, columns=required_features)
    p_team1 = predict_team1_win_prob(model, x_df)

    cache: dict[tuple[int, int], dict[str, float]] = {}
    for i, (a, b) in enumerate(keys):
        p1 = float(p_team1[i])
        p2 = 1.0 - p1
        seed_a = ctx.seed_lookup.get(a)
        seed_b = ctx.seed_lookup.get(b)
        if seed_a is None or seed_b is None:
            upset_prob = np.nan
        elif seed_a > seed_b:
            upset_prob = p1
        elif seed_b > seed_a:
            upset_prob = p2
        else:
            upset_prob = np.nan
        cache[(a, b)] = {"p_team1": p1, "upset_prob": upset_prob}
    return cache


def main() -> None:
    """Run Monte Carlo tournament simulation and write report artifacts."""
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    slots_df, seed_to_team = _load_slots_and_seeds(args.data_dir, args.season)
    ordered_slots = _build_slot_order(slots_df, seed_to_team)
    depth_map = _compute_slot_depths(slots_df, ordered_slots, seed_to_team)
    round_map = _depth_to_roundapprox(depth_map)
    slots_map = slots_df.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    model = load_model(args.model_path)
    required_features = infer_required_features(model)
    ctx = load_season_context(args.data_dir, args.season)
    seeded_teams = sorted(set(seed_to_team.values()))

    pair_cache = _precompute_pair_probabilities(model, required_features, ctx, seeded_teams)
    rng = np.random.default_rng(args.seed)

    adv_counts: dict[int, dict[str, int]] = {tid: {stage: 0 for stage in ADV_STAGES} for tid in seeded_teams}
    matchup_counts: dict[tuple[str, int, int], int] = defaultdict(int)
    matchup_sum_p_teamA: dict[tuple[str, int, int], float] = defaultdict(float)
    matchup_sum_upset: dict[tuple[str, int, int], float] = defaultdict(float)

    for _ in range(args.n_sims):
        winners_by_slot: dict[str, int] = {}

        for slot in ordered_slots:
            strong = str(slots_map[slot]["StrongSeed"])
            weak = str(slots_map[slot]["WeakSeed"])
            team_a = seed_to_team.get(strong, winners_by_slot.get(strong))
            team_b = seed_to_team.get(weak, winners_by_slot.get(weak))
            if team_a is None or team_b is None:
                raise ValueError(f"Failed resolving teams for slot {slot}: {strong} vs {weak}")

            team1, team2 = (team_a, team_b) if team_a < team_b else (team_b, team_a)
            pair = (team1, team2)
            if pair in pair_cache:
                p_team1 = pair_cache[pair]["p_team1"]
            else:
                p_team1 = 0.5
            p_a_raw, p_b_raw = map_pair_probs(team_a, team_b, p_team1)
            p_a_adj = apply_temperature_scaling(p_a_raw, args.temperature)

            winner = team_a if rng.random() < p_a_adj else team_b
            winners_by_slot[slot] = winner

            round_label = round_map.get(slot, "PreRound")
            if round_label in ADV_STAGES:
                adv_counts[winner][round_label] += 1

            # Track encounter stats in canonical team-ID order to avoid duplicates.
            can_a, can_b = (team_a, team_b) if team_a < team_b else (team_b, team_a)
            p_can_a, p_can_b = map_pair_probs(can_a, can_b, p_team1)
            seed_can_a = ctx.seed_lookup.get(can_a)
            seed_can_b = ctx.seed_lookup.get(can_b)
            if seed_can_a is None or seed_can_b is None:
                upset_prob = np.nan
            elif seed_can_a > seed_can_b:
                upset_prob = p_can_a
            elif seed_can_b > seed_can_a:
                upset_prob = p_can_b
            else:
                upset_prob = np.nan

            key = (round_label, can_a, can_b)
            matchup_counts[key] += 1
            matchup_sum_p_teamA[key] += p_can_a
            matchup_sum_upset[key] += upset_prob if not np.isnan(upset_prob) else 0.0

    adv_rows: list[dict[str, Any]] = []
    for tid in seeded_teams:
        name = ctx.team_id_to_name.get(tid, f"TeamID {tid}")
        seed_num = ctx.seed_lookup.get(tid, np.nan)
        row = {"TeamID": tid, "TeamName": name, "SeedNum": seed_num}
        for stage in ADV_STAGES:
            row[f"P_{stage}"] = adv_counts[tid][stage] / args.n_sims
        adv_rows.append(row)

    adv_df = pd.DataFrame(adv_rows).sort_values("P_Champion", ascending=False).reset_index(drop=True)
    adv_path = args.out_dir / "advancement_probabilities.csv"
    adv_df.to_csv(adv_path, index=False)

    top_title_path = args.out_dir / "top_title_odds.csv"
    adv_df.head(25)[["TeamID", "TeamName", "SeedNum", "P_Champion"]].to_csv(top_title_path, index=False)

    matchup_rows: list[dict[str, Any]] = []
    for (round_approx, team_a, team_b), count in matchup_counts.items():
        team_a_name, team_b_name = get_team_names(team_a, team_b, ctx)
        avg_p_team_a = matchup_sum_p_teamA[(round_approx, team_a, team_b)] / count
        avg_upset = matchup_sum_upset[(round_approx, team_a, team_b)] / count
        matchup_rows.append(
            {
                "RoundApprox": round_approx,
                "Round": round_approx,
                "TeamAName": team_a_name,
                "TeamBName": team_b_name,
                "Frequency": count,
                "AvgWinProbTeamA": avg_p_team_a,
                "AvgUpsetProb": avg_upset,
            }
        )
    matchups_df = pd.DataFrame(matchup_rows).sort_values("Frequency", ascending=False).reset_index(drop=True)
    matchups_path = args.out_dir / "simulated_matchups.csv"
    matchups_df.to_csv(matchups_path, index=False)

    print(f"Wrote: {adv_path}")
    print(f"Wrote: {top_title_path}")
    print(f"Wrote: {matchups_path}")


if __name__ == "__main__":
    main()
