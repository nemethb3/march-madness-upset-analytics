"""Data loading, model loading, scoring, and simulation helpers for Streamlit app."""

from __future__ import annotations

import hashlib
import itertools
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from app.components.text import simulation_effort_to_n_sims
from src.build_round1_from_slots import build_round1_matchups_from_frames
from src.inference_utils import (
    apply_temperature_scaling,
    build_factor_strings,
    build_feature_row,
    build_season_context_from_frames,
    infer_required_features,
    load_model,
    map_pair_probs,
    predict_team1_win_prob,
)

DEMO_DIR = Path("app/demo_data")
SEED_CODE_PATTERN = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")
ADV_STAGES = ["R32", "S16", "E8", "F4", "TitleGame", "Champion"]


@st.cache_data
def read_csv_bytes(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Read uploaded CSV bytes into dataframe (cached)."""
    return pd.read_csv(pd.io.common.BytesIO(file_bytes))


@st.cache_data
def load_demo_csv(path_str: str) -> pd.DataFrame:
    """Load demo CSV file (cached)."""
    return pd.read_csv(path_str)


@st.cache_data
def load_local_csv(path_str: str) -> pd.DataFrame:
    """Load local CSV by path (cached)."""
    return pd.read_csv(path_str)


@st.cache_resource
def load_cached_model(path_str: str):
    """Load model from disk with resource caching."""
    return load_model(Path(path_str))


def discover_local_models() -> dict[str, Path]:
    """Discover available model files and return label->path map."""
    models_dir = Path("outputs/models")
    out: dict[str, Path] = {}
    calibrated = models_dir / "logistic_regression_calibrated.joblib"
    pipeline = models_dir / "logistic_regression_pipeline.joblib"
    if calibrated.exists():
        out["Logistic (Calibrated)"] = calibrated
    if pipeline.exists():
        out["Logistic (Uncalibrated)"] = pipeline
    return out


def model_hash_from_path(path: Path | None) -> str:
    """Build model hash for cache keys."""
    if path is None or not path.exists():
        return "heuristic"
    stat = path.stat()
    base = f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def _seed_num_from_code(seed_code: str) -> int:
    """Parse numeric seed from seed code."""
    m = re.search(r"\d+", str(seed_code))
    if m is None:
        raise ValueError(f"Could not parse seed number from {seed_code}")
    return int(m.group(0))


def _heuristic_p_team_a(seed_a: float, seed_b: float) -> float:
    """Fallback seed-based probability when no model is available."""
    if pd.isna(seed_a) or pd.isna(seed_b):
        return 0.5
    x = (float(seed_b) - float(seed_a)) / 2.0
    p = 1.0 / (1.0 + np.exp(-x))
    return float(np.clip(p, 0.02, 0.98))


def render_sidebar() -> dict[str, Any]:
    """Render global sidebar and return app context dictionary."""
    st.sidebar.header("Data Mode")
    up_seeds = st.sidebar.file_uploader("Upload MNCAATourneySeeds.csv", type=["csv"], key="up_seeds")
    up_slots = st.sidebar.file_uploader("Upload MNCAATourneySlots.csv", type=["csv"], key="up_slots")
    up_team_feats = st.sidebar.file_uploader(
        "Upload team_season_features.csv (optional)", type=["csv"], key="up_team_feats"
    )

    # Mode selection: if both required uploads are present use Upload Mode, otherwise Demo Mode.
    if up_seeds is not None and up_slots is not None:
        seeds_df = read_csv_bytes(up_seeds.name, up_seeds.getvalue())
        slots_df = read_csv_bytes(up_slots.name, up_slots.getvalue())
        mode = "Upload Mode"
    else:
        seeds_df = load_demo_csv(str(DEMO_DIR / "demo_seeds.csv"))
        slots_df = load_demo_csv(str(DEMO_DIR / "demo_slots.csv"))
        mode = "Demo Mode"

    # Team features: upload > local processed > demo.
    team_feats_msg = ""
    if up_team_feats is not None:
        team_features_df = read_csv_bytes(up_team_feats.name, up_team_feats.getvalue())
    else:
        local_feats = Path("data/processed/team_season_features.csv")
        if local_feats.exists():
            team_features_df = load_local_csv(str(local_feats))
        else:
            team_features_df = load_demo_csv(str(DEMO_DIR / "demo_team_features.csv"))
            if mode == "Upload Mode":
                team_feats_msg = "To run predictions, upload team_season_features.csv from your local pipeline output."

    if team_feats_msg:
        st.sidebar.info(team_feats_msg)

    # Optional teams table for names
    teams_df = None
    local_teams = Path("data/raw/MTeams.csv")
    if local_teams.exists():
        teams_df = load_local_csv(str(local_teams))[["TeamID", "TeamName"]]
    elif "TeamName" in team_features_df.columns:
        teams_df = team_features_df[["TeamID", "TeamName"]].dropna().drop_duplicates()

    st.sidebar.divider()
    st.sidebar.header("Controls")
    seasons = sorted(pd.Series(seeds_df["Season"]).dropna().astype(int).unique().tolist())
    default_season = seasons[-1]
    season = st.sidebar.selectbox("Season", options=seasons, index=len(seasons) - 1)

    model_options = discover_local_models()
    option_labels = list(model_options.keys()) + ["Heuristic (No model file)"]
    default_idx = 0 if option_labels else None
    selected_model_label = st.sidebar.selectbox("Model", options=option_labels, index=default_idx)
    model_path = model_options.get(selected_model_label)
    model = load_cached_model(str(model_path)) if model_path is not None else None
    required_features = infer_required_features(model) if model is not None else []

    randomness = st.sidebar.slider("Randomness", min_value=0.75, max_value=1.0, value=0.85, step=0.01)
    upset_threshold = st.sidebar.slider(
        "Upset threshold",
        min_value=0.10,
        max_value=0.60,
        value=0.30,
        step=0.01,
        help="Higher means fewer underdog picks.",
    )
    risk_tolerance = st.sidebar.slider("Risk tolerance", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    effort_label = st.sidebar.selectbox("Simulation effort", options=["Fast", "Balanced", "Thorough"], index=1)
    n_sims = simulation_effort_to_n_sims(effort_label)

    context = build_season_context_from_frames(
        season=season,
        seeds_df=seeds_df,
        team_features_df=team_features_df,
        teams_df=teams_df,
    )

    return {
        "mode": mode,
        "seeds_df": seeds_df,
        "slots_df": slots_df,
        "team_features_df": team_features_df,
        "teams_df": teams_df,
        "season": season,
        "model": model,
        "model_path": model_path,
        "model_hash": model_hash_from_path(model_path),
        "required_features": required_features,
        "randomness": randomness,
        "upset_threshold": upset_threshold,
        "risk_tolerance": risk_tolerance,
        "n_sims": n_sims,
        "context": context,
    }


def build_round1_df(ctx: dict[str, Any]) -> pd.DataFrame:
    """Build round1 matchups dataframe using in-memory data sources."""
    return build_round1_matchups_from_frames(
        slots_df=ctx["slots_df"],
        seeds_df=ctx["seeds_df"],
        season=ctx["season"],
        teams_df=ctx["teams_df"],
    )


def score_round1_matchups(round1_df: pd.DataFrame, ctx: dict[str, Any], top_k: int = 3) -> pd.DataFrame:
    """Score round1 matchups with model probabilities and explanation factors."""
    model = ctx["model"]
    required_features = ctx["required_features"]
    season_ctx = ctx["context"]

    records: list[dict[str, Any]] = []
    scored_indices: list[int] = []
    scored_features: list[dict[str, Any]] = []

    for _, row in round1_df.iterrows():
        rec = {
            "Season": ctx["season"],
            "Slot": row["Slot"],
            "TeamAID": row["TeamAID"],
            "TeamBID": row["TeamBID"],
            "TeamAName": row["TeamAName"],
            "TeamBName": row["TeamBName"],
            "TeamASeedNum": row["TeamASeedNum"],
            "TeamBSeedNum": row["TeamBSeedNum"],
            "P_TeamAWin": np.nan,
            "P_TeamBWin": np.nan,
            "WorseSeedTeam": np.nan,
            "UpsetProb": np.nan,
            "RecommendedPick": np.nan,
            "Confidence": np.nan,
            "Error": "",
        }
        if pd.isna(row["TeamAID"]) or pd.isna(row["TeamBID"]):
            rec["Error"] = "missing TeamID from seed mapping"
            records.append(rec)
            continue

        team_a = int(row["TeamAID"])
        team_b = int(row["TeamBID"])
        team1, team2 = (team_a, team_b) if team_a < team_b else (team_b, team_a)

        if model is None:
            p_a = _heuristic_p_team_a(row["TeamASeedNum"], row["TeamBSeedNum"])
            p_b = 1.0 - p_a
            factors = [
                f"Seed gap: {int(row['TeamASeedNum']) - int(row['TeamBSeedNum']):+d}",
                "Model: heuristic",
                "No model file detected",
            ]
            rec["_factors"] = factors
            rec["_pA"] = p_a
            rec["_pB"] = p_b
            records.append(rec)
            continue

        full_row, err = build_feature_row(team1, team2, season_ctx)
        if err is not None:
            rec["Error"] = err
            records.append(rec)
            continue
        missing = [f for f in required_features if f not in full_row or pd.isna(full_row[f])]
        if missing:
            rec["Error"] = f"missing required model features: {missing[:6]}"
            records.append(rec)
            continue

        scored_features.append({f: full_row[f] for f in required_features})
        scored_indices.append(len(records))
        rec["_team_a"] = team_a
        rec["_team_b"] = team_b
        records.append(rec)

    if model is not None and scored_features:
        x_df = pd.DataFrame(scored_features, columns=required_features)
        p_team1 = predict_team1_win_prob(model, x_df)
        factor_rows = build_factor_strings(model, x_df, required_features, top_k=top_k)
        for i, rec_idx in enumerate(scored_indices):
            rec = records[rec_idx]
            p_a, p_b = map_pair_probs(int(rec["_team_a"]), int(rec["_team_b"]), float(p_team1[i]))
            rec["_pA"] = p_a
            rec["_pB"] = p_b
            rec["_factors"] = factor_rows[i]

    for rec in records:
        if rec.get("Error"):
            for k in range(top_k):
                rec[f"Factor{k+1}"] = "N/A"
            continue
        p_a = float(rec["_pA"])
        p_b = float(rec["_pB"])
        seed_a = int(rec["TeamASeedNum"])
        seed_b = int(rec["TeamBSeedNum"])
        if seed_a > seed_b:
            worse = rec["TeamAName"]
            upset = p_a
        elif seed_b > seed_a:
            worse = rec["TeamBName"]
            upset = p_b
        else:
            worse = "TieSeed"
            upset = np.nan

        rec["P_TeamAWin"] = p_a
        rec["P_TeamBWin"] = p_b
        rec["WorseSeedTeam"] = worse
        rec["UpsetProb"] = upset
        rec["RecommendedPick"] = rec["TeamAName"] if p_a >= p_b else rec["TeamBName"]
        rec["Confidence"] = max(p_a, p_b)
        rec["SeedPair"] = f"{min(seed_a, seed_b)}-{max(seed_a, seed_b)}"
        for k in range(top_k):
            rec[f"Factor{k+1}"] = rec["_factors"][k] if k < len(rec["_factors"]) else "N/A"

    out = pd.DataFrame(records)
    keep_cols = [
        "Season",
        "Slot",
        "TeamAID",
        "TeamBID",
        "TeamAName",
        "TeamBName",
        "TeamASeedNum",
        "TeamBSeedNum",
        "SeedPair",
        "P_TeamAWin",
        "P_TeamBWin",
        "WorseSeedTeam",
        "UpsetProb",
        "RecommendedPick",
        "Confidence",
        "Factor1",
        "Factor2",
        "Factor3",
        "Error",
    ]
    for c in keep_cols:
        if c not in out.columns:
            out[c] = np.nan if c != "Error" else ""
    return out[keep_cols].sort_values("Slot").reset_index(drop=True)


def _load_seed_to_team(seeds_df: pd.DataFrame, season: int) -> dict[str, int]:
    """Build seed code to TeamID map."""
    seed_col = "Seed" if "Seed" in seeds_df.columns else "SeedStr"
    s = seeds_df[seeds_df["Season"] == season][[seed_col, "TeamID"]]
    return {str(r[seed_col]): int(r["TeamID"]) for _, r in s.iterrows()}


def _slot_order(slots_df: pd.DataFrame, seed_to_team: dict[str, int], season: int) -> tuple[list[str], dict[str, dict[str, str]]]:
    """Return topological slot order and slot map."""
    slots_s = slots_df[slots_df["Season"] == season][["Slot", "StrongSeed", "WeakSeed"]].copy()
    remaining = slots_s.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    available = set(seed_to_team.keys())
    ordered: list[str] = []
    while remaining:
        progressed = False
        for slot in list(remaining.keys()):
            strong = str(remaining[slot]["StrongSeed"])
            weak = str(remaining[slot]["WeakSeed"])
            if strong in available and weak in available:
                ordered.append(slot)
                available.add(slot)
                del remaining[slot]
                progressed = True
        if not progressed:
            raise ValueError("Could not resolve slot order from uploaded slots.")
    return ordered, slots_s.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")


def _depth_round_map(ordered: list[str], slot_map: dict[str, dict[str, str]], seed_to_team: dict[str, int]) -> dict[str, str]:
    """Map each slot to round label based on dependency depth."""
    depth: dict[str, int] = {}
    for slot in ordered:
        strong = str(slot_map[slot]["StrongSeed"])
        weak = str(slot_map[slot]["WeakSeed"])
        ds = 0 if strong in seed_to_team else depth[strong]
        dw = 0 if weak in seed_to_team else depth[weak]
        depth[slot] = 1 + max(ds, dw)
    uniq = sorted(set(depth.values()))
    stages = ADV_STAGES[-len(uniq) :] if len(uniq) <= len(ADV_STAGES) else ADV_STAGES
    mapping = {d: stages[i] for i, d in enumerate(uniq[-len(stages) :])}
    return {slot: mapping.get(d, f"Round{d}") for slot, d in depth.items()}


def _pair_upset_prob(team_a: int, team_b: int, p_a: float, ctx: Any) -> float:
    """Compute upset probability for a matchup using team-order probability."""
    seed_a = ctx.seed_lookup.get(team_a)
    seed_b = ctx.seed_lookup.get(team_b)
    if seed_a is None or seed_b is None:
        return np.nan
    if seed_a > seed_b:
        return p_a
    if seed_b > seed_a:
        return 1.0 - p_a
    return np.nan


@st.cache_data
def run_simulation_cached(
    season: int,
    n_sims: int,
    randomness: float,
    model_hash: str,
    seeds_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    teams_df: pd.DataFrame | None,
    model_path_str: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run path-dependent Monte Carlo simulation with cacheable inputs."""
    model = load_cached_model(model_path_str) if model_path_str else None
    required_features = infer_required_features(model) if model is not None else []
    season_ctx = build_season_context_from_frames(season, seeds_df, team_features_df, teams_df=teams_df)

    seed_to_team = _load_seed_to_team(seeds_df, season)
    ordered, slot_map = _slot_order(slots_df, seed_to_team, season)
    round_map = _depth_round_map(ordered, slot_map, seed_to_team)
    seeded_teams = sorted(set(seed_to_team.values()))

    pair_cache: dict[tuple[int, int], float] = {}
    if model is not None and required_features:
        rows, keys = [], []
        for a, b in itertools.combinations(seeded_teams, 2):
            frow, err = build_feature_row(a, b, season_ctx)
            if err is not None:
                continue
            if any((f not in frow) or pd.isna(frow[f]) for f in required_features):
                continue
            rows.append({f: frow[f] for f in required_features})
            keys.append((a, b))
        if rows:
            x = pd.DataFrame(rows, columns=required_features)
            p1 = predict_team1_win_prob(model, x)
            for i, key in enumerate(keys):
                pair_cache[key] = float(p1[i])

    rng = np.random.default_rng(42)
    adv_counts = {tid: {s: 0 for s in ADV_STAGES} for tid in seeded_teams}
    matchup_counts = defaultdict(int)
    matchup_p_sum = defaultdict(float)
    matchup_upset_sum = defaultdict(float)

    for _ in range(n_sims):
        winners_by_slot: dict[str, int] = {}
        for slot in ordered:
            strong = str(slot_map[slot]["StrongSeed"])
            weak = str(slot_map[slot]["WeakSeed"])
            team_a = seed_to_team.get(strong, winners_by_slot.get(strong))
            team_b = seed_to_team.get(weak, winners_by_slot.get(weak))
            if team_a is None or team_b is None:
                continue

            t1, t2 = (team_a, team_b) if team_a < team_b else (team_b, team_a)
            if model is not None and (t1, t2) in pair_cache:
                p_team1 = pair_cache[(t1, t2)]
                p_a, _ = map_pair_probs(team_a, team_b, p_team1)
            else:
                seed_a = season_ctx.seed_lookup.get(team_a, np.nan)
                seed_b = season_ctx.seed_lookup.get(team_b, np.nan)
                p_a = _heuristic_p_team_a(seed_a, seed_b)

            p_adj = apply_temperature_scaling(p_a, randomness)
            winner = team_a if rng.random() < p_adj else team_b
            winners_by_slot[slot] = winner

            round_name = round_map[slot]
            if round_name in ADV_STAGES:
                adv_counts[winner][round_name] += 1

            can_a, can_b = (team_a, team_b) if team_a < team_b else (team_b, team_a)
            p_can_a = p_a if team_a == can_a else 1.0 - p_a
            upset = _pair_upset_prob(can_a, can_b, p_can_a, season_ctx)
            key = (round_name, can_a, can_b)
            matchup_counts[key] += 1
            matchup_p_sum[key] += p_can_a
            matchup_upset_sum[key] += upset if pd.notna(upset) else 0.0

    adv_rows = []
    for tid in seeded_teams:
        row = {
            "TeamID": tid,
            "TeamName": season_ctx.team_id_to_name.get(tid, f"Team {tid}"),
            "SeedNum": season_ctx.seed_lookup.get(tid, np.nan),
        }
        for s in ADV_STAGES:
            row[f"P_{s}"] = adv_counts[tid][s] / n_sims
        adv_rows.append(row)
    adv_df = pd.DataFrame(adv_rows).sort_values("P_Champion", ascending=False).reset_index(drop=True)

    matchup_rows = []
    for (rnd, a, b), ct in matchup_counts.items():
        matchup_rows.append(
            {
                "Round": rnd,
                "TeamAName": season_ctx.team_id_to_name.get(a, f"Team {a}"),
                "TeamBName": season_ctx.team_id_to_name.get(b, f"Team {b}"),
                "Frequency": ct,
                "AvgWinProbTeamA": matchup_p_sum[(rnd, a, b)] / ct,
                "AvgUpsetProb": matchup_upset_sum[(rnd, a, b)] / ct,
            }
        )
    matchup_df = pd.DataFrame(matchup_rows).sort_values("Frequency", ascending=False).reset_index(drop=True)
    return adv_df, matchup_df

