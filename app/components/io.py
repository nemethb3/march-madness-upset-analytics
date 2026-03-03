"""Core app I/O, scoring, bracket, and simulation helpers."""

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

from components.bootstrap import ensure_repo_root_on_path
from components.data_registry import BundleMissingError, list_available_seasons, load_season_bundle
from components.explanations import build_underdog_reasons

ensure_repo_root_on_path()

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

ADV_STAGES = ["Round 1", "Round 2", "Sweet 16", "Elite 8", "Final Four", "Championship"]
SEED_CODE_PATTERN = re.compile(r"^[WXYZ][0-9]{2}[ab]?$")


@st.cache_data
def cached_load_bundle(season: int):
    """Load season bundle with caching."""
    bundle = load_season_bundle(season)
    return bundle.seeds, bundle.slots, bundle.team_features, bundle.team_id_map


@st.cache_resource
def load_cached_model(path_str: str):
    """Load model with resource caching."""
    return load_model(Path(path_str))


def _model_candidates() -> dict[str, Path]:
    """Discover model files in priority order."""
    models_dir = Path("outputs/models")
    out: dict[str, Path] = {}
    calibrated = models_dir / "logistic_regression_calibrated.joblib"
    pipeline = models_dir / "logistic_regression_pipeline.joblib"
    if calibrated.exists():
        out["Logistic (Calibrated)"] = calibrated
    if pipeline.exists():
        out["Logistic (Base)"] = pipeline
    return out


def _n_sims_from_effort(effort: str) -> int:
    """Map effort label to simulation count."""
    mapping = {"Fast": 5000, "Balanced": 20000, "Thorough": 50000}
    return mapping[effort]


def _model_hash(path: Path | None) -> str:
    """Create model fingerprint for cache keying."""
    if path is None or not path.exists():
        return "heuristic"
    stat = path.stat()
    return hashlib.md5(f"{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}".encode("utf-8")).hexdigest()


def _heuristic_prob(seed_a: float, seed_b: float) -> float:
    """Seed-only fallback probability."""
    if pd.isna(seed_a) or pd.isna(seed_b):
        return 0.5
    x = (float(seed_b) - float(seed_a)) / 2.0
    return float(np.clip(1.0 / (1.0 + np.exp(-x)), 0.02, 0.98))


def render_sidebar() -> dict[str, Any]:
    """Render global controls and return app context."""
    st.sidebar.header("Season")
    available = list_available_seasons()
    if not available:
        st.error("No season bundles found in data/app/{season}.")
        st.stop()

    season = st.sidebar.selectbox("Season", options=available, index=len(available) - 1)

    mode = "Bundle Mode"
    try:
        seeds_df, slots_df, team_features_df, team_id_map = cached_load_bundle(season)
    except BundleMissingError:
        demo_season = None
        for s in sorted(available, reverse=True):
            try:
                seeds_df, slots_df, team_features_df, team_id_map = cached_load_bundle(s)
                demo_season = s
                break
            except BundleMissingError:
                continue
        if demo_season is None:
            st.error("No complete season bundles found in data/app/.")
            st.stop()
        mode = "Demo Mode"
        st.warning("Season data not available yet. App running in demo mode.")
        season = demo_season
    except Exception as exc:
        st.error(f"Bundle load failed: {exc}")
        st.stop()

    st.sidebar.divider()
    st.sidebar.header("Controls")
    model_options = _model_candidates()
    model_labels = list(model_options.keys())
    if model_labels:
        selected = st.sidebar.selectbox("Model", options=model_labels, index=0)
        model_path = model_options[selected]
        model = load_cached_model(str(model_path))
        model_loaded = True
        required_features = infer_required_features(model)
    else:
        model_path = None
        model = None
        model_loaded = False
        required_features = []

    randomness = st.sidebar.slider("Randomness", min_value=0.75, max_value=1.0, value=0.85, step=0.01)
    upset_threshold = st.sidebar.slider("Upset threshold", min_value=0.10, max_value=0.60, value=0.30, step=0.01)
    risk_tolerance = st.sidebar.slider("Risk tolerance", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    effort = st.sidebar.selectbox("Simulation effort", options=["Fast", "Balanced", "Thorough"], index=1)
    n_sims = min(_n_sims_from_effort(effort), 50000)

    teams_df = team_id_map[["TeamID", "TeamName"]].copy()
    season_ctx = build_season_context_from_frames(
        season=season,
        seeds_df=seeds_df,
        team_features_df=team_features_df,
        teams_df=teams_df,
    )

    status = {
        "model_loaded": model_loaded,
        "season_bundle_loaded": True,
        "feature_rows": len(team_features_df[team_features_df["Season"] == season]),
        "seed_rows": len(seeds_df[seeds_df["Season"] == season]),
        "slot_rows": len(slots_df[slots_df["Season"] == season]),
        "mode": mode,
        "model_file": str(model_path) if model_path else "None",
    }
    with st.sidebar.expander("Backend Status"):
        st.write(status)

    return {
        "mode": mode,
        "season": season,
        "seeds_df": seeds_df,
        "slots_df": slots_df,
        "team_features_df": team_features_df,
        "teams_df": teams_df,
        "model": model,
        "model_path": model_path,
        "model_hash": _model_hash(model_path),
        "required_features": required_features,
        "randomness": randomness,
        "upset_threshold": upset_threshold,
        "risk_tolerance": risk_tolerance,
        "n_sims": n_sims,
        "season_ctx": season_ctx,
        "backend_status": status,
    }


def build_round1_df(ctx: dict[str, Any]) -> pd.DataFrame:
    """Build round1 matchups."""
    return build_round1_matchups_from_frames(
        slots_df=ctx["slots_df"],
        seeds_df=ctx["seeds_df"],
        season=ctx["season"],
        teams_df=ctx["teams_df"],
    )


def _seed_lookup_for_df(seeds_df: pd.DataFrame, season: int) -> dict[int, int]:
    """TeamID -> SeedNum mapping."""
    temp = seeds_df[seeds_df["Season"] == season].copy()
    if "SeedNum" not in temp.columns:
        col = "Seed" if "Seed" in temp.columns else "SeedStr"
        temp["SeedNum"] = temp[col].astype(str).str.extract(r"(\d+)").astype(float)
    temp = temp.dropna(subset=["SeedNum"])
    return {int(r["TeamID"]): int(r["SeedNum"]) for _, r in temp.iterrows()}


def score_matchups_df(matchups_df: pd.DataFrame, ctx: dict[str, Any], top_k: int = 3) -> pd.DataFrame:
    """Score arbitrary matchup dataframe with TeamAID/TeamBID columns."""
    model = ctx["model"]
    required_features = ctx["required_features"]
    season_ctx = ctx["season_ctx"]
    seed_lookup = _seed_lookup_for_df(ctx["seeds_df"], ctx["season"])

    records: list[dict[str, Any]] = []
    scored_idx: list[int] = []
    scored_feats: list[dict[str, Any]] = []

    for _, row in matchups_df.iterrows():
        rec = row.to_dict()
        rec.setdefault("Error", "")
        team_a = row.get("TeamAID")
        team_b = row.get("TeamBID")
        if pd.isna(team_a) or pd.isna(team_b):
            rec["Error"] = "Unresolved matchup participants"
            records.append(rec)
            continue

        team_a = int(team_a)
        team_b = int(team_b)
        team1, team2 = (team_a, team_b) if team_a < team_b else (team_b, team_a)
        rec["Team1ID"] = team1
        rec["Team2ID"] = team2
        rec["Team1Name"] = season_ctx.team_id_to_name.get(team1, f"Team {team1}")
        rec["Team2Name"] = season_ctx.team_id_to_name.get(team2, f"Team {team2}")
        rec["TeamASeedNum"] = seed_lookup.get(team_a, rec.get("TeamASeedNum", np.nan))
        rec["TeamBSeedNum"] = seed_lookup.get(team_b, rec.get("TeamBSeedNum", np.nan))

        if model is None:
            p_a = _heuristic_prob(rec["TeamASeedNum"], rec["TeamBSeedNum"])
            rec["_pA"] = p_a
            rec["_factors"] = ["Seed advantage", "Recent form edge", "Bracket volatility"]
            records.append(rec)
            continue

        frow, err = build_feature_row(team1, team2, season_ctx)
        if err is not None:
            rec["Error"] = err
            records.append(rec)
            continue
        missing = [f for f in required_features if f not in frow or pd.isna(frow[f])]
        if missing:
            rec["Error"] = f"missing required model features: {missing[:6]}"
            records.append(rec)
            continue

        rec["_team_a"] = team_a
        rec["_team_b"] = team_b
        rec["_full_features"] = frow
        scored_feats.append({f: frow[f] for f in required_features})
        scored_idx.append(len(records))
        records.append(rec)

    if model is not None and scored_feats:
        x_df = pd.DataFrame(scored_feats, columns=required_features)
        p_team1 = predict_team1_win_prob(model, x_df)
        factors = build_factor_strings(model, x_df, required_features, top_k=top_k)
        for i, idx in enumerate(scored_idx):
            rec = records[idx]
            p_a, p_b = map_pair_probs(int(rec["_team_a"]), int(rec["_team_b"]), float(p_team1[i]))
            rec["_pA"] = p_a
            rec["_pB"] = p_b
            rec["_factors"] = factors[i]

    for rec in records:
        if rec.get("Error"):
            for k in range(top_k):
                rec[f"Factor{k+1}"] = "N/A"
            continue

        p_a = float(rec["_pA"])
        p_b = float(1.0 - p_a)
        seed_a = int(rec.get("TeamASeedNum", 99))
        seed_b = int(rec.get("TeamBSeedNum", 99))

        if seed_a > seed_b:
            underdog, favorite = rec.get("TeamAName"), rec.get("TeamBName")
            upset = p_a
            underdog_seed, favorite_seed = seed_a, seed_b
        elif seed_b > seed_a:
            underdog, favorite = rec.get("TeamBName"), rec.get("TeamAName")
            upset = p_b
            underdog_seed, favorite_seed = seed_b, seed_a
        else:
            underdog, favorite = rec.get("TeamAName"), rec.get("TeamBName")
            upset = np.nan
            underdog_seed, favorite_seed = seed_a, seed_b

        rec["P_TeamAWin"] = p_a
        rec["P_TeamBWin"] = p_b
        rec["UpsetProb"] = upset
        rec["Underdog"] = underdog
        rec["Favorite"] = favorite
        rec["UnderdogSeed"] = underdog_seed
        rec["FavoriteSeed"] = favorite_seed
        rec["WorseSeedTeam"] = underdog
        rec["RecommendedPick"] = rec.get("TeamAName") if p_a >= p_b else rec.get("TeamBName")
        rec["Confidence"] = max(p_a, p_b)

        if upset >= 0.30:
            rec["AlertLevel"] = "High"
        elif upset >= 0.20:
            rec["AlertLevel"] = "Medium"
        elif upset >= 0.15:
            rec["AlertLevel"] = "Watch"
        else:
            rec["AlertLevel"] = "Low"

        for k in range(top_k):
            rec[f"Factor{k+1}"] = rec["_factors"][k] if k < len(rec["_factors"]) else "N/A"

        rec["Reasons"] = build_underdog_reasons(pd.Series(rec), feature_diffs=rec.get("_full_features"))

    return pd.DataFrame(records)


def score_round1_matchups(round1_df: pd.DataFrame, ctx: dict[str, Any], top_k: int = 3) -> pd.DataFrame:
    """Score round1 matchups."""
    return score_matchups_df(round1_df.copy(), ctx, top_k=top_k)


def get_slot_structure(ctx: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, str]], dict[str, str], dict[str, str]]:
    """Return ordered slots, slot map, round map, and region map."""
    slots = ctx["slots_df"][ctx["slots_df"]["Season"] == ctx["season"]][["Slot", "StrongSeed", "WeakSeed"]].copy()
    seed_col = "Seed" if "Seed" in ctx["seeds_df"].columns else "SeedStr"
    seeds = ctx["seeds_df"][ctx["seeds_df"]["Season"] == ctx["season"]][[seed_col, "TeamID"]]
    seed_tokens = set(seeds[seed_col].astype(str).tolist())

    rem = slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    ordered: list[str] = []
    available = set(seed_tokens)
    while rem:
        progressed = False
        for slot in list(rem.keys()):
            s = str(rem[slot]["StrongSeed"])
            w = str(rem[slot]["WeakSeed"])
            if s in available and w in available:
                ordered.append(slot)
                available.add(slot)
                del rem[slot]
                progressed = True
        if not progressed:
            break
    slot_map = slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")

    depth: dict[str, int] = {}
    for slot in ordered:
        s = str(slot_map[slot]["StrongSeed"])
        w = str(slot_map[slot]["WeakSeed"])
        ds = 0 if s in seed_tokens else depth.get(s, 0)
        dw = 0 if w in seed_tokens else depth.get(w, 0)
        depth[slot] = 1 + max(ds, dw)
    uniq = sorted(set(depth.values()))
    stage_map = {d: ADV_STAGES[min(i, len(ADV_STAGES) - 1)] for i, d in enumerate(uniq)}
    round_map = {slot: stage_map.get(depth.get(slot, 1), "Round 1") for slot in slot_map}

    seed_region = {str(s): str(s)[0] for s in seed_tokens if SEED_CODE_PATTERN.match(str(s))}

    def token_region(token: str) -> str:
        if token in seed_region:
            return seed_region[token]
        if token in slot_map:
            left = token_region(str(slot_map[token]["StrongSeed"]))
            right = token_region(str(slot_map[token]["WeakSeed"]))
            return left if left == right else "National"
        return "National"

    region_map = {slot: token_region(slot) if slot in slot_map else "National" for slot in slot_map}
    return ordered, slot_map, round_map, region_map


def resolve_bracket_state(ctx: dict[str, Any], picks: dict[str, int]) -> pd.DataFrame:
    """Resolve slot participants and optional pick outcomes for all rounds."""
    ordered, slot_map, round_map, region_map = get_slot_structure(ctx)
    seed_col = "Seed" if "Seed" in ctx["seeds_df"].columns else "SeedStr"
    seeds = ctx["seeds_df"][ctx["seeds_df"]["Season"] == ctx["season"]][[seed_col, "TeamID"]].copy()
    seed_to_team = {str(r[seed_col]): int(r["TeamID"]) for _, r in seeds.iterrows()}
    seed_lookup = _seed_lookup_for_df(ctx["seeds_df"], ctx["season"])
    names = dict(zip(ctx["teams_df"]["TeamID"], ctx["teams_df"]["TeamName"]))

    rows: list[dict[str, Any]] = []
    winners: dict[str, int] = {}

    def token_display(token: str) -> str:
        if token in seed_to_team:
            tid = seed_to_team[token]
            return f"({seed_lookup.get(tid, '?')}) {names.get(tid, f'Team {tid}')}"
        if token in slot_map:
            return f"Winner of {token}"
        return token

    for slot in ordered:
        strong_t = str(slot_map[slot]["StrongSeed"])
        weak_t = str(slot_map[slot]["WeakSeed"])

        team_a = seed_to_team.get(strong_t, winners.get(strong_t))
        team_b = seed_to_team.get(weak_t, winners.get(weak_t))
        row = {
            "Slot": slot,
            "Round": round_map.get(slot, "Round 1"),
            "Region": region_map.get(slot, "National"),
            "TeamAID": team_a,
            "TeamBID": team_b,
            "TeamADisplay": token_display(strong_t) if team_a is None else f"({seed_lookup.get(team_a, '?')}) {names.get(team_a, f'Team {team_a}')}",
            "TeamBDisplay": token_display(weak_t) if team_b is None else f"({seed_lookup.get(team_b, '?')}) {names.get(team_b, f'Team {team_b}')}",
        }

        if team_a is not None and team_b is not None:
            scored = score_matchups_df(
                pd.DataFrame(
                    [
                        {
                            "Slot": slot,
                            "TeamAID": team_a,
                            "TeamBID": team_b,
                            "TeamAName": names.get(team_a, f"Team {team_a}"),
                            "TeamBName": names.get(team_b, f"Team {team_b}"),
                            "TeamASeedNum": seed_lookup.get(team_a, np.nan),
                            "TeamBSeedNum": seed_lookup.get(team_b, np.nan),
                        }
                    ]
                ),
                ctx,
                top_k=3,
            ).iloc[0]
            for c in [
                "UpsetProb",
                "RecommendedPick",
                "Confidence",
                "Factor1",
                "Factor2",
                "Factor3",
                "Error",
                "Reasons",
                "WorseSeedTeam",
                "Underdog",
                "Favorite",
                "UnderdogSeed",
                "FavoriteSeed",
            ]:
                row[c] = scored.get(c)

            picked = picks.get(slot)
            if picked is None:
                row["PickName"] = ""
            else:
                row["PickName"] = names.get(picked, f"Team {picked}")
                winners[slot] = int(picked)
        else:
            row["UpsetProb"] = np.nan
            row["RecommendedPick"] = ""
            row["Confidence"] = np.nan
            row["Factor1"] = row["Factor2"] = row["Factor3"] = ""
            row["Reasons"] = []
            row["Error"] = ""
            row["PickName"] = ""
        rows.append(row)

    return pd.DataFrame(rows)


def auto_pick_bracket(ctx: dict[str, Any], threshold: float) -> dict[str, int]:
    """Auto-pick bracket favoring favorites unless upset probability exceeds threshold."""
    picks: dict[str, int] = {}
    bracket_df = resolve_bracket_state(ctx, picks={})
    names = dict(zip(ctx["teams_df"]["TeamID"], ctx["teams_df"]["TeamName"]))
    for _, r in bracket_df.iterrows():
        if pd.isna(r["TeamAID"]) or pd.isna(r["TeamBID"]):
            continue
        team_a = int(r["TeamAID"])
        team_b = int(r["TeamBID"])
        if pd.notna(r["UpsetProb"]) and float(r["UpsetProb"]) >= threshold and pd.notna(r.get("WorseSeedTeam")):
            pick_name = str(r.get("WorseSeedTeam"))
        else:
            pick_name = str(r.get("RecommendedPick", ""))
        if pick_name == names.get(team_a):
            picks[str(r["Slot"])] = team_a
        elif pick_name == names.get(team_b):
            picks[str(r["Slot"])] = team_b
        else:
            picks[str(r["Slot"])] = team_a if team_a < team_b else team_b
    return picks


@st.cache_data
def run_simulation_cached(
    season: int,
    n_sims: int,
    randomness: float,
    model_hash: str,
    seeds_df: pd.DataFrame,
    slots_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    teams_df: pd.DataFrame,
    model_path_str: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo simulation with cached inputs."""
    model = load_cached_model(model_path_str) if model_path_str else None
    required = infer_required_features(model) if model is not None else []
    season_ctx = build_season_context_from_frames(season, seeds_df, team_features_df, teams_df=teams_df)
    names = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))

    seed_col = "Seed" if "Seed" in seeds_df.columns else "SeedStr"
    s = seeds_df[seeds_df["Season"] == season][[seed_col, "TeamID"]]
    seed_to_team = {str(r[seed_col]): int(r["TeamID"]) for _, r in s.iterrows()}
    seed_lookup = _seed_lookup_for_df(seeds_df, season)

    slots = slots_df[slots_df["Season"] == season][["Slot", "StrongSeed", "WeakSeed"]].copy()
    rem = slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")
    ordered: list[str] = []
    avail = set(seed_to_team.keys())
    while rem:
        moved = False
        for slot in list(rem.keys()):
            a = str(rem[slot]["StrongSeed"])
            b = str(rem[slot]["WeakSeed"])
            if a in avail and b in avail:
                ordered.append(slot)
                avail.add(slot)
                del rem[slot]
                moved = True
        if not moved:
            break
    slot_map = slots.set_index("Slot")[["StrongSeed", "WeakSeed"]].to_dict(orient="index")

    # Depth -> round label
    depth = {}
    for slot in ordered:
        a = str(slot_map[slot]["StrongSeed"])
        b = str(slot_map[slot]["WeakSeed"])
        da = 0 if a in seed_to_team else depth.get(a, 0)
        db = 0 if b in seed_to_team else depth.get(b, 0)
        depth[slot] = 1 + max(da, db)
    uniq = sorted(set(depth.values()))
    stage_map = {d: ADV_STAGES[min(i, len(ADV_STAGES)-1)] for i, d in enumerate(uniq)}
    round_map = {slot: stage_map.get(depth.get(slot, 1), "Round 1") for slot in slot_map}

    pair_cache = {}
    seeded = sorted(set(seed_to_team.values()))
    if model is not None and required:
        rows, keys = [], []
        for a, b in itertools.combinations(seeded, 2):
            frow, err = build_feature_row(a, b, season_ctx)
            if err is not None:
                continue
            if any((f not in frow) or pd.isna(frow[f]) for f in required):
                continue
            rows.append({f: frow[f] for f in required})
            keys.append((a, b))
        if rows:
            x = pd.DataFrame(rows, columns=required)
            p1 = predict_team1_win_prob(model, x)
            for i, key in enumerate(keys):
                pair_cache[key] = float(p1[i])

    rng = np.random.default_rng(42)
    adv = {tid: {s: 0 for s in ["R32", "S16", "E8", "F4", "TitleGame", "Champion"]} for tid in seeded}
    round_to_adv = {
        "Round 1": "R32",
        "Round 2": "S16",
        "Sweet 16": "E8",
        "Elite 8": "F4",
        "Final Four": "TitleGame",
        "Championship": "Champion",
    }
    match_ct = defaultdict(int)
    match_p = defaultdict(float)
    match_upset = defaultdict(float)

    for _ in range(n_sims):
        winners = {}
        for slot in ordered:
            a_t = str(slot_map[slot]["StrongSeed"])
            b_t = str(slot_map[slot]["WeakSeed"])
            ta = seed_to_team.get(a_t, winners.get(a_t))
            tb = seed_to_team.get(b_t, winners.get(b_t))
            if ta is None or tb is None:
                continue
            t1, t2 = (ta, tb) if ta < tb else (tb, ta)
            if model is not None and (t1, t2) in pair_cache:
                p1 = pair_cache[(t1, t2)]
                p_a, _ = map_pair_probs(ta, tb, p1)
            else:
                p_a = _heuristic_prob(seed_lookup.get(ta, np.nan), seed_lookup.get(tb, np.nan))
            p_adj = apply_temperature_scaling(p_a, randomness)
            winner = ta if rng.random() < p_adj else tb
            winners[slot] = winner

            adv_col = round_to_adv.get(round_map.get(slot, "Round 1"))
            if adv_col:
                adv[winner][adv_col] += 1

            ca, cb = (ta, tb) if ta < tb else (tb, ta)
            p_ca = p_a if ta == ca else 1.0 - p_a
            sa, sb = seed_lookup.get(ca), seed_lookup.get(cb)
            if sa is None or sb is None:
                upset = np.nan
            elif sa > sb:
                upset = p_ca
            elif sb > sa:
                upset = 1.0 - p_ca
            else:
                upset = np.nan
            key = (round_map.get(slot, "Round 1"), ca, cb)
            match_ct[key] += 1
            match_p[key] += p_ca
            match_upset[key] += upset if pd.notna(upset) else 0.0

    adv_rows = []
    for tid in seeded:
        row = {"TeamID": tid, "TeamName": names.get(tid, f"Team {tid}"), "SeedNum": seed_lookup.get(tid, np.nan)}
        for col in ["R32", "S16", "E8", "F4", "TitleGame", "Champion"]:
            row[f"P_{col}"] = adv[tid][col] / n_sims
        adv_rows.append(row)
    adv_df = pd.DataFrame(adv_rows).sort_values("P_Champion", ascending=False).reset_index(drop=True)

    match_rows = []
    for (rnd, a, b), ct in match_ct.items():
        match_rows.append(
            {
                "Round": rnd,
                "TeamA": names.get(a, f"Team {a}"),
                "TeamB": names.get(b, f"Team {b}"),
                "Frequency": ct,
                "AvgWinProbTeamA": match_p[(rnd, a, b)] / ct,
                "AvgUpsetProb": match_upset[(rnd, a, b)] / ct,
            }
        )
    match_df = pd.DataFrame(match_rows).sort_values("Frequency", ascending=False).reset_index(drop=True)
    return adv_df, match_df
