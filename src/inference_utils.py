"""Shared inference utilities for matchup scoring and explanations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


@dataclass
class SeasonContext:
    """Season-specific lookup context for feature building."""

    season: int
    team_id_to_name: dict[int, str]
    name_to_id: dict[str, int]
    seed_lookup: dict[int, int]
    feat_lookup: dict[int, dict[str, Any]]
    base_cols: list[str]
    numeric_base_cols: set[str]


def load_model(model_path: Path) -> Any:
    """Load a serialized sklearn model from joblib."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def infer_required_features(model: Any) -> list[str]:
    """Infer required feature names/order from a fitted model or underlying estimator."""
    if hasattr(model, "feature_names_in_"):
        return [str(c) for c in model.feature_names_in_]

    if isinstance(model, CalibratedClassifierCV) and hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
        base_est = model.calibrated_classifiers_[0].estimator
        if hasattr(base_est, "estimator"):
            base_est = base_est.estimator
        if hasattr(base_est, "feature_names_in_"):
            return [str(c) for c in base_est.feature_names_in_]

    if isinstance(model, Pipeline) and hasattr(model[-1], "feature_names_in_"):
        return [str(c) for c in model[-1].feature_names_in_]

    raise ValueError(
        "Could not infer required feature names from the saved model. "
        "Retrain using dataframe inputs so feature_names_in_ is persisted."
    )


def _is_int_like(value: Any) -> bool:
    """Return True when a value can be interpreted as an integer TeamID."""
    if isinstance(value, (int, np.integer)):
        return True
    if value is None:
        return False
    text = str(value).strip()
    return text.isdigit()


def resolve_team_id(team_value: Any, name_to_id: dict[str, int], valid_ids: set[int]) -> tuple[int | None, str | None]:
    """Resolve input team as TeamID from either integer-like value or exact team name."""
    if pd.isna(team_value):
        return None, "empty team value"

    if _is_int_like(team_value):
        team_id = int(str(team_value).strip())
        if team_id in valid_ids:
            return team_id, None
        return None, f"unknown TeamID: {team_id}"

    team_name = str(team_value).strip()
    team_id = name_to_id.get(team_name)
    if team_id is None:
        return None, f"unknown team name: {team_name}"
    return team_id, None


def load_season_context(data_dir: Path, season: int) -> SeasonContext:
    """Load season-specific seeds and team-season features for inference."""
    teams_path = data_dir / "raw" / "MTeams.csv"
    seeds_path = data_dir / "processed" / "tourney_seeds_clean.csv"
    feats_path = data_dir / "processed" / "team_season_features.csv"

    teams_df = pd.read_csv(teams_path, usecols=["TeamID", "TeamName"])
    seeds_df = pd.read_csv(seeds_path)
    feats_df = pd.read_csv(feats_path)

    team_id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    name_to_id = dict(zip(teams_df["TeamName"], teams_df["TeamID"]))

    season_seeds = seeds_df[seeds_df["Season"] == season][["TeamID", "SeedNum"]]
    feature_base_cols = [c for c in feats_df.columns if c not in {"Season", "TeamID"}]
    season_feats = feats_df[feats_df["Season"] == season][["TeamID", *feature_base_cols]]

    seed_lookup = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
    feat_lookup = season_feats.set_index("TeamID")[feature_base_cols].to_dict(orient="index")
    numeric_base_cols = {c for c in feature_base_cols if pd.api.types.is_numeric_dtype(feats_df[c])}

    return SeasonContext(
        season=season,
        team_id_to_name=team_id_to_name,
        name_to_id=name_to_id,
        seed_lookup=seed_lookup,
        feat_lookup=feat_lookup,
        base_cols=feature_base_cols,
        numeric_base_cols=numeric_base_cols,
    )


def build_season_context_from_frames(
    season: int,
    seeds_df: pd.DataFrame,
    team_features_df: pd.DataFrame,
    teams_df: pd.DataFrame | None = None,
) -> SeasonContext:
    """Build SeasonContext from in-memory dataframes."""
    if teams_df is not None and {"TeamID", "TeamName"}.issubset(teams_df.columns):
        team_id_to_name = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    else:
        team_ids = pd.unique(team_features_df["TeamID"])
        team_id_to_name = {int(t): f"Team {int(t)}" for t in team_ids}
    name_to_id = {v: k for k, v in team_id_to_name.items()}

    seed_col = "SeedNum" if "SeedNum" in seeds_df.columns else None
    if seed_col is None:
        if "Seed" in seeds_df.columns:
            tmp = seeds_df.copy()
            tmp["SeedNum"] = tmp["Seed"].astype(str).str.extract(r"(\d+)").astype(float)
            seed_col = "SeedNum"
            seeds_src = tmp
        elif "SeedStr" in seeds_df.columns:
            tmp = seeds_df.copy()
            tmp["SeedNum"] = tmp["SeedStr"].astype(str).str.extract(r"(\d+)").astype(float)
            seed_col = "SeedNum"
            seeds_src = tmp
        else:
            raise ValueError("Seeds dataframe must contain SeedNum or Seed/SeedStr.")
    else:
        seeds_src = seeds_df.copy()

    season_seeds = seeds_src[seeds_src["Season"] == season][["TeamID", seed_col]].rename(columns={seed_col: "SeedNum"})
    feature_base_cols = [c for c in team_features_df.columns if c not in {"Season", "TeamID"}]
    season_feats = team_features_df[team_features_df["Season"] == season][["TeamID", *feature_base_cols]]

    seed_lookup = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
    feat_lookup = season_feats.set_index("TeamID")[feature_base_cols].to_dict(orient="index")
    numeric_base_cols = {c for c in feature_base_cols if pd.api.types.is_numeric_dtype(team_features_df[c])}

    return SeasonContext(
        season=season,
        team_id_to_name={int(k): str(v) for k, v in team_id_to_name.items()},
        name_to_id={str(k): int(v) for k, v in name_to_id.items()},
        seed_lookup={int(k): int(v) for k, v in seed_lookup.items() if pd.notna(v)},
        feat_lookup=feat_lookup,
        base_cols=feature_base_cols,
        numeric_base_cols=numeric_base_cols,
    )


def build_feature_row(team1_id: int, team2_id: int, ctx: SeasonContext) -> tuple[dict[str, Any] | None, str | None]:
    """Build full feature row for deterministic matchup ordering Team1ID < Team2ID."""
    if team1_id >= team2_id:
        return None, "team ordering violation: Team1ID must be < Team2ID"

    t1_seed = ctx.seed_lookup.get(team1_id)
    t2_seed = ctx.seed_lookup.get(team2_id)
    t1_stats = ctx.feat_lookup.get(team1_id)
    t2_stats = ctx.feat_lookup.get(team2_id)

    errors: list[str] = []
    if t1_seed is None:
        errors.append(f"missing seed for Team1ID {team1_id} in season {ctx.season}")
    if t2_seed is None:
        errors.append(f"missing seed for Team2ID {team2_id} in season {ctx.season}")
    if t1_stats is None:
        errors.append(f"missing team-season features for Team1ID {team1_id} in season {ctx.season}")
    if t2_stats is None:
        errors.append(f"missing team-season features for Team2ID {team2_id} in season {ctx.season}")
    if errors:
        return None, "; ".join(errors)

    row: dict[str, Any] = {
        "Team1Seed": int(t1_seed),
        "Team2Seed": int(t2_seed),
        "SeedDiff": int(t1_seed) - int(t2_seed),
    }
    for col in ctx.base_cols:
        t1_val = t1_stats.get(col)
        t2_val = t2_stats.get(col)
        row[f"Team1_{col}"] = t1_val
        row[f"Team2_{col}"] = t2_val
        if col in ctx.numeric_base_cols:
            row[f"Diff_{col}"] = t1_val - t2_val
    return row, None


def get_team_names(team1_id: int, team2_id: int, ctx: SeasonContext) -> tuple[str, str]:
    """Return team display names for IDs."""
    return (
        ctx.team_id_to_name.get(team1_id, f"TeamID {team1_id}"),
        ctx.team_id_to_name.get(team2_id, f"TeamID {team2_id}"),
    )


def predict_team1_win_prob(model: Any, x_df: pd.DataFrame) -> np.ndarray:
    """Return Team1Win class probabilities aligned to label=1."""
    probs = model.predict_proba(x_df)
    classes = list(getattr(model, "classes_", [0, 1]))
    idx = classes.index(1) if 1 in classes else 1
    return probs[:, idx]


def map_pair_probs(team_a_id: int, team_b_id: int, p_team1_win: float) -> tuple[float, float]:
    """
    Map canonical Team1 probability to input team order.

    Team1 is always min(TeamAID, TeamBID). This enforces:
    P(A beats B) = 1 - P(B beats A).
    """
    if team_a_id < team_b_id:
        p_a = float(p_team1_win)
    else:
        p_a = float(1.0 - p_team1_win)
    p_b = float(1.0 - p_a)
    return p_a, p_b


def apply_temperature_scaling(p: float, temperature: float) -> float:
    """
    Apply temperature adjustment to Bernoulli probability.

    p_adj = p^T / (p^T + (1-p)^T)
    """
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    p_clip = float(np.clip(p, 1e-12, 1 - 1e-12))
    if abs(temperature - 1.0) < 1e-12:
        return p_clip
    num = p_clip**temperature
    den = num + (1 - p_clip) ** temperature
    return float(num / den)


def _explainable_estimator(model: Any) -> Any:
    """Resolve explainable estimator used for linear contribution decomposition."""
    if isinstance(model, CalibratedClassifierCV):
        if hasattr(model, "calibrated_classifiers_") and model.calibrated_classifiers_:
            est = model.calibrated_classifiers_[0].estimator
            if hasattr(est, "estimator"):
                return est.estimator
            return est
        raise ValueError("Calibrated model has no fitted calibrated_classifiers_.")
    return model


def build_factor_strings(model: Any, x_df: pd.DataFrame, feature_names: list[str], top_k: int) -> list[list[str]]:
    """Build top-k signed contribution strings for each row from logistic coefficients."""
    est = _explainable_estimator(model)
    if not hasattr(est, "predict_proba"):
        return [["N/A"] * top_k for _ in range(len(x_df))]

    estimator = est
    x_transformed: np.ndarray | Any = x_df.to_numpy()
    if isinstance(est, Pipeline):
        estimator = est.steps[-1][1]
        if len(est.steps) > 1:
            x_transformed = est[:-1].transform(x_df)

    if not hasattr(estimator, "coef_"):
        return [["N/A"] * top_k for _ in range(len(x_df))]

    coef = estimator.coef_.ravel()
    contributions = np.asarray(x_transformed) * coef

    rows: list[list[str]] = []
    for row in contributions:
        top_idx = np.argsort(np.abs(row))[::-1][:top_k]
        formatted = [f"{feature_names[i]}: {row[i]:+0.2f}" for i in top_idx]
        if len(formatted) < top_k:
            formatted += ["N/A"] * (top_k - len(formatted))
        rows.append(formatted)
    return rows
