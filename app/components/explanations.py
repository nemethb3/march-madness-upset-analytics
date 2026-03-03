"""Plain-English explanation helpers for upset reasons."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

FEATURE_LABELS: dict[str, str] = {
    "win_pct": "Better overall record",
    "avg_margin": "Higher scoring margin",
    "strength_proxy": "Played tougher schedule",
    "fg3a_rate": "Takes more 3s (higher variance)",
    "tov_rate": "Protects the ball better (lower turnover rate)",
    "orb_pct": "Better offensive rebounding (extra possessions)",
    "ft_pct": "More reliable free throw shooting",
    "off_rtg": "More efficient offense",
    "def_rtg": "Stronger defense (lower opponent efficiency)",
    "net_rtg": "Better two-way efficiency",
    "massey_rank_mean": "Better power rating",
    "massey_rank_median": "More consistent power ranking",
}

# +1 means higher is better, -1 means lower is better.
FEATURE_DIRECTIONS: dict[str, int] = {
    "win_pct": 1,
    "avg_margin": 1,
    "strength_proxy": 1,
    "fg3a_rate": 1,
    "tov_rate": -1,
    "orb_pct": 1,
    "ft_pct": 1,
    "off_rtg": 1,
    "def_rtg": -1,
    "net_rtg": 1,
    "massey_rank_mean": -1,
    "massey_rank_median": -1,
}

FALLBACK_REASONS = [
    "Takes more 3s (higher variance)",
    "Protects the ball better (lower turnover rate)",
    "Stronger defense (lower opponent efficiency)",
    "More efficient offense",
    "Better offensive rebounding (extra possessions)",
    "Can create extra transition opportunities",
    "Has a profile that can disrupt favorites",
]


def _deterministic_fallback(season: int, underdog_team_id: int, favorite_team_id: int, max_reasons: int) -> list[str]:
    """Deterministic fallback reason set for stable demo behavior."""
    token = f"{season}-{underdog_team_id}-{favorite_team_id}"
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    offset = int(digest[:8], 16) % len(FALLBACK_REASONS)
    ordered = FALLBACK_REASONS[offset:] + FALLBACK_REASONS[:offset]
    return ordered[:max(4, min(max_reasons, 5))]


def build_underdog_reasons(
    underdog_team_id: int,
    favorite_team_id: int,
    season: int,
    team_features_df: pd.DataFrame,
    feature_labels: dict[str, str] | None = None,
    max_reasons: int = 5,
) -> list[str]:
    """
    Build plain-English reasons using underdog-favorite feature differentials.

    Returns 4-5 reasons; falls back deterministically when data is missing.
    """
    labels = feature_labels or FEATURE_LABELS
    if team_features_df.empty:
        return _deterministic_fallback(season, underdog_team_id, favorite_team_id, max_reasons)

    season_df = team_features_df[team_features_df["Season"] == season]
    if season_df.empty:
        return _deterministic_fallback(season, underdog_team_id, favorite_team_id, max_reasons)

    u = season_df[season_df["TeamID"] == underdog_team_id]
    f = season_df[season_df["TeamID"] == favorite_team_id]
    if u.empty or f.empty:
        return _deterministic_fallback(season, underdog_team_id, favorite_team_id, max_reasons)

    u_row = u.iloc[0]
    f_row = f.iloc[0]
    numeric_cols = [c for c in season_df.columns if c not in {"Season", "TeamID", "TeamName"} and pd.api.types.is_numeric_dtype(season_df[c])]

    candidates: list[tuple[float, str]] = []
    for col in numeric_cols:
        direction = FEATURE_DIRECTIONS.get(col)
        label = labels.get(col)
        if direction is None or label is None:
            continue
        u_val = u_row.get(col)
        f_val = f_row.get(col)
        if pd.isna(u_val) or pd.isna(f_val):
            continue
        diff = float(u_val) - float(f_val)
        favor_score = diff * direction
        if favor_score > 0:
            candidates.append((abs(favor_score), label))

    # Prefer stronger effects first, deduplicating label text.
    candidates.sort(key=lambda x: x[0], reverse=True)
    reasons: list[str] = []
    seen = set()
    for _, label in candidates:
        if label in seen:
            continue
        seen.add(label)
        reasons.append(label)
        if len(reasons) >= max_reasons:
            break

    if len(reasons) < 4:
        fallback = _deterministic_fallback(season, underdog_team_id, favorite_team_id, max_reasons=5)
        for item in fallback:
            if item not in reasons:
                reasons.append(item)
            if len(reasons) >= 4:
                break

    return reasons[: max(4, min(max_reasons, 5))]

