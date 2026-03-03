"""Plain-English explanation helpers for upset reasons."""

from __future__ import annotations

from typing import Any

import pandas as pd

LABEL_MAP = {
    "diff_win_pct": "Better overall record",
    "diff_avg_margin": "Higher scoring margin",
    "diff_sos": "Played tougher schedule",
    "diff_3pa_rate": "Takes more 3-pointers",
    "diff_tov_rate": "Better turnover control",
    "diff_orb_rate": "Strong offensive rebounding",
    "diff_ft_rate": "Gets to the free throw line",
    "diff_def_eff": "Stronger defense",
    "diff_off_eff": "More efficient offense",
    "diff_massey": "Better power rating",
}

FEATURE_CANDIDATES = [
    ("diff_win_pct", ["Diff_win_pct", "diff_win_pct"], "higher"),
    ("diff_avg_margin", ["Diff_avg_margin", "diff_avg_margin"], "higher"),
    ("diff_sos", ["Diff_strength_proxy", "diff_sos"], "higher"),
    ("diff_3pa_rate", ["Diff_fg3a_rate", "Diff_fg3a_pg", "diff_3pa_rate"], "higher"),
    ("diff_tov_rate", ["Diff_tov_rate", "diff_tov_rate"], "lower"),
    ("diff_orb_rate", ["Diff_orb_pct", "Diff_or_pg", "diff_orb_rate"], "higher"),
    ("diff_ft_rate", ["Diff_fta_pg", "Diff_ft_pct", "diff_ft_rate"], "higher"),
    ("diff_def_eff", ["Diff_def_rtg", "diff_def_eff"], "lower"),
    ("diff_off_eff", ["Diff_off_rtg", "diff_off_eff"], "higher"),
    ("diff_massey", ["Diff_massey_rank_mean", "Diff_massey_rank_median", "diff_massey"], "lower"),
]


def _first_available_value(row: pd.Series, cols: list[str]) -> float | None:
    """Return first numeric value found in row from candidate columns."""
    for c in cols:
        if c in row.index and pd.notna(row[c]):
            try:
                return float(row[c])
            except Exception:
                continue
    return None


def build_underdog_reasons(row: pd.Series, feature_diffs: dict[str, Any] | None = None) -> list[str]:
    """Return top 4-5 plain-English factors that favor the underdog."""
    underdog = str(row.get("Underdog", "underdog"))
    underdog_is_team1 = str(row.get("WorseSeedTeam", "")) == str(row.get("Team1Name", ""))

    scored: list[tuple[float, str]] = []
    for key, candidates, pref in FEATURE_CANDIDATES:
        val = _first_available_value(row, candidates)
        if val is None:
            continue
        # Diff columns are Team1 - Team2. Convert to underdog perspective.
        underdog_adv = val if underdog_is_team1 else -val
        if pref == "lower":
            underdog_adv = -underdog_adv
        if underdog_adv > 0:
            scored.append((abs(underdog_adv), LABEL_MAP[key]))

    scored.sort(key=lambda x: x[0], reverse=True)
    reasons = [f"{label}" for _, label in scored[:5]]
    if len(reasons) < 4:
        fallback = [
            "Recent form suggests competitive upside",
            "Matchup profile supports upset potential",
            "Style fit can disrupt the favorite",
            "Momentum and game volatility favor this underdog",
        ]
        for item in fallback:
            if item not in reasons:
                reasons.append(item)
            if len(reasons) >= 4:
                break
    return reasons[:5]

