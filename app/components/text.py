"""Text helpers and user-facing explanations."""

from __future__ import annotations

FEATURE_LABELS = {
    "SeedDiff": "Seed gap",
    "Diff_win_pct": "Win rate edge",
    "Diff_avg_margin": "Scoring margin edge",
    "Diff_net_rtg": "Efficiency edge",
    "Diff_massey_rank_mean": "Massey rank edge",
    "Diff_GiantKillerScore": "Giant-killer profile edge",
    "Team1_fg3a_rate": "Team1 three-point attempt rate",
    "Team2_fg3a_rate": "Team2 three-point attempt rate",
}


def pretty_factor(factor_text: str) -> str:
    """Map raw factor strings to plain-English labels."""
    if ":" not in factor_text:
        return factor_text
    feat, val = factor_text.split(":", 1)
    label = FEATURE_LABELS.get(feat.strip(), feat.strip().replace("_", " "))
    return f"{label}: {val.strip()}"


def simulation_effort_to_n_sims(label: str) -> int:
    """Map UI effort label to simulation count."""
    mapping = {"Fast": 5000, "Balanced": 20000, "Thorough": 50000}
    return mapping[label]

