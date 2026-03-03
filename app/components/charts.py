"""Plotly chart helpers for dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def upset_bar_chart(df: pd.DataFrame):
    """Top upset probability bar chart."""
    top = df.sort_values("UpsetProb", ascending=False).head(10).copy()
    top["Matchup"] = top["TeamAName"].astype(str) + " vs " + top["TeamBName"].astype(str)
    return px.bar(top, x="UpsetProb", y="Matchup", orientation="h", title="Top 10 Upset Probabilities")


def upset_histogram(df: pd.DataFrame):
    """Upset probability distribution histogram."""
    return px.histogram(df, x="UpsetProb", nbins=12, title="Upset Probability Distribution")


def title_odds_chart(df: pd.DataFrame):
    """Top title odds chart."""
    top = df.sort_values("P_Champion", ascending=False).head(10)
    return px.bar(top, x="TeamName", y="P_Champion", title="Top 10 Championship Odds")

