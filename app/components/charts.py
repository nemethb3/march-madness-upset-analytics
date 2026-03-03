"""Plotly chart helpers for dashboard pages."""

from __future__ import annotations

import pandas as pd
import plotly.express as px


def get_col(df: pd.DataFrame, name_options: list[str]) -> pd.Series:
    """Return the first matching column as a Series, robust to duplicate names."""
    for name in name_options:
        if name in df.columns:
            col_obj = df.loc[:, name]
            if isinstance(col_obj, pd.DataFrame):
                return col_obj.iloc[:, 0]
            return col_obj
    raise KeyError(f"None of the columns found: {name_options}")


def upset_bar_chart(df: pd.DataFrame):
    """Top upset probability bar chart."""
    safe = df.loc[:, ~df.columns.duplicated()].copy()
    top = safe.sort_values("UpsetProb", ascending=False).head(10).copy()
    underdog = get_col(top, ["Underdog", "TeamAName"])
    favorite = get_col(top, ["Favorite", "TeamBName"])
    top["Matchup"] = underdog.astype(str) + " vs " + favorite.astype(str)
    return px.bar(top, x="UpsetProb", y="Matchup", orientation="h", title="Top 10 Upset Probabilities")


def upset_histogram(df: pd.DataFrame):
    """Upset probability distribution histogram."""
    safe = df.loc[:, ~df.columns.duplicated()].copy()
    return px.histogram(safe, x="UpsetProb", nbins=12, title="Upset Probability Distribution")


def title_odds_chart(df: pd.DataFrame):
    """Top title odds chart."""
    top = df.sort_values("P_Champion", ascending=False).head(10)
    return px.bar(top, x="TeamName", y="P_Champion", title="Top 10 Championship Odds")
