"""Upset Alerts page."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from components.charts import upset_bar_chart, upset_histogram
from components.io import build_round1_df, render_sidebar, score_round1_matchups

st.set_page_config(page_title="March Madness Analytics", page_icon="🏀", layout="wide")
ctx = render_sidebar()

st.title("Upset Alerts")
st.subheader("Round 1 Alerts")

round1 = build_round1_df(ctx)
if round1.empty:
    st.error("Round 1 matchups are not available for this season bundle.")
    st.stop()

scored = score_round1_matchups(round1, ctx, top_k=5)
scored = scored[scored["Error"] == ""].copy()
if scored.empty:
    st.error("No valid matchups could be scored for this season bundle.")
    st.stop()

def _alert_level(p: float) -> str:
    if p >= 0.30:
        return "High"
    if p >= 0.20:
        return "Medium"
    if p >= 0.15:
        return "Watch"
    return "Low"


scored["AlertLevel"] = scored["UpsetProb"].map(_alert_level)
scored["SeedMatchup"] = scored.apply(
    lambda r: f"{min(int(r['TeamASeedNum']), int(r['TeamBSeedNum']))} vs {max(int(r['TeamASeedNum']), int(r['TeamBSeedNum']))}",
    axis=1,
)

f1, f2 = st.columns([2, 1])
seed_options = sorted(scored["SeedMatchup"].unique().tolist())
seed_filter = f1.multiselect("Seed matchup filter", options=seed_options, default=seed_options)
level_filter = f2.multiselect("Alert level", options=["High", "Medium", "Watch", "Low"], default=["High", "Medium", "Watch"])

view = scored[scored["SeedMatchup"].isin(seed_filter) & scored["AlertLevel"].isin(level_filter)].copy()
view = view.sort_values("UpsetProb", ascending=False)

tabs = st.tabs(["Alert Cards", "Charts"])

with tabs[0]:
    if view.empty:
        st.info("No games match your current filters.")
    for _, row in view.iterrows():
        underdog = row["Underdog"]
        favorite = row["Favorite"]
        underdog_seed = int(row["UnderdogSeed"])
        favorite_seed = int(row["FavoriteSeed"])
        level = row["AlertLevel"]
        level_icon = {"High": "🚨", "Medium": "⚠️", "Watch": "👀", "Low": "•"}[level]
        reasons = row["Reasons"] if isinstance(row["Reasons"], list) else []

        with st.container(border=True):
            st.markdown(f"### {level_icon} Upset Alert")
            st.markdown(f"**({underdog_seed}) {underdog} vs ({favorite_seed}) {favorite}**")
            st.markdown(f"**Upset chance:** {float(row['UpsetProb']):.1%}  \n**Alert level:** {level}")
            st.markdown(f"**Why {underdog} could win:**")
            for reason in reasons[:5]:
                st.markdown(f"- {reason}")

    export_cols = ["Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb", "AlertLevel", "Reasons"]
    export_df = view[export_cols].copy()
    export_df["Reasons"] = export_df["Reasons"].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
    st.download_button(
        "Export upset alerts CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_upset_alerts.csv",
        mime="text/csv",
    )

with tabs[1]:
    top10 = view.nlargest(10, "UpsetProb")
    if not top10.empty:
        st.plotly_chart(upset_bar_chart(top10.rename(columns={"Underdog": "TeamAName", "Favorite": "TeamBName"})), use_container_width=True)
    hist_df = view[["UpsetProb"]].copy()
    if not hist_df.empty:
        st.plotly_chart(upset_histogram(hist_df), use_container_width=True)

with st.expander("What does this mean?"):
    st.write(
        "These cards rank the most plausible Round 1 upsets. Higher upset chance means the underdog has a stronger path to winning."
    )

