"""Simulations page."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from components.charts import title_odds_chart
from components.io import render_sidebar, run_simulation_cached

st.set_page_config(page_title="March Madness Analytics", page_icon="🏀", layout="wide")
ctx = render_sidebar()

st.title("Simulations")
st.caption("Path-dependent tournament simulations with matchup-specific probabilities.")

if st.button("Run simulations"):
    with st.spinner(f"Running {ctx['n_sims']:,} simulations..."):
        adv_df, matchup_df = run_simulation_cached(
            season=ctx["season"],
            n_sims=ctx["n_sims"],
            randomness=ctx["randomness"],
            model_hash=ctx["model_hash"],
            seeds_df=ctx["seeds_df"],
            slots_df=ctx["slots_df"],
            team_features_df=ctx["team_features_df"],
            teams_df=ctx["teams_df"],
            model_path_str=str(ctx["model_path"]) if ctx["model_path"] is not None else None,
        )
    st.session_state["sim_adv_df"] = adv_df
    st.session_state["sim_matchup_df"] = matchup_df

adv_df = st.session_state.get("sim_adv_df")
matchup_df = st.session_state.get("sim_matchup_df")

if adv_df is None:
    st.info("Run simulations to populate title odds and advancement tables.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Top Title Odds", "Advancement Probabilities", "Common Matchups"])

with tab1:
    st.dataframe(adv_df[["TeamName", "SeedNum", "P_Champion"]].head(10), hide_index=True, use_container_width=True)
    st.plotly_chart(title_odds_chart(adv_df), use_container_width=True)

with tab2:
    st.dataframe(adv_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download advancement CSV",
        adv_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_advancement_probabilities.csv",
        mime="text/csv",
    )

with tab3:
    st.dataframe(matchup_df.head(100), use_container_width=True, hide_index=True)
    st.download_button(
        "Download matchup frequencies CSV",
        matchup_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_simulated_matchups.csv",
        mime="text/csv",
    )

with st.expander("What does this mean?"):
    st.write(
        "Each simulation advances winners through the real bracket slots. "
        "Probabilities are recalculated against the actual opponent in each simulated path."
    )

