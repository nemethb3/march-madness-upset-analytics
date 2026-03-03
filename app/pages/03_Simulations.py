"""Simulations page."""

from __future__ import annotations

from components.bootstrap import ensure_repo_root_on_path

ensure_repo_root_on_path()

import streamlit as st

from components.charts import title_odds_chart
from components.io import render_sidebar, run_simulation_cached

st.set_page_config(page_title="Simulations", page_icon="🎲", layout="wide")
ctx = render_sidebar()

st.title("Tournament Simulations")

if st.button("Run simulations"):
    with st.spinner("Running Monte Carlo simulations..."):
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
    st.session_state["adv_df"] = adv_df
    st.session_state["matchup_df"] = matchup_df

adv_df = st.session_state.get("adv_df")
matchup_df = st.session_state.get("matchup_df")

if adv_df is None:
    st.info("Click 'Run simulations' to generate title odds and advancement probabilities.")
    st.stop()

tab1, tab2, tab3 = st.tabs(["Title Odds", "Advancement", "Common Matchups"])

with tab1:
    st.dataframe(adv_df[["TeamName", "SeedNum", "P_Champion"]].head(10), hide_index=True, use_container_width=True)
    st.plotly_chart(title_odds_chart(adv_df), use_container_width=True)

with tab2:
    st.dataframe(adv_df, use_container_width=True, hide_index=True)
    st.download_button(
        "Download advancement probabilities",
        data=adv_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_advancement_probabilities.csv",
        mime="text/csv",
    )

with tab3:
    st.dataframe(matchup_df.head(50), use_container_width=True, hide_index=True)
    st.download_button(
        "Download simulated matchups",
        data=matchup_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_simulated_matchups.csv",
        mime="text/csv",
    )

with st.expander("What does this mean?"):
    st.write(
        "Simulations advance winners through actual bracket slots. Each game probability depends on the real "
        "opponent reached in that simulation path."
    )
