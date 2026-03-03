"""Bracket Builder page."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from components.io import auto_pick_bracket, render_sidebar, resolve_bracket_state

st.set_page_config(page_title="March Madness Analytics", page_icon="🏀", layout="wide")
ctx = render_sidebar()

st.title("Bracket Builder")

if "bracket_picks" not in st.session_state:
    st.session_state["bracket_picks"] = {}

view_choice = st.radio("View", options=["Analysis View", "Bracket View"], horizontal=True)
view_mode = view_choice == "Bracket View"
picks = st.session_state["bracket_picks"]

adj_threshold = np.clip(
    ctx["upset_threshold"] - 0.15 * ctx["risk_tolerance"] - 0.10 * (1.0 - ctx["randomness"]),
    0.05,
    0.90,
)

col_a, col_b = st.columns([1, 2])
if col_a.button("Auto-pick all rounds"):
    st.session_state["bracket_picks"] = auto_pick_bracket(ctx, adj_threshold)
    picks = st.session_state["bracket_picks"]
col_b.caption(f"Auto-pick threshold (risk-adjusted): **{adj_threshold:.2f}**")

bracket_df = resolve_bracket_state(ctx, picks=picks)
if bracket_df.empty:
    st.error("Bracket structure is unavailable for this season bundle.")
    st.stop()

if not view_mode:
    round_order = ["Round 1", "Round 2", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    tabs = st.tabs(round_order)
    for tab, rnd in zip(tabs, round_order):
        with tab:
            round_df = bracket_df[bracket_df["Round"] == rnd].copy()
            if round_df.empty:
                st.info(f"No {rnd} games available.")
                continue
            for region in sorted(round_df["Region"].fillna("National").unique().tolist()):
                reg_df = round_df[round_df["Region"] == region]
                st.markdown(f"### {region}")
                for _, row in reg_df.iterrows():
                    with st.container(border=True):
                        st.markdown(f"**{row['Slot']}**")
                        st.write(f"{row['TeamADisplay']}  \nvs  \n{row['TeamBDisplay']}")
                        if row["TeamAID"] is not None and row["TeamBID"] is not None and not (
                            pd.isna(row["TeamAID"]) or pd.isna(row["TeamBID"])
                        ):
                            slot_key = str(row["Slot"])
                            team_a_id = int(row["TeamAID"])
                            team_b_id = int(row["TeamBID"])
                            options = {str(team_a_id): row["TeamADisplay"], str(team_b_id): row["TeamBDisplay"]}
                            default = str(picks.get(slot_key, team_a_id))
                            picked = st.radio(
                                "Pick winner",
                                options=list(options.keys()),
                                format_func=lambda x, m=options: m[x],
                                index=0 if default not in options else list(options.keys()).index(default),
                                key=f"pick_{slot_key}",
                                horizontal=True,
                            )
                            st.session_state["bracket_picks"][slot_key] = int(picked)
                            st.caption(
                                f"Recommended: {row['RecommendedPick']} | "
                                f"Confidence: {float(row['Confidence']):.1%}"
                                if not pd.isna(row["Confidence"])
                                else "Recommended pick pending"
                            )
                            if isinstance(row["Reasons"], list):
                                with st.expander("Why this matchup leans this way"):
                                    for reason in row["Reasons"][:5]:
                                        st.markdown(f"- {reason}")
                        else:
                            st.caption("Waiting for prior-round winners.")
else:
    st.subheader("Bracket View")
    cols = st.columns(6)
    round_order = ["Round 1", "Round 2", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    for i, rnd in enumerate(round_order):
        with cols[i]:
            st.markdown(f"**{rnd}**")
            df = bracket_df[bracket_df["Round"] == rnd].copy()
            if df.empty:
                st.write("-")
                continue
            for _, row in df.iterrows():
                picked = st.session_state["bracket_picks"].get(str(row["Slot"]))
                if picked is not None:
                    name = ctx["season_ctx"].team_id_to_name.get(int(picked), f"Team {picked}")
                    seed = "?"
                    if int(picked) == int(row["TeamAID"]) if row["TeamAID"] == row["TeamAID"] else False:
                        seed = row.get("TeamASeedNum", "?")
                    elif row["TeamBID"] == row["TeamBID"] and int(picked) == int(row["TeamBID"]):
                        seed = row.get("TeamBSeedNum", "?")
                    st.write(f"({seed}) {name}")
                else:
                    st.write(f"{row['TeamADisplay']} vs {row['TeamBDisplay']}")

with st.expander("What does this mean?"):
    st.write(
        "Analysis View shows each matchup with recommendations and reasons. Bracket View shows your current picks across rounds."
    )
