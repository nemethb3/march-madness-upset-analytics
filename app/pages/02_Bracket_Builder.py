"""Bracket Builder page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from components.io import build_round1_df, render_sidebar, score_round1_matchups
from components.text import pretty_factor

st.set_page_config(page_title="Bracket Builder", page_icon="🧩", layout="wide")
ctx = render_sidebar()

st.title("Bracket Builder")

round1 = build_round1_df(ctx)
scored = score_round1_matchups(round1, ctx, top_k=3)

if scored.empty:
    st.warning("No matchups available for bracket building.")
    st.stop()

threshold_adjusted = np.clip(
    ctx["upset_threshold"] - 0.15 * ctx["risk_tolerance"] - 0.10 * (1.0 - ctx["randomness"]),
    0.05,
    0.90,
)

if st.button("Auto-pick bracket"):
    for _, r in scored.iterrows():
        key = f"pick_{r['Slot']}"
        if r["Error"]:
            st.session_state[key] = ""
            continue
        pick = r["RecommendedPick"]
        if pd.notna(r["UpsetProb"]) and float(r["UpsetProb"]) >= threshold_adjusted:
            pick = r["WorseSeedTeam"]
        st.session_state[key] = pick

st.caption(f"Auto-pick trigger (adjusted): **{threshold_adjusted:.2f}**")

picks = []
for _, r in scored.sort_values("Slot").iterrows():
    with st.container(border=True):
        st.subheader(f"{r['Slot']}: {r['TeamAName']} vs {r['TeamBName']}")
        if r["Error"]:
            st.error(r["Error"])
            picks.append({"Slot": r["Slot"], "Pick": "", "Reason": r["Error"]})
            continue

        key = f"pick_{r['Slot']}"
        options = [r["TeamAName"], r["TeamBName"]]
        default_index = 0 if r["RecommendedPick"] == r["TeamAName"] else 1
        if key not in st.session_state:
            st.session_state[key] = options[default_index]
        choice = st.radio("Pick winner", options=options, key=key, horizontal=True)

        st.write(
            f"UpsetProb: **{r['UpsetProb']:.3f}** | "
            f"Recommended: **{r['RecommendedPick']}** | Confidence: **{r['Confidence']:.3f}**"
        )
        with st.expander("Why? (Top 3 factors)"):
            st.write(f"1. {pretty_factor(str(r['Factor1']))}")
            st.write(f"2. {pretty_factor(str(r['Factor2']))}")
            st.write(f"3. {pretty_factor(str(r['Factor3']))}")

        picks.append({"Slot": r["Slot"], "Pick": choice, "Reason": f"UpsetProb={r['UpsetProb']:.3f}"})

picks_df = pd.DataFrame(picks)
st.download_button(
    "Export picks CSV",
    data=picks_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{ctx['season']}_bracket_picks.csv",
    mime="text/csv",
)

with st.expander("What does this mean?"):
    st.write(
        "Auto-pick selects favorites by default, then flips to underdogs when upset probability crosses your "
        "risk-adjusted threshold."
    )
