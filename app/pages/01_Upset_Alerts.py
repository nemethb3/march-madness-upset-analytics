"""Upset Alerts page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from app.components.charts import upset_bar_chart, upset_histogram
from app.components.io import build_round1_df, render_sidebar, score_round1_matchups

st.set_page_config(page_title="Upset Alerts", page_icon="🚨", layout="wide")
ctx = render_sidebar()

st.title("Round 1 Upset Alerts")

round1 = build_round1_df(ctx)
scored = score_round1_matchups(round1, ctx, top_k=3)

if round1.empty:
    st.warning("No Round 1 slots found for this season.")
    st.stop()

filters_col1, filters_col2 = st.columns([2, 1])
seed_pairs = sorted([x for x in scored["SeedPair"].dropna().unique().tolist()])
selected_pairs = filters_col1.multiselect("Filter by seed matchup", options=seed_pairs, default=seed_pairs)
alerts_only = filters_col2.checkbox("Alerts only", value=False)

view = scored.copy()
if selected_pairs:
    view = view[view["SeedPair"].isin(selected_pairs)]
if alerts_only:
    view = view[view["UpsetProb"] >= ctx["upset_threshold"]]

tabs = st.tabs(["Table", "Charts"])

with tabs[0]:
    st.dataframe(view.sort_values("UpsetProb", ascending=False), use_container_width=True, hide_index=True)
    st.download_button(
        "Download alerts CSV",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_round1_upset_alerts.csv",
        mime="text/csv",
    )

with tabs[1]:
    chart_df = view[view["Error"] == ""].copy()
    if chart_df.empty:
        st.info("No scored rows available for charts.")
    else:
        st.plotly_chart(upset_bar_chart(chart_df), use_container_width=True)
        st.plotly_chart(upset_histogram(chart_df), use_container_width=True)

with st.expander("What does this mean?"):
    st.write(
        "Upset probability is the chance the worse-seeded team wins. Sort by UpsetProb to find the most likely upsets."
    )

