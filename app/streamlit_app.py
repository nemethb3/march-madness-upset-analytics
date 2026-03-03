"""March Madness Analytics Dashboard entry page."""

from __future__ import annotations

import streamlit as st

from components.io import render_sidebar

st.set_page_config(page_title="March Madness Analytics", page_icon="🏀", layout="wide")

ctx = render_sidebar()

st.title("March Madness Upset Analytics")
st.caption("Public-hosting friendly dashboard with upload mode and demo mode.")

col1, col2, col3 = st.columns(3)
col1.metric("Data Mode", ctx["mode"])
col2.metric("Season", ctx["season"])
col3.metric("Simulation Effort", f"{ctx['n_sims']:,} sims")

st.markdown(
    """
Use the pages in the left sidebar:

- **Upset Alerts**: identify high-upset-probability Round 1 games  
- **Bracket Builder**: make picks manually or auto-pick with your risk settings  
- **Simulations**: run path-dependent tournament Monte Carlo and inspect title odds
"""
)

with st.expander("What does this mean?"):
    st.write(
        "The app computes matchup-specific probabilities from your selected model (or a demo heuristic fallback), "
        "then uses those probabilities to generate upset alerts and tournament simulations."
    )
