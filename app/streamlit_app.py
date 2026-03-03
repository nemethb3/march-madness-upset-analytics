"""March Madness Analytics dashboard home."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from components.io import render_sidebar

st.set_page_config(page_title="March Madness Analytics", page_icon="🏀", layout="wide")

ctx = render_sidebar()

st.title("March Madness Analytics")
st.caption("Upset alerts, bracket picks, and tournament simulations in one place.")

col1, col2, col3 = st.columns(3)
col1.metric("Mode", ctx["mode"])
col2.metric("Season", ctx["season"])
col3.metric("Simulation Effort", f"{ctx['n_sims']:,}")

st.markdown(
    """
Use the pages in the sidebar:

- **Upset Alerts** for ranked upset opportunities  
- **Bracket Builder** for all-round picks and bracket view  
- **Simulations** for title odds and advancement probabilities
"""
)

with st.expander("What does this mean?"):
    st.write(
        "The dashboard loads a season bundle automatically and computes matchup-specific probabilities. "
        "You can explore upset opportunities, build picks, and run bracket simulations without uploading data files."
    )

