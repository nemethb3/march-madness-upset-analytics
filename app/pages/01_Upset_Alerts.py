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


def normalize_alerts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize alerts dataframe into a stable schema for UI rendering."""
    out = df.copy()

    has_schema_a = {"Underdog", "Favorite", "UnderdogSeed", "FavoriteSeed"}.issubset(out.columns)
    if has_schema_a:
        if "UpsetProb" not in out.columns or out["UpsetProb"].isna().all():
            if "P_UnderdogWin" in out.columns:
                out["UpsetProb"] = out["P_UnderdogWin"]
            elif {"P_TeamAWin", "P_TeamBWin", "Underdog", "TeamAName", "TeamBName"}.issubset(out.columns):
                out["UpsetProb"] = np.where(
                    out["Underdog"] == out["TeamAName"],
                    out["P_TeamAWin"],
                    np.where(out["Underdog"] == out["TeamBName"], out["P_TeamBWin"], np.nan),
                )

        if "SeedPair" not in out.columns:
            out["SeedPair"] = out.apply(
                lambda r: f"{min(int(r['FavoriteSeed']), int(r['UnderdogSeed']))}-{max(int(r['FavoriteSeed']), int(r['UnderdogSeed']))}",
                axis=1,
            )
        if "Season" not in out.columns:
            out["Season"] = ctx["season"]

        required = ["Season", "SeedPair", "Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb"]
        return out[required + [c for c in out.columns if c not in required]]

    schema_b = {"TeamAName", "TeamBName", "TeamASeedNum", "TeamBSeedNum", "P_TeamAWin", "P_TeamBWin", "WorseSeedTeam"}
    if schema_b.issubset(out.columns):
        out["Underdog"] = out["WorseSeedTeam"]
        out["Favorite"] = np.where(out["Underdog"] == out["TeamAName"], out["TeamBName"], out["TeamAName"])
        out["UnderdogSeed"] = out[["TeamASeedNum", "TeamBSeedNum"]].max(axis=1)
        out["FavoriteSeed"] = out[["TeamASeedNum", "TeamBSeedNum"]].min(axis=1)
        out["UpsetProb"] = np.where(
            out["Underdog"] == out["TeamAName"],
            out["P_TeamAWin"],
            np.where(out["Underdog"] == out["TeamBName"], out["P_TeamBWin"], np.nan),
        )
        if "SeedPair" not in out.columns:
            out["SeedPair"] = out.apply(lambda r: f"{int(r['FavoriteSeed'])}-{int(r['UnderdogSeed'])}", axis=1)
        if "Season" not in out.columns:
            out["Season"] = ctx["season"]

        required = ["Season", "SeedPair", "Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb"]
        return out[required + [c for c in out.columns if c not in required]]

    st.error(f"Upset alerts data missing required columns. Found: {sorted(df.columns.tolist())}")
    st.stop()


before_cols = scored.columns.tolist()
scored = normalize_alerts_df(scored)
after_cols = scored.columns.tolist()

if "Reasons" not in scored.columns:
    scored["Reasons"] = [[] for _ in range(len(scored))]

with st.sidebar.expander("Upset Alerts Debug"):
    st.write({"columns_before_normalize": before_cols, "columns_after_normalize": after_cols})


def _alert_level(p: float) -> str:
    if p >= 0.30:
        return "High"
    if p >= 0.20:
        return "Medium"
    if p >= 0.15:
        return "Watch"
    return "Low"


scored["AlertLevel"] = scored["UpsetProb"].map(_alert_level)
scored["SeedMatchup"] = scored.apply(lambda r: f"{int(r['FavoriteSeed'])} vs {int(r['UnderdogSeed'])}", axis=1)

f1, f2 = st.columns([2, 1])
seed_options = sorted(scored["SeedMatchup"].dropna().unique().tolist())
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

