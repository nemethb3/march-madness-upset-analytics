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
if scored.empty:
    st.error("No valid matchups could be scored for this season bundle.")
    st.stop()


def _enforce_alert_seed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create/backfill Favorite/Underdog seed schema before normalization."""
    out = df.copy()

    has_team_seed_schema = {"TeamASeedNum", "TeamBSeedNum", "TeamAName", "TeamBName"}.issubset(out.columns)
    if has_team_seed_schema:
        out["TeamASeedNum"] = pd.to_numeric(out["TeamASeedNum"], errors="coerce")
        out["TeamBSeedNum"] = pd.to_numeric(out["TeamBSeedNum"], errors="coerce")

        team_a_is_fav = out["TeamASeedNum"] <= out["TeamBSeedNum"]
        out["Favorite"] = np.where(team_a_is_fav, out["TeamAName"], out["TeamBName"])
        out["Underdog"] = np.where(team_a_is_fav, out["TeamBName"], out["TeamAName"])
        out["FavoriteSeed"] = np.where(team_a_is_fav, out["TeamASeedNum"], out["TeamBSeedNum"])
        out["UnderdogSeed"] = np.where(team_a_is_fav, out["TeamBSeedNum"], out["TeamASeedNum"])

        if {"TeamAID", "TeamBID"}.issubset(out.columns):
            out["FavoriteTeamID"] = np.where(team_a_is_fav, out["TeamAID"], out["TeamBID"])
            out["UnderdogTeamID"] = np.where(team_a_is_fav, out["TeamBID"], out["TeamAID"])

    if {"FavoriteSeed", "UnderdogSeed"}.issubset(out.columns):
        out["FavoriteSeed"] = pd.to_numeric(out["FavoriteSeed"], errors="coerce")
        out["UnderdogSeed"] = pd.to_numeric(out["UnderdogSeed"], errors="coerce")
        # Backfill from TeamA/TeamB seeds if available.
        if {"TeamASeedNum", "TeamBSeedNum"}.issubset(out.columns):
            team_min = out[["TeamASeedNum", "TeamBSeedNum"]].min(axis=1)
            team_max = out[["TeamASeedNum", "TeamBSeedNum"]].max(axis=1)
            out["FavoriteSeed"] = out["FavoriteSeed"].fillna(team_min)
            out["UnderdogSeed"] = out["UnderdogSeed"].fillna(team_max)

    return out


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

        if "Season" not in out.columns:
            out["Season"] = ctx["season"]

        required = ["Season", "SeedPair", "Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb"]
        out = out[[c for c in required if c in out.columns] + [c for c in out.columns if c not in required]]

    schema_b = {"TeamAName", "TeamBName", "TeamASeedNum", "TeamBSeedNum", "P_TeamAWin", "P_TeamBWin", "WorseSeedTeam"}
    if not has_schema_a and schema_b.issubset(out.columns):
        out["Underdog"] = out["WorseSeedTeam"]
        out["Favorite"] = np.where(out["Underdog"] == out["TeamAName"], out["TeamBName"], out["TeamAName"])
        out["UnderdogSeed"] = out[["TeamASeedNum", "TeamBSeedNum"]].max(axis=1)
        out["FavoriteSeed"] = out[["TeamASeedNum", "TeamBSeedNum"]].min(axis=1)
        out["UpsetProb"] = np.where(
            out["Underdog"] == out["TeamAName"],
            out["P_TeamAWin"],
            np.where(out["Underdog"] == out["TeamBName"], out["P_TeamBWin"], np.nan),
        )
        if "Season" not in out.columns:
            out["Season"] = ctx["season"]

        required = ["Season", "SeedPair", "Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb"]
        out = out[[c for c in required if c in out.columns] + [c for c in out.columns if c not in required]]

    # Harden against NaNs in seeds and build SeedPair safely.
    if "FavoriteSeed" in out.columns:
        out["FavoriteSeed"] = pd.to_numeric(out["FavoriteSeed"], errors="coerce")
    else:
        out["FavoriteSeed"] = np.nan
    if "UnderdogSeed" in out.columns:
        out["UnderdogSeed"] = pd.to_numeric(out["UnderdogSeed"], errors="coerce")
    else:
        out["UnderdogSeed"] = np.nan

    low = out[["FavoriteSeed", "UnderdogSeed"]].min(axis=1)
    high = out[["FavoriteSeed", "UnderdogSeed"]].max(axis=1)
    seed_pair = low.astype("Int64").astype(str) + " vs " + high.astype("Int64").astype(str)
    invalid_seed = low.isna() | high.isna()
    seed_pair = seed_pair.where(~invalid_seed, "Unknown")
    out["SeedPair"] = seed_pair
    out["is_valid"] = ~invalid_seed

    if "Season" not in out.columns:
        out["Season"] = ctx["season"]
    if "UpsetProb" not in out.columns:
        out["UpsetProb"] = np.nan

    required = ["Season", "SeedPair", "Favorite", "Underdog", "FavoriteSeed", "UnderdogSeed", "UpsetProb", "is_valid"]
    missing_required = [c for c in ["Favorite", "Underdog"] if c not in out.columns]
    if missing_required:
        st.error(f"Upset alerts data missing required columns. Found: {sorted(df.columns.tolist())}")
        st.stop()

    out = out[[c for c in required if c in out.columns] + [c for c in out.columns if c not in required]]
    out = out.loc[:, ~out.columns.duplicated()].copy()
    return out


before_cols = scored.columns.tolist()
before_shape = scored.shape
scored = _enforce_alert_seed_columns(scored)
pre_norm_missing_fav = int(pd.to_numeric(scored.get("FavoriteSeed"), errors="coerce").isna().sum()) if "FavoriteSeed" in scored.columns else len(scored)
pre_norm_missing_dog = int(pd.to_numeric(scored.get("UnderdogSeed"), errors="coerce").isna().sum()) if "UnderdogSeed" in scored.columns else len(scored)
scored = normalize_alerts_df(scored)
after_cols = scored.columns.tolist()
after_shape = scored.shape

if "Reasons" not in scored.columns:
    scored["Reasons"] = [[] for _ in range(len(scored))]

# Keep full scored dataframe for seed-pair options and charts.
full_df = scored[scored["Error"] == ""].copy() if "Error" in scored.columns else scored.copy()
invalid_count = int((~full_df["is_valid"]).sum()) if "is_valid" in full_df.columns else 0
if invalid_count > 0:
    st.warning(f"Removed {invalid_count} matchup rows with missing/invalid seed values.")
full_df = full_df[full_df["is_valid"]].copy() if "is_valid" in full_df.columns else full_df
if full_df.empty:
    st.error("No valid matchups could be scored for this season bundle.")
    st.stop()

with st.sidebar.expander("Upset Alerts Debug"):
    if ctx.get("debug_mode", False):
        st.write({"columns_before_normalize": before_cols, "columns_after_normalize": after_cols, "round1_rows": len(round1)})
    else:
        st.caption("Enable 'Debug mode' in the sidebar to view diagnostics.")


def _alert_level(p: float) -> str:
    if p >= 0.30:
        return "High"
    if p >= 0.20:
        return "Medium"
    if p >= 0.15:
        return "Watch"
    return "Low"


full_df["AlertLevel"] = full_df["UpsetProb"].map(_alert_level)
full_df["SeedMatchup"] = full_df.apply(lambda r: f"{int(r['FavoriteSeed'])} vs {int(r['UnderdogSeed'])}", axis=1)

# Alerts view (filtered) is intentionally separate from full Round 1 scored data.
alerts_df = full_df.copy()
if ctx.get("upset_threshold") is not None:
    alerts_df = alerts_df[alerts_df["UpsetProb"] >= float(ctx["upset_threshold"])].copy()

f1, f2 = st.columns([2, 1])
seed_pair_options = sorted(full_df["SeedPair"].dropna().unique().tolist())
seed_filter = f1.multiselect(
    "Seed matchup filter",
    options=seed_pair_options,
    default=seed_pair_options,
    key="alerts_selected_seed_pairs",
)
level_filter = f2.multiselect(
    "Alert level",
    options=["High", "Medium", "Watch", "Low"],
    default=["High", "Medium", "Watch"],
    key="alerts_selected_levels",
)

view_df = full_df.copy()
count_total = len(view_df)
if seed_filter:
    view_df = view_df[view_df["SeedPair"].isin(seed_filter)]
count_after_seed = len(view_df)
if level_filter:
    view_df = view_df[view_df["AlertLevel"].isin(level_filter)]
count_after_level = len(view_df)
if ctx.get("upset_threshold") is not None:
    view_df = view_df[view_df["UpsetProb"] >= float(ctx["upset_threshold"])]
count_after_threshold = len(view_df)
view_df = view_df.sort_values("UpsetProb", ascending=False)

tabs = st.tabs(["Upset Alerts", "Charts"])

with tabs[0]:
    if ctx.get("debug_mode", False):
        with st.expander("Debug: alerts dataframe"):
            st.write(
                {
                    "season": ctx["season"],
                    "shape_before_normalize": before_shape,
                    "shape_after_normalize": after_shape,
                    "full_df_shape": full_df.shape,
                    "alerts_df_shape": alerts_df.shape,
                    "view_df_shape": view_df.shape,
                    "upset_prob_min": float(full_df["UpsetProb"].min()) if not full_df.empty else None,
                    "upset_prob_max": float(full_df["UpsetProb"].max()) if not full_df.empty else None,
                }
            )
            st.write({"na_FavoriteSeed_before_normalize": pre_norm_missing_fav, "na_UnderdogSeed_before_normalize": pre_norm_missing_dog})
            st.write(
                {
                    "na_FavoriteSeed_final": int(full_df["FavoriteSeed"].isna().sum()),
                    "na_UnderdogSeed_final": int(full_df["UnderdogSeed"].isna().sum()),
                }
            )
            st.write(full_df["SeedPair"].value_counts().sort_index())

    if view_df.empty:
        st.info("No games match your current filters. Try selecting more seed pairs/alert levels or lowering the upset threshold.")
        if ctx.get("debug_mode", False):
            st.write(
                {
                    "counts_total": count_total,
                    "counts_after_seed_filter": count_after_seed,
                    "counts_after_alert_filter": count_after_level,
                    "counts_after_threshold": count_after_threshold,
                }
            )
    show_all = st.checkbox("Show all games", value=False, key="alerts_show_all_games")
    show_df = view_df.copy() if show_all else view_df.head(10).copy()
    for _, row in show_df.iterrows():
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

    if (not show_all) and len(view_df) > 10:
        with st.expander("Show all games"):
            for _, row in view_df.iloc[10:].iterrows():
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
    export_df = view_df[export_cols].copy()
    export_df["Reasons"] = export_df["Reasons"].apply(lambda x: "; ".join(x) if isinstance(x, list) else "")
    st.download_button(
        "Export upset alerts CSV",
        export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"{ctx['season']}_upset_alerts.csv",
        mime="text/csv",
    )

with tabs[1]:
    chart_df = full_df.sort_values("UpsetProb", ascending=False).copy()
    if chart_df.empty:
        st.info("No data available for charts.")
    else:
        top10 = chart_df.nlargest(10, "UpsetProb")
        st.plotly_chart(upset_bar_chart(top10), use_container_width=True)
        hist_df = chart_df[["UpsetProb"]].copy()
        st.plotly_chart(upset_histogram(hist_df), use_container_width=True)

with st.expander("What does this mean?"):
    st.write(
        "These cards rank the most plausible Round 1 upsets. Higher upset chance means the underdog has a stronger path to winning."
    )
