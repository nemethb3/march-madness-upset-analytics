"""Generate Round 1 upset alerts using matchup-specific model probabilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.build_round1_from_slots import build_round1_matchups
from src.historical_upset_rates import build_historical_upset_rates
from src.inference_utils import (
    build_factor_strings,
    build_feature_row,
    infer_required_features,
    load_model,
    load_season_context,
    map_pair_probs,
    predict_team1_win_prob,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Generate Round 1 upset alerts.")
    parser.add_argument("--season", type=int, required=True, help="Tournament season.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to saved model.")
    parser.add_argument("--delta", type=float, default=0.10, help="Bracket-group alert threshold.")
    parser.add_argument("--historical_margin", type=float, default=0.08, help="Historical baseline margin threshold.")
    parser.add_argument("--top_k", type=int, default=2, help="Top-k upset games per seed-pair to flag.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output CSV path.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root (default: data).")
    return parser.parse_args()


def _load_historical_rates(data_dir: Path) -> pd.DataFrame:
    """Load historical upset rates from processed path or build from raw files."""
    out_path = data_dir / "processed" / "historical_upset_rates.csv"
    if out_path.exists():
        return pd.read_csv(out_path)
    out_df = build_historical_upset_rates(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    return out_df


def main() -> None:
    """Build and save upset alert table."""
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    round1 = build_round1_matchups(args.data_dir, args.season)
    model = load_model(args.model_path)
    required_features = infer_required_features(model)
    ctx = load_season_context(args.data_dir, args.season)
    hist_rates = _load_historical_rates(args.data_dir)

    records: list[dict[str, Any]] = []
    scored_indices: list[int] = []
    scored_feature_rows: list[dict[str, Any]] = []

    for _, row in round1.iterrows():
        rec: dict[str, Any] = {
            "Season": args.season,
            "Slot": row["Slot"],
            "TeamAName": row["TeamAName"],
            "TeamBName": row["TeamBName"],
            "TeamASeedNum": row["TeamASeedNum"],
            "TeamBSeedNum": row["TeamBSeedNum"],
            "WorseSeedTeam": np.nan,
            "UpsetProb": np.nan,
            "PercentileInSeedPair": np.nan,
            "GroupMeanSeedPair": np.nan,
            "GroupStdSeedPair": np.nan,
            "HistoricalUpsetRate": np.nan,
            "IsAlertBracket": False,
            "IsAlertHistorical": False,
            "IsAlertDelta": False,
            "IsAlertTopK": False,
            "IsGiantKillerProfile": False,
            "RecommendedPick": np.nan,
            "Confidence": np.nan,
            "Factor1": np.nan,
            "Factor2": np.nan,
            "Factor3": np.nan,
            "Error": "",
            "_TeamAID": np.nan,
            "_TeamBID": np.nan,
            "_Team1ID": np.nan,
            "_Team2ID": np.nan,
            "_SeedPair": np.nan,
        }

        if pd.isna(row["TeamAID"]) or pd.isna(row["TeamBID"]):
            rec["Error"] = "missing TeamID mapping from slot seeds"
            records.append(rec)
            continue

        team_a_id = int(row["TeamAID"])
        team_b_id = int(row["TeamBID"])
        team1_id, team2_id = (team_a_id, team_b_id) if team_a_id < team_b_id else (team_b_id, team_a_id)
        rec["_TeamAID"] = team_a_id
        rec["_TeamBID"] = team_b_id
        rec["_Team1ID"] = team1_id
        rec["_Team2ID"] = team2_id

        full_row, err = build_feature_row(team1_id, team2_id, ctx)
        if err is not None:
            rec["Error"] = err
            records.append(rec)
            continue

        missing_required = [f for f in required_features if f not in full_row or pd.isna(full_row[f])]
        if missing_required:
            rec["Error"] = f"missing required model features: {missing_required[:8]}"
            records.append(rec)
            continue

        scored_feature_rows.append({f: full_row[f] for f in required_features})
        scored_indices.append(len(records))
        records.append(rec)

    if scored_feature_rows:
        x_df = pd.DataFrame(scored_feature_rows, columns=required_features)
        p_team1 = predict_team1_win_prob(model, x_df)
        factors = build_factor_strings(model, x_df, required_features, top_k=3)

        for i, rec_idx in enumerate(scored_indices):
            rec = records[rec_idx]
            p1 = float(p_team1[i])

            team_a_id = int(rec["_TeamAID"])
            team_b_id = int(rec["_TeamBID"])
            p_a, p_b = map_pair_probs(team_a_id, team_b_id, p1)
            seed_a = int(rec["TeamASeedNum"])
            seed_b = int(rec["TeamBSeedNum"])
            seed_pair = (min(seed_a, seed_b), max(seed_a, seed_b))
            rec["_SeedPair"] = seed_pair

            if seed_a > seed_b:
                worse_seed_team = rec["TeamAName"]
                upset_prob = p_a
                underdog_team_id = team_a_id
            elif seed_b > seed_a:
                worse_seed_team = rec["TeamBName"]
                upset_prob = p_b
                underdog_team_id = team_b_id
            else:
                worse_seed_team = "TieSeed"
                upset_prob = np.nan
                underdog_team_id = team_a_id

            rec["WorseSeedTeam"] = worse_seed_team
            rec["UpsetProb"] = upset_prob
            rec["RecommendedPick"] = rec["TeamAName"] if p_a >= p_b else rec["TeamBName"]
            rec["Confidence"] = max(p_a, p_b)
            rec["Factor1"], rec["Factor2"], rec["Factor3"] = factors[i]
            gk = ctx.feat_lookup.get(underdog_team_id, {}).get("GiantKillerScore", np.nan)
            rec["IsGiantKillerProfile"] = bool(pd.notna(gk) and float(gk) > 0.65)
            rec["Error"] = ""

    out_df = pd.DataFrame(records)
    ok_mask = out_df["Error"] == ""

    if ok_mask.any():
        out_df.loc[ok_mask, "GroupMeanSeedPair"] = out_df.loc[ok_mask].groupby("_SeedPair")["UpsetProb"].transform("mean")
        out_df.loc[ok_mask, "GroupStdSeedPair"] = (
            out_df.loc[ok_mask].groupby("_SeedPair")["UpsetProb"].transform("std").fillna(0.0)
        )
        out_df.loc[ok_mask, "PercentileInSeedPair"] = out_df.loc[ok_mask].groupby("_SeedPair")["UpsetProb"].rank(
            pct=True, method="average"
        )

        hist = hist_rates.rename(columns={"SeedA": "HistSeedA", "SeedB": "HistSeedB"})
        out_df["_HistSeedA"] = np.minimum(out_df["TeamASeedNum"], out_df["TeamBSeedNum"])
        out_df["_HistSeedB"] = np.maximum(out_df["TeamASeedNum"], out_df["TeamBSeedNum"])
        if "HistoricalUpsetRate" in out_df.columns:
            out_df = out_df.drop(columns=["HistoricalUpsetRate"])
        out_df = out_df.merge(
            hist[["HistSeedA", "HistSeedB", "HistoricalUpsetRate"]],
            left_on=["_HistSeedA", "_HistSeedB"],
            right_on=["HistSeedA", "HistSeedB"],
            how="left",
        )

        out_df.loc[ok_mask, "IsAlertBracket"] = out_df.loc[ok_mask, "UpsetProb"] >= (
            out_df.loc[ok_mask, "GroupMeanSeedPair"] + args.delta
        )
        out_df.loc[ok_mask, "IsAlertHistorical"] = out_df.loc[ok_mask, "UpsetProb"] >= (
            out_df.loc[ok_mask, "HistoricalUpsetRate"] + args.historical_margin
        )
        out_df.loc[ok_mask, "IsAlertDelta"] = out_df.loc[ok_mask, "IsAlertBracket"]

        tmp = out_df.loc[ok_mask].copy()
        tmp["_RankDesc"] = tmp.groupby("_SeedPair")["UpsetProb"].rank(method="first", ascending=False)
        topk_idx = tmp.index[tmp["_RankDesc"] <= max(args.top_k, 1)]
        out_df.loc[topk_idx, "IsAlertTopK"] = True

    output_cols = [
        "Season",
        "Slot",
        "TeamAName",
        "TeamBName",
        "TeamASeedNum",
        "TeamBSeedNum",
        "WorseSeedTeam",
        "UpsetProb",
        "PercentileInSeedPair",
        "GroupMeanSeedPair",
        "GroupStdSeedPair",
        "HistoricalUpsetRate",
        "IsAlertBracket",
        "IsAlertHistorical",
        "IsAlertDelta",
        "IsAlertTopK",
        "IsGiantKillerProfile",
        "RecommendedPick",
        "Confidence",
        "Factor1",
        "Factor2",
        "Factor3",
        "Error",
    ]
    for c in output_cols:
        if c not in out_df.columns:
            out_df[c] = np.nan if c != "Error" else ""
    out_df = out_df[output_cols].sort_values(["TeamASeedNum", "TeamBSeedNum", "Slot"]).reset_index(drop=True)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Wrote upset alerts: {args.out_csv}")
    print(f"Scored successfully: {int((out_df['Error'] == '').sum())}")
    print(f"Rows with errors: {int((out_df['Error'] != '').sum())}")


if __name__ == "__main__":
    main()
