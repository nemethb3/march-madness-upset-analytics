"""Inference CLI for matchup scoring and local explanations."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inference_utils import (
    build_factor_strings,
    build_feature_row,
    get_team_names,
    infer_required_features,
    load_model,
    load_season_context,
    predict_team1_win_prob,
    resolve_team_id,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for matchup prediction."""
    parser = argparse.ArgumentParser(description="Predict NCAA matchup outcomes for a given season.")
    parser.add_argument("--season", type=int, required=True, help="Season to use for seeds/team features.")
    parser.add_argument("--matchups_csv", type=Path, required=True, help="CSV with Season,TeamA,TeamB columns.")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to trained model joblib.")
    parser.add_argument("--out_csv", type=Path, required=True, help="Output CSV path for predictions.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top contribution factors to include.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root (default: data).")
    return parser.parse_args()


def main() -> None:
    """Run inference for input matchups and write predictions CSV."""
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    matchups = pd.read_csv(args.matchups_csv)
    required = {"Season", "TeamA", "TeamB"}
    missing = sorted(required - set(matchups.columns))
    if missing:
        raise ValueError(f"Matchup CSV missing required columns: {missing}")

    model = load_model(args.model_path)
    required_features = infer_required_features(model)
    ctx = load_season_context(args.data_dir, args.season)
    valid_ids = set(ctx.team_id_to_name.keys())

    records: list[dict[str, Any]] = []
    scored_index: list[int] = []
    scored_rows: list[dict[str, Any]] = []

    for _, row in matchups.iterrows():
        rec: dict[str, Any] = {
            "Season": args.season,
            "Team1ID": np.nan,
            "Team2ID": np.nan,
            "Team1Name": np.nan,
            "Team2Name": np.nan,
            "Team1Seed": np.nan,
            "Team2Seed": np.nan,
            "P_Team1Win": np.nan,
            "P_Team2Win": np.nan,
            "WorseSeedTeam": np.nan,
            "UpsetProb": np.nan,
            "RecommendedPick": np.nan,
            "Confidence": np.nan,
            "Error": "",
        }

        team_a_id, err_a = resolve_team_id(row["TeamA"], ctx.name_to_id, valid_ids)
        team_b_id, err_b = resolve_team_id(row["TeamB"], ctx.name_to_id, valid_ids)
        errs = [e for e in [err_a, err_b] if e is not None]
        if not errs and team_a_id == team_b_id:
            errs.append("TeamA and TeamB resolve to the same team")
        if errs:
            rec["Error"] = "; ".join(errs)
            records.append(rec)
            continue

        team1_id, team2_id = (team_a_id, team_b_id) if team_a_id < team_b_id else (team_b_id, team_a_id)
        team1_name, team2_name = get_team_names(team1_id, team2_id, ctx)
        rec["Team1ID"] = int(team1_id)
        rec["Team2ID"] = int(team2_id)
        rec["Team1Name"] = team1_name
        rec["Team2Name"] = team2_name

        full_row, row_err = build_feature_row(team1_id, team2_id, ctx)
        if row_err is not None:
            rec["Error"] = row_err
            records.append(rec)
            continue

        rec["Team1Seed"] = int(full_row["Team1Seed"])
        rec["Team2Seed"] = int(full_row["Team2Seed"])

        missing_required = [f for f in required_features if f not in full_row or pd.isna(full_row[f])]
        if missing_required:
            rec["Error"] = f"missing required model features: {missing_required[:8]}"
            records.append(rec)
            continue

        scored_rows.append({f: full_row[f] for f in required_features})
        scored_index.append(len(records))
        records.append(rec)

    if scored_rows:
        x_df = pd.DataFrame(scored_rows, columns=required_features)
        p_team1 = predict_team1_win_prob(model, x_df)
        p_team2 = 1.0 - p_team1
        factors = build_factor_strings(model, x_df, required_features, top_k=max(args.top_k, 1))

        for i, rec_idx in enumerate(scored_index):
            rec = records[rec_idx]
            p1 = float(p_team1[i])
            p2 = float(p_team2[i])
            t1_seed = int(rec["Team1Seed"])
            t2_seed = int(rec["Team2Seed"])

            if t1_seed > t2_seed:
                worse_team, upset_prob = rec["Team1Name"], p1
            elif t2_seed > t1_seed:
                worse_team, upset_prob = rec["Team2Name"], p2
            else:
                worse_team, upset_prob = "TieSeed", np.nan

            rec["P_Team1Win"] = p1
            rec["P_Team2Win"] = p2
            rec["WorseSeedTeam"] = worse_team
            rec["UpsetProb"] = upset_prob
            rec["RecommendedPick"] = rec["Team1Name"] if p1 >= p2 else rec["Team2Name"]
            rec["Confidence"] = max(p1, p2)
            rec["Error"] = ""
            for k in range(max(args.top_k, 1)):
                rec[f"Factor{k + 1}"] = factors[i][k]

    factor_cols = [f"Factor{i}" for i in range(1, max(args.top_k, 1) + 1)]
    output_cols = [
        "Season",
        "Team1ID",
        "Team2ID",
        "Team1Name",
        "Team2Name",
        "Team1Seed",
        "Team2Seed",
        "P_Team1Win",
        "P_Team2Win",
        "WorseSeedTeam",
        "UpsetProb",
        "RecommendedPick",
        "Confidence",
        *factor_cols,
        "Error",
    ]
    out_df = pd.DataFrame(records)
    for col in output_cols:
        if col not in out_df.columns:
            out_df[col] = np.nan if col != "Error" else ""
    out_df = out_df[output_cols]
    out_df.to_csv(args.out_csv, index=False)

    ok_count = int((out_df["Error"] == "").sum())
    err_count = int((out_df["Error"] != "").sum())
    print(f"Scored successfully: {ok_count}")
    print(f"Rows with errors: {err_count}")
    print(f"Wrote predictions: {args.out_csv}")


if __name__ == "__main__":
    main()

