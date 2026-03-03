"""Wrapper CLI to run pipeline, upset alerts, and tournament simulation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Run full bracket analysis workflow.")
    parser.add_argument("--season", type=int, required=True, help="Tournament season to analyze.")
    parser.add_argument("--n_sims", type=int, default=50000, help="Number of tournament simulations.")
    parser.add_argument("--data_dir", type=Path, default=Path("data"), help="Data directory root.")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs"), help="Output directory root.")
    parser.add_argument("--temperature", type=float, default=0.85, help="Simulation temperature scaling.")
    parser.add_argument("--delta", type=float, default=0.10, help="Bracket-mean alert margin.")
    parser.add_argument("--historical_margin", type=float, default=0.08, help="Historical alert margin.")
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    """Run a subprocess command and raise on failure."""
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    """Execute the full bracket analysis sequence."""
    args = parse_args()
    season_out = args.output_dir / "reports" / f"{args.season}_bracket_analysis"
    season_out.mkdir(parents=True, exist_ok=True)

    _run(
        [
            sys.executable,
            "run_pipeline.py",
            "--data_dir",
            str(args.data_dir),
            "--output_dir",
            str(args.output_dir),
            "--calibrate",
            "--calibration_method",
            "sigmoid",
        ]
    )

    round1_csv = args.data_dir / "brackets" / "round1_from_slots.csv"
    _run(
        [
            sys.executable,
            "-m",
            "src.build_round1_from_slots",
            "--season",
            str(args.season),
            "--out_csv",
            str(round1_csv),
            "--data_dir",
            str(args.data_dir),
        ]
    )

    upset_out = season_out / "round1_upset_alerts.csv"
    _run(
        [
            sys.executable,
            "-m",
            "src.upset_alerts",
            "--season",
            str(args.season),
            "--model_path",
            str(args.output_dir / "models" / "logistic_regression_calibrated.joblib"),
            "--delta",
            str(args.delta),
            "--historical_margin",
            str(args.historical_margin),
            "--out_csv",
            str(upset_out),
            "--data_dir",
            str(args.data_dir),
        ]
    )

    _run(
        [
            sys.executable,
            "-m",
            "src.simulate_tournament",
            "--season",
            str(args.season),
            "--n_sims",
            str(args.n_sims),
            "--model_path",
            str(args.output_dir / "models" / "logistic_regression_calibrated.joblib"),
            "--temperature",
            str(args.temperature),
            "--out_dir",
            str(season_out),
            "--data_dir",
            str(args.data_dir),
        ]
    )

    summary_path = season_out / "analysis_summary.md"
    summary_lines = [
        f"# Bracket Analysis Summary ({args.season})",
        "",
        "- Pipeline: completed with calibrated logistic model (`sigmoid`).",
        f"- Simulations: `{args.n_sims}`",
        f"- Temperature: `{args.temperature}`",
        f"- Alert delta: `{args.delta}`",
        f"- Historical margin: `{args.historical_margin}`",
        "",
        "## Generated Files",
        f"- `{season_out / 'advancement_probabilities.csv'}`",
        f"- `{season_out / 'round1_upset_alerts.csv'}`",
        f"- `{season_out / 'simulated_matchups.csv'}`",
        f"- `{summary_path}`",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()

