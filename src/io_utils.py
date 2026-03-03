"""I/O utilities for loading and saving project artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src import config


def validate_columns(df: pd.DataFrame, required_columns: Iterable[str], name: str) -> None:
    """Validate that required columns exist in a dataframe."""
    missing = sorted(set(required_columns) - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def read_csv_checked(path: Path, required_columns: Iterable[str], name: str) -> pd.DataFrame:
    """Read CSV and validate required columns."""
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    df = pd.read_csv(path)
    validate_columns(df, required_columns, name=name)
    return df


def raw_path(data_dir: Path, filename: str) -> Path:
    """Return path to a file in data/raw."""
    return data_dir / config.RAW_DIRNAME / filename


def processed_path(data_dir: Path, filename: str) -> Path:
    """Return path to a file in data/processed."""
    return data_dir / config.PROCESSED_DIRNAME / filename


def models_path(output_dir: Path, filename: str) -> Path:
    """Return path to a file in outputs/models."""
    return output_dir / config.MODELS_DIRNAME / filename


def reports_path(output_dir: Path, filename: str) -> Path:
    """Return path to a file in outputs/reports."""
    return output_dir / config.REPORTS_DIRNAME / filename

