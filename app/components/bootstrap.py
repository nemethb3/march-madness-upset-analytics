"""Bootstrap helpers for Streamlit Cloud import paths."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_repo_root_on_path() -> Path:
    """
    Ensure repository root is on sys.path so top-level `src` imports work.

    This is needed on Streamlit Community Cloud where script directory (`app/`)
    is on sys.path but repo root may not be.
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent

    # Fallback to conventional ../../ relative from app/components/
    fallback = current.parents[2]
    if str(fallback) not in sys.path:
        sys.path.insert(0, str(fallback))
    return fallback

