"""Odds for the live planner.

The Odds API is not called. A snapshot already on disk is the only input.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live.xmi import LiveInputError


def load_odds_snapshot(path: Path | str) -> pd.DataFrame:
    """Read a CSV snapshot. A missing file raises. Nothing is downloaded."""
    file = Path(path)
    if not file.is_file():
        raise LiveInputError(
            f"No odds snapshot at {file}. The live planner does not call the Odds API."
        )
    frame = pd.read_csv(file)
    if frame.empty:
        raise LiveInputError(f"Odds snapshot {file} is empty.")
    return frame


def fetch_odds_api(*_args: object, **_kwargs: object) -> None:
    """The paid odds endpoint is closed on this path."""
    raise LiveInputError(
        "The live planner does not call the Odds API. Pass a snapshot CSV instead."
    )
