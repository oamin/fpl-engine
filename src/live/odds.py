"""CSV reader for a snapshot already on disk.

Live prices are Betfair Exchange. This module does not call a bookmaker API.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.live.xmi import LiveInputError


def load_odds_snapshot(path: Path | str) -> pd.DataFrame:
    """Read a CSV snapshot. A missing file raises. Nothing is downloaded."""
    file = Path(path)
    if not file.is_file():
        raise LiveInputError(f"No odds snapshot at {file}.")
    frame = pd.read_csv(file)
    if frame.empty:
        raise LiveInputError(f"Odds snapshot {file} is empty.")
    return frame
