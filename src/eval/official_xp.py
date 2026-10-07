"""Official xP is usable only on gameweeks where the column was filled.

An all-zero gameweek is an unfilled scrape. It is not a forecast of zero.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache"


def unfilled_gameweeks(frame: pd.DataFrame, column: str = "xP") -> list[int]:
    """Gameweeks whose maximum value in ``column`` is not strictly positive."""
    work = frame.copy()
    work["_gw"] = pd.to_numeric(work["GW"] if "GW" in work.columns else work["gw"], errors="coerce")
    work["_xp"] = pd.to_numeric(work[column], errors="coerce")
    missing: list[int] = []
    for gw, block in work.dropna(subset=["_gw"]).groupby("_gw", sort=True):
        peak = block["_xp"].max()
        if not pd.notna(peak) or float(peak) <= 0.0:
            missing.append(int(gw))
    return missing


def usable_gameweeks(frame: pd.DataFrame, column: str = "score_official_xp") -> set[int]:
    """The complement of :func:`unfilled_gameweeks` on a scored frame."""
    work = frame.copy()
    work["GW"] = pd.to_numeric(work["gw"], errors="coerce")
    missing = set(unfilled_gameweeks(work, column))
    present = {int(gw) for gw in work["GW"].dropna().astype(int).unique()}
    return present - missing


def sheet_fill(season: str) -> dict[str, object]:
    """Read one cached sheet. ``modified`` is described, not treated as a clock."""
    path = CACHE / f"merged_gw_{season.replace('-', '_')}.csv"
    frame = pd.read_csv(path, usecols=lambda name: name in {
        "GW", "minutes", "xP", "modified", "value", "selected", "transfers_balance", "total_points",
    })
    missing = unfilled_gameweeks(frame, "xP")
    modified = "absent"
    if "modified" in frame.columns:
        values = frame["modified"].map(lambda item: str(item).strip().lower())
        modified = "boolean false on every row" if set(values) <= {"false", "0", "nan"} else "present"
    played = pd.to_numeric(frame.get("minutes"), errors="coerce").fillna(0) <= 0
    xp = pd.to_numeric(frame["xP"], errors="coerce")
    high_blank = int(((played) & (xp >= 4)).sum()) if "minutes" in frame.columns else 0

    def _stat(column: str, how: str) -> float | None:
        if column not in frame.columns:
            return None
        values = pd.to_numeric(frame[column], errors="coerce")
        if not values.notna().any():
            return None
        return float(getattr(values, how)())

    transfers = pd.to_numeric(frame["transfers_balance"], errors="coerce") if "transfers_balance" in frame.columns else None
    return {
        "season": season,
        "unfilled": missing,
        "n_gameweeks": int(frame["GW"].nunique()),
        "modified": modified,
        "high_xp_blanks": high_blank,
        "value_median": _stat("value", "median"),
        "value_max": _stat("value", "max"),
        "selected_mean": _stat("selected", "mean"),
        "transfers_all_zero": bool(transfers.fillna(0).eq(0).all()) if transfers is not None else None,
    }
