"""Encompassing regression on pre-deadline captures only.

``total_points = a + b * official_xp + c * score_xp``. Historical Vaastav
``xP`` is not an input. A survival call needs 20 such gameweeks. Fewer than
that, including zero, is held.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.eval.provenance import assert_predeadline_xp, aware_utc, commit_is_before, load_deadlines

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "data" / "predictions" / "2026-27"
OUTCOMES = ROOT / "data" / "cache" / "player_gw_2026_27.csv"
LIVE_FROM_GW = 6


def engine_coefficient(y: np.ndarray, official: np.ndarray, engine: np.ndarray) -> float | None:
    """OLS coefficient on the engine. None when the three columns are not full rank."""
    design = np.column_stack([np.ones(len(y)), official, engine])
    coef, _resid, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < 3 or not np.isfinite(coef[2]):
        return None
    return float(coef[2])


def coefficient_rows(frame: pd.DataFrame, deadlines: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """Refuse a sheet that has no pre-deadline capture. This walker does not score history."""
    assert_predeadline_xp(frame, deadlines if deadlines is not None else load_deadlines())
    raise RuntimeError("refusing xP: historical sheets are not the encompassing population")


def _git_commit_at(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "--", str(path)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    text = out.stdout.strip()
    return text or None


def _read_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["__path"] = str(path)
    return frame


def live_encompassing(root: Path | None = None) -> dict[str, Any]:
    """Score only gameweeks that have both captures and outcomes. This batch is held."""
    base = root or ROOT
    deadlines = load_deadlines(base / "data" / "predictions" / "2026-27" / "deadlines.json")
    folder = base / "data" / "predictions" / "2026-27"
    official_paths = sorted(folder.glob("gw*/official_*.csv"))
    engine_paths = sorted(
        path for path in folder.glob("gw*/*.csv") if not path.name.startswith("official_")
        and path.name != "deadlines.json"
    )
    for path in official_paths:
        assert_predeadline_xp(_read_csv(path).drop(columns="__path"), deadlines)
    engine_rows: list[dict[str, Any]] = []
    for path in engine_paths:
        frame = pd.read_csv(path)
        if "score" not in frame.columns or "gw" not in frame.columns:
            raise RuntimeError(f"refusing engine capture {path.name}")
        gw = int(frame["gw"].iloc[0])
        if int(frame["gw"].nunique()) != 1 or gw < LIVE_FROM_GW:
            raise RuntimeError(f"refusing engine capture {path.name}")
        canonical = deadlines[str(gw)]
        created = aware_utc(frame["created_at"].iloc[0])
        stated = aware_utc(frame["deadline"].iloc[0])
        if stated != aware_utc(canonical) or created >= aware_utc(canonical):
            raise RuntimeError(f"refusing engine capture {path.name}")
        committed = _git_commit_at(path)
        engine_rows.append(
            {
                "path": str(path.relative_to(base)),
                "gw": gw,
                "n": int(len(frame)),
                "git_commit_at": committed,
                "committed_before_deadline": bool(committed and commit_is_before(committed, canonical)),
            }
        )
    outcomes = pd.read_csv(base / "data" / "cache" / "player_gw_2026_27.csv")
    played = {int(gw) for gw in pd.to_numeric(outcomes["gw"], errors="coerce").dropna().astype(int)}
    evaluable = [
        row["gw"]
        for row in engine_rows
        if row["gw"] in played and any(f"gw{row['gw']:02d}" in path.as_posix() for path in official_paths)
    ]
    return {
        "status": "held",
        "survives": None,
        "n_gws": int(len(evaluable)),
        "evaluable_gws": evaluable,
        "engine_captures": engine_rows,
        "official_captures": [str(path.relative_to(base)) for path in official_paths],
        "outcomes_through_gw": max(played) if played else None,
        "reason": (
            "No gameweek has a pre-deadline official xP, a pre-deadline engine score, "
            "and a played outcome. The stop-forecasting decision stays held."
        ),
    }
