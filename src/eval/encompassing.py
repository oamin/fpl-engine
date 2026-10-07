"""Walk-forward encompassing regression.

``total_points = a + b * official xP + c * score_xp``. The fit uses only
earlier usable gameweeks in the same season. The coefficient that has to
survive is ``c``, on the three seasons that were not the tuning season.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.eval.official_xp import usable_gameweeks

TUNING_SEASON = "2025-26"


def engine_coefficient(y: np.ndarray, official: np.ndarray, engine: np.ndarray) -> float | None:
    """OLS coefficient on the engine. None when the three columns are not full rank."""
    design = np.column_stack([np.ones(len(y)), official, engine])
    coef, _resid, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < 3 or not np.isfinite(coef[2]):
        return None
    return float(coef[2])


def coefficient_rows(frame: pd.DataFrame, protocol: dict[str, Any]) -> list[dict[str, Any]]:
    spec = protocol["encompassing"]
    start = int(spec["eval_gw_start"])
    min_train = int(spec["min_train_gws"])
    gw_end = int(protocol["gw_end"])
    filled = usable_gameweeks(frame)
    season = str(frame["season"].iloc[0])
    work = frame.loc[frame["gw"].isin(filled)].copy()
    work = work.dropna(subset=["total_points", "score_official_xp", "score_xp"])
    rows: list[dict[str, Any]] = []
    for gw in sorted(int(value) for value in work["gw"].unique()):
        if gw < start or gw > gw_end:
            continue
        train = work.loc[work["gw"] < gw]
        if int(train["gw"].nunique()) < min_train or len(train) < min_train:
            continue
        coef = engine_coefficient(
            train["total_points"].to_numpy(float),
            train["score_official_xp"].to_numpy(float),
            train["score_xp"].to_numpy(float),
        )
        if coef is None:
            continue
        rows.append(
            {
                "season": season,
                "gw": gw,
                "coefficient": coef,
                "n_train": int(len(train)),
                "n_train_gws": int(train["gw"].nunique()),
            }
        )
    return rows
