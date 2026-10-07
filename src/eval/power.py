"""Smallest paired effect a 20-gameweek cluster interval can detect.

Residuals are closed-season gameweek deltas with their mean removed, so the
null has mean zero and the observed week-to-week spread. This is the power
of that interval. It is not a second look at the live encompassing test.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.eval.decision_spec import (
    POWER_BOOTSTRAP,
    POWER_GRID_MAX,
    POWER_LEVEL,
    POWER_SEED,
    POWER_SIMS,
    POWER_STEP,
    POWER_WEEKS,
)
from src.eval.gates import cluster_interval


def _interval(sample: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    summary = cluster_interval(
        {"sim": np.asarray(sample, dtype=float)},
        seasons=("sim",),
        n_boot=n_boot,
        seed=seed,
    )
    return float(summary["lo"]), float(summary["hi"])


def _detects(sample: np.ndarray, n_boot: int, seed: int) -> bool:
    lo, _hi = _interval(sample, n_boot, seed)
    return lo > 0.0


def _draws(residuals: np.ndarray, n_sims: int, n_weeks: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.choice(residuals, size=(n_sims, n_weeks), replace=True)


def power_at(
    residuals: np.ndarray,
    effect: float,
    *,
    n_weeks: int,
    n_sims: int,
    n_boot: int,
    seed: int,
) -> float:
    """Share of draws whose 95% interval lies entirely above zero."""
    draws = _draws(residuals, n_sims, n_weeks, seed) + float(effect)
    hits = 0
    for row in draws:
        if _detects(row, n_boot, seed):
            hits += 1
    return hits / float(n_sims)


def null_half_widths(
    residuals: np.ndarray,
    *,
    n_weeks: int,
    n_sims: int,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    draws = _draws(residuals, n_sims, n_weeks, seed)
    widths = np.empty(n_sims, dtype=float)
    for index, row in enumerate(draws):
        lo, hi = _interval(row, n_boot, seed)
        widths[index] = (hi - lo) / 2.0
    return widths


def minimum_detectable(
    deltas: np.ndarray,
    *,
    n_weeks: int = POWER_WEEKS,
    power: float = POWER_LEVEL,
    n_boot: int = POWER_BOOTSTRAP,
    n_sims: int = POWER_SIMS,
    step: float = POWER_STEP,
    seed: int = POWER_SEED,
    grid_max: float = POWER_GRID_MAX,
) -> dict[str, Any]:
    """Smallest positive effect on the grid with power at least `power`."""
    observed = np.asarray(deltas, dtype=float)
    if observed.size < 2 or not np.isfinite(observed).all():
        raise ValueError("power needs a finite series of gameweek deltas")
    residuals = observed - observed.mean()
    widths = null_half_widths(
        residuals, n_weeks=n_weeks, n_sims=n_sims, n_boot=n_boot, seed=seed
    )
    found: float | None = None
    found_power = 0.0
    effect = step
    while effect <= grid_max + 1e-12:
        level = power_at(
            residuals,
            effect,
            n_weeks=n_weeks,
            n_sims=n_sims,
            n_boot=n_boot,
            seed=seed,
        )
        if level >= power:
            found = float(effect)
            found_power = float(level)
            break
        effect = round(effect + step, 10)
    return {
        "mde": found,
        "power_at_mde": found_power,
        "power_target": float(power),
        "median_null_half_width": float(np.median(widths)),
        "n_weeks": int(n_weeks),
        "n_deltas": int(observed.size),
        "residual_sd": float(residuals.std(ddof=1)),
        "n_sims": int(n_sims),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "step": float(step),
        "grid_max": float(grid_max),
    }
