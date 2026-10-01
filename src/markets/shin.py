"""Shin (1993) de-vig for complete market partitions."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import root_scalar


def shin_devig(raw_implied: np.ndarray | list[float]) -> np.ndarray:
    """Return fair probs summing to 1. Falls back to multiplicative norm."""
    pi = np.asarray(raw_implied, dtype=float)
    if pi.ndim != 1 or pi.size < 2 or np.any(pi <= 0) or not np.all(np.isfinite(pi)):
        raise ValueError("Shin needs a positive finite implied-prob vector")
    overround = float(pi.sum())
    if overround <= 1.0 + 1e-12:
        return pi / overround

    def fair(z: float) -> np.ndarray:
        return (np.sqrt(z * z + 4.0 * (1.0 - z) * pi * pi) - z) / (2.0 * (1.0 - z))

    def objective(z: float) -> float:
        return float(fair(z).sum() - 1.0)

    try:
        z_hat = float(root_scalar(objective, bracket=(1e-6, 1.0 - 1e-6)).root)
        probs = fair(z_hat)
        s = float(probs.sum())
        if s <= 0 or not np.all(np.isfinite(probs)):
            raise RuntimeError("Shin non-finite")
        return probs / s
    except (ValueError, RuntimeError):
        return pi / overround


def shin_from_decimals(decimals: list[float]) -> list[float]:
    raw = [1.0 / d for d in decimals]
    return [float(x) for x in shin_devig(raw)]


def shin_1x2(home: float, draw: float, away: float) -> tuple[float, float, float]:
    p = shin_from_decimals([home, draw, away])
    return p[0], p[1], p[2]


def shin_pair(a: float, b: float) -> tuple[float, float]:
    p = shin_from_decimals([a, b])
    return p[0], p[1]


def implied_safe(decimal: float | None) -> float | None:
    if decimal is None or not math.isfinite(decimal) or decimal <= 1.0:
        return None
    return 1.0 / decimal
