"""Scoreline matrix from AH / OU / BTTS fair probabilities.

Fits a Dixon–Coles-style grid ``M[i,j] = P(home=i, away=j)`` so that the
matrix matches three market functionals. No shrinkage toward a neutral prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


GRID_MAX = 6


def _poisson_pmf_grid(lam: float, k_max: int = GRID_MAX) -> np.ndarray:
    """P(K=0..k_max-1) and P(K>=k_max) in the last bin."""
    lam = float(max(lam, 1e-6))
    vals = np.array([float(poisson.pmf(k, lam)) for k in range(k_max)], dtype=float)
    tail = float(max(0.0, 1.0 - vals.sum()))
    return np.concatenate([vals, np.array([tail])])


def dixon_coles_tau(i: int, j: int, lam_h: float, lam_a: float, rho: float) -> float:
    """Classic Dixon–Coles low-score adjustment factor."""
    if i == 0 and j == 0:
        return 1.0 - lam_h * lam_a * rho
    if i == 0 and j == 1:
        return 1.0 + lam_h * rho
    if i == 1 and j == 0:
        return 1.0 + lam_a * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def build_matrix(
    lam_h: float,
    lam_a: float,
    rho: float = 0.0,
    *,
    k_max: int = GRID_MAX,
) -> np.ndarray:
    """Return a (k_max+1)×(k_max+1) probability matrix (last index = k_max+)."""
    ph = _poisson_pmf_grid(lam_h, k_max)
    pa = _poisson_pmf_grid(lam_a, k_max)
    n = k_max + 1
    raw = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            # Tail bins use unadjusted product; DC τ only on exact 0/1 scores.
            ii = i if i < k_max else k_max  # treat 6+ as 6 for τ (no-op)
            jj = j if j < k_max else k_max
            tau = 1.0
            if i < k_max and j < k_max:
                tau = dixon_coles_tau(i, j, lam_h, lam_a, rho)
            raw[i, j] = max(0.0, tau) * ph[i] * pa[j]
    total = float(raw.sum())
    if total <= 0.0:
        raw[:] = 1.0 / raw.size
        return raw
    return raw / total


def prob_btts(matrix: np.ndarray) -> float:
    return float(matrix[1:, 1:].sum())


p_btts = prob_btts


def prob_over_25(matrix: np.ndarray) -> float:
    """P(total goals >= 3), treating the last bin as exactly GRID_MAX for the sum.

    For the edge bin (GRID_MAX+), count as GRID_MAX goals on that side so the
    over line is slightly conservative on the truncated tail.
    """
    n = matrix.shape[0]
    k_max = n - 1
    total = 0.0
    for i in range(n):
        gi = i if i < k_max else k_max
        for j in range(n):
            gj = j if j < k_max else k_max
            if gi + gj >= 3:
                total += float(matrix[i, j])
    return float(total)


# Back-compat alias used in early call sites / tests.
p_over_25 = prob_over_25


def _is_quarter_line(line: float) -> bool:
    """True for ±0.25, ±0.75, ±1.25, … (not halves or integers)."""
    fourths = line * 4.0
    if abs(fourths - round(fourths)) > 1e-9:
        return False
    return int(round(fourths)) % 2 != 0


def p_ah_cover(matrix: np.ndarray, line: float) -> float:
    """Expected win fraction for backing home on Asian handicap ``line``.

    ``line`` is the home handicap (negative if home gives goals). Settlement:
    - integer line: win / push / lose
    - half line (±0.5, ±1.5, …): win / lose only
    - quarter line (±0.25, ±0.75, …): average of the two neighbouring half lines
    """
    line = float(line)
    if _is_quarter_line(line):
        # e.g. -0.25 → avg(0.0, -0.5); -0.75 → avg(-0.5, -1.0)
        lo = math.floor(line * 2.0) / 2.0
        hi = math.ceil(line * 2.0) / 2.0
        if abs(lo - hi) < 1e-12:
            hi = lo + 0.5
        return 0.5 * (p_ah_cover(matrix, lo) + p_ah_cover(matrix, hi))

    n = matrix.shape[0]
    k_max = n - 1
    win = 0.0
    push = 0.0
    integer_line = abs(line - round(line)) < 1e-9
    for i in range(n):
        gi = float(i if i < k_max else k_max)
        for j in range(n):
            gj = float(j if j < k_max else k_max)
            # Home handicap applied to home goals for the margin.
            margin = (gi + line) - gj
            p = float(matrix[i, j])
            if integer_line:
                if margin > 1e-12:
                    win += p
                elif abs(margin) <= 1e-12:
                    push += p
            else:
                if margin > 1e-12:
                    win += p
    # Fair cover probability for pricing: win + 0.5 * push (void returns stake).
    return float(win + 0.5 * push)


@dataclass(frozen=True)
class ScorelineFit:
    lam_h: float
    lam_a: float
    rho: float
    matrix: np.ndarray
    fitted: dict[str, float]
    targets: dict[str, float]
    residual: dict[str, float]
    sse: float
    success: bool
    message: str

    def summary(self) -> dict[str, Any]:
        return {
            "lam_h": self.lam_h,
            "lam_a": self.lam_a,
            "rho": self.rho,
            "e_total": self.lam_h + self.lam_a,
            "fitted": dict(self.fitted),
            "targets": dict(self.targets),
            "residual": dict(self.residual),
            "sse": self.sse,
            "success": self.success,
            "message": self.message,
            "p_cs_home": float(self.matrix[:, 0].sum()),
            "p_cs_away": float(self.matrix[0, :].sum()),
            "p_home_win": float(
                sum(
                    self.matrix[i, j]
                    for i in range(self.matrix.shape[0])
                    for j in range(self.matrix.shape[1])
                    if (i if i < GRID_MAX else GRID_MAX) > (j if j < GRID_MAX else GRID_MAX)
                )
            ),
        }


def _clip01(x: float) -> float:
    return float(min(1.0 - 1e-6, max(1e-6, x)))


def fit_scoreline(
    *,
    p_ah: float,
    ah_line: float,
    p_over_25: float,
    p_btts: float,
    weights: tuple[float, float, float] = (1.0, 1.0, 1.0),
    k_max: int = GRID_MAX,
) -> ScorelineFit:
    """Fit ``(lam_h, lam_a, rho)`` to three fair market probabilities."""
    targets = {
        "p_ah": _clip01(p_ah),
        "ah_line": float(ah_line),
        "p_over_25": _clip01(p_over_25),
        "p_btts": _clip01(p_btts),
    }
    w_ah, w_ou, w_btts = weights

    # Independent-Poisson warm start from OU + a mild favourite tilt from AH.
    e_total = 2.45 + 1.10 * (2.0 * targets["p_over_25"] - 1.0)
    e_total = float(min(4.5, max(1.2, e_total)))
    # AH line negative ⇒ home favourite ⇒ larger lam_h.
    tilt = 0.35 * (2.0 * targets["p_ah"] - 1.0) - 0.15 * float(ah_line)
    lam_h0 = 0.5 * e_total + 0.5 * tilt
    lam_a0 = e_total - lam_h0
    lam_h0 = float(min(3.2, max(0.25, lam_h0)))
    lam_a0 = float(min(3.2, max(0.25, lam_a0)))
    x0 = np.array([math.log(lam_h0), math.log(lam_a0), 0.0], dtype=float)

    def unpack(x: np.ndarray) -> tuple[float, float, float]:
        lam_h = float(math.exp(x[0]))
        lam_a = float(math.exp(x[1]))
        # rho in (-0.2, 0.2) roughly, soft via tanh
        rho = 0.25 * math.tanh(float(x[2]))
        return lam_h, lam_a, rho

    def residuals(x: np.ndarray) -> np.ndarray:
        lam_h, lam_a, rho = unpack(x)
        mat = build_matrix(lam_h, lam_a, rho, k_max=k_max)
        return np.array(
            [
                math.sqrt(w_ah) * (p_ah_cover(mat, ah_line) - targets["p_ah"]),
                math.sqrt(w_ou) * (prob_over_25(mat) - targets["p_over_25"]),
                math.sqrt(w_btts) * (prob_btts(mat) - targets["p_btts"]),
            ],
            dtype=float,
        )

    def objective(x: np.ndarray) -> float:
        r = residuals(x)
        return float(np.dot(r, r))

    result = minimize(objective, x0, method="L-BFGS-B")
    lam_h, lam_a, rho = unpack(result.x)
    mat = build_matrix(lam_h, lam_a, rho, k_max=k_max)
    fitted = {
        "p_ah": p_ah_cover(mat, ah_line),
        "p_over_25": prob_over_25(mat),
        "p_btts": prob_btts(mat),
    }
    residual = {
        "p_ah": fitted["p_ah"] - targets["p_ah"],
        "p_over_25": fitted["p_over_25"] - targets["p_over_25"],
        "p_btts": fitted["p_btts"] - targets["p_btts"],
    }
    # SSE only over markets with positive weight (BTTS often absent from football-data).
    sse = float(
        w_ah * residual["p_ah"] ** 2
        + w_ou * residual["p_over_25"] ** 2
        + w_btts * residual["p_btts"] ** 2
    )
    return ScorelineFit(
        lam_h=lam_h,
        lam_a=lam_a,
        rho=rho,
        matrix=mat,
        fitted=fitted,
        targets={k: targets[k] for k in ("p_ah", "p_over_25", "p_btts")},
        residual=residual,
        sse=sse,
        success=bool(result.success) and sse < 0.01,
        message=str(result.message),
    )


def fair_pair_from_decimals(over_or_yes: float, under_or_no: float) -> tuple[float, float]:
    """Multiplicative de-vig for a two-way market → (p_first, p_second)."""
    from src.markets.shin import shin_pair

    return shin_pair(over_or_yes, under_or_no)


def team_moments(matrix: np.ndarray) -> dict[str, float]:
    """Extract side moments used by player XP: xG, xCS, xGC events."""
    n = matrix.shape[0]
    k_max = n - 1
    xg_h = xg_a = xcs_h = xcs_a = xgc_h = xgc_a = 0.0
    for i in range(n):
        gi = float(i if i < k_max else k_max)
        for j in range(n):
            gj = float(j if j < k_max else k_max)
            p = float(matrix[i, j])
            xg_h += gi * p
            xg_a += gj * p
            if j == 0:
                xcs_h += p
            if i == 0:
                xcs_a += p
            xgc_h += math.floor(gj / 2.0) * p
            xgc_a += math.floor(gi / 2.0) * p
    return {
        "xg_home": xg_h,
        "xg_away": xg_a,
        "xcs_home": xcs_h,
        "xcs_away": xcs_a,
        "xgc_home": xgc_h,
        "xgc_away": xgc_a,
    }
