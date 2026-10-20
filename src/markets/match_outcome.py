"""Match outcome distribution ``M`` and rolls inside it.

First cut of the ``score_xp_v2`` decomposition:

- ``match_outcome_probs`` — Dixon–Coles grid ``M[i,j] = P(home=i, away=j)``
  fitted to bookie AH / OU / BTTS fair probabilities (BTTS optional).
- ``match_outcome_roll`` — one scoreline drawn from ``M`` (every cell in the
  support is eligible; probability mass is exactly ``M``).

Downstream player xP still reads side moments from ``M``; cascade short/long
should eventually call these instead of ad-hoc λ ± band Poisson draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.markets.scoreline import (
    GRID_MAX,
    ScorelineFit,
    build_matrix,
    fair_pair_from_decimals,
    fit_scoreline,
    team_moments,
)


@dataclass(frozen=True)
class MatchOutcomeProbs:
    """Bookie- (or λ-) defined match scoreline distribution ``M``."""

    matrix: np.ndarray
    lam_h: float
    lam_a: float
    rho: float
    ah_line: float | None
    markets: tuple[str, ...]
    weights: tuple[float, float, float]
    fit_success: bool
    sse: float
    targets: dict[str, float]
    fitted: dict[str, float]
    moments: dict[str, float]

    @property
    def M(self) -> np.ndarray:
        return self.matrix

    def summary(self) -> dict[str, Any]:
        return {
            "lam_h": self.lam_h,
            "lam_a": self.lam_a,
            "rho": self.rho,
            "ah_line": self.ah_line,
            "markets": list(self.markets),
            "weights": list(self.weights),
            "fit_success": self.fit_success,
            "sse": self.sse,
            "targets": dict(self.targets),
            "fitted": dict(self.fitted),
            "moments": dict(self.moments),
            "shape": list(self.matrix.shape),
            "mass": float(self.matrix.sum()),
        }


@dataclass(frozen=True)
class MatchOutcomeRoll:
    """One scoreline drawn from ``M``."""

    goals_home: int
    goals_away: int
    p: float
    i: int
    j: int
    home_tail: bool
    away_tail: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "goals_home": self.goals_home,
            "goals_away": self.goals_away,
            "p": self.p,
            "i": self.i,
            "j": self.j,
            "home_tail": self.home_tail,
            "away_tail": self.away_tail,
        }


def match_outcome_probs(
    *,
    p_ah: float,
    ah_line: float,
    p_over_25: float,
    p_btts: float | None = None,
    btts_available: bool | None = None,
    k_max: int = GRID_MAX,
) -> MatchOutcomeProbs:
    """Fit ``M`` to AH + OU 2.5 (+ BTTS when available).

    When ``p_btts`` is ``None`` or ``btts_available`` is false, BTTS weight is
    0 (football-data / unpriced live path). Otherwise weight 1.
    """
    use_btts = bool(
        p_btts is not None
        if btts_available is None
        else btts_available
    )
    if use_btts and p_btts is None:
        raise ValueError("btts_available requires p_btts")
    p_b = 0.5 if p_btts is None else float(p_btts)
    weights = (1.0, 1.0, 1.0 if use_btts else 0.0)
    fit = fit_scoreline(
        p_ah=float(p_ah),
        ah_line=float(ah_line),
        p_over_25=float(p_over_25),
        p_btts=p_b,
        weights=weights,
        k_max=int(k_max),
    )
    markets: list[str] = ["AH", "OU2.5"]
    if use_btts:
        markets.append("BTTS")
    return _from_fit(fit, ah_line=float(ah_line), markets=tuple(markets), weights=weights)


def match_outcome_probs_from_decimals(
    *,
    ah_home_dec: float,
    ah_away_dec: float,
    ah_line: float,
    over_dec: float,
    under_dec: float,
    btts_yes_dec: float | None = None,
    btts_no_dec: float | None = None,
    k_max: int = GRID_MAX,
) -> MatchOutcomeProbs:
    """Bookie decimal prices → Shin fair probs → ``match_outcome_probs``."""
    p_ah, _ = fair_pair_from_decimals(float(ah_home_dec), float(ah_away_dec))
    p_over, _ = fair_pair_from_decimals(float(over_dec), float(under_dec))
    p_btts: float | None = None
    if btts_yes_dec is not None and btts_no_dec is not None:
        p_btts, _ = fair_pair_from_decimals(float(btts_yes_dec), float(btts_no_dec))
    return match_outcome_probs(
        p_ah=p_ah,
        ah_line=float(ah_line),
        p_over_25=p_over,
        p_btts=p_btts,
        k_max=k_max,
    )


def match_outcome_probs_from_lambdas(
    lam_h: float,
    lam_a: float,
    rho: float = 0.0,
    *,
    k_max: int = GRID_MAX,
    ah_line: float | None = None,
) -> MatchOutcomeProbs:
    """Build ``M`` directly from Dixon–Coles λ (Opta / scenario path)."""
    mat = build_matrix(float(lam_h), float(lam_a), float(rho), k_max=int(k_max))
    mom = team_moments(mat)
    return MatchOutcomeProbs(
        matrix=mat,
        lam_h=float(lam_h),
        lam_a=float(lam_a),
        rho=float(rho),
        ah_line=ah_line,
        markets=("lambda",),
        weights=(0.0, 0.0, 0.0),
        fit_success=True,
        sse=0.0,
        targets={},
        fitted={},
        moments=mom,
    )


def match_outcome_roll(
    probs: MatchOutcomeProbs | np.ndarray,
    rng: Any,
    *,
    k_max: int = GRID_MAX,
) -> MatchOutcomeRoll:
    """Draw one ``(goals_home, goals_away)`` from ``M``.

    Every outcome cell with positive mass is in the window; the draw is
    categorical over the flattened matrix (last index = ``k_max+`` goals).
    """
    mat = probs.matrix if isinstance(probs, MatchOutcomeProbs) else np.asarray(probs, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("M must be a square scoreline matrix")
    flat = mat.ravel()
    total = float(flat.sum())
    if total <= 0.0:
        raise ValueError("M has zero mass")
    p = flat / total
    # Support both random.Random and numpy Generator.
    if hasattr(rng, "random") and not hasattr(rng, "integers"):
        u = float(rng.random())
        cdf = np.cumsum(p)
        idx = int(np.searchsorted(cdf, u, side="right"))
        idx = min(idx, p.size - 1)
    else:
        idx = int(np.asarray(rng.choice(p.size, p=p)))
    n = mat.shape[0]
    i, j = divmod(idx, n)
    home_tail = i >= k_max
    away_tail = j >= k_max
    gh = int(k_max if home_tail else i)
    ga = int(k_max if away_tail else j)
    return MatchOutcomeRoll(
        goals_home=gh,
        goals_away=ga,
        p=float(mat[i, j] / total),
        i=int(i),
        j=int(j),
        home_tail=bool(home_tail),
        away_tail=bool(away_tail),
    )


def match_outcome_roll_many(
    probs: MatchOutcomeProbs | np.ndarray,
    n: int,
    rng: Any,
    *,
    k_max: int = GRID_MAX,
) -> list[MatchOutcomeRoll]:
    """``n`` independent rolls from the same ``M``."""
    return [match_outcome_roll(probs, rng, k_max=k_max) for _ in range(int(n))]


def _from_fit(
    fit: ScorelineFit,
    *,
    ah_line: float,
    markets: Sequence[str],
    weights: tuple[float, float, float],
) -> MatchOutcomeProbs:
    return MatchOutcomeProbs(
        matrix=np.asarray(fit.matrix, dtype=float),
        lam_h=float(fit.lam_h),
        lam_a=float(fit.lam_a),
        rho=float(fit.rho),
        ah_line=float(ah_line),
        markets=tuple(markets),
        weights=weights,
        fit_success=bool(fit.success),
        sse=float(fit.sse),
        targets=dict(fit.targets),
        fitted=dict(fit.fitted),
        moments=team_moments(fit.matrix),
    )


def probs_close_to_uniform_margin(
    probs: MatchOutcomeProbs,
    rolls: Sequence[MatchOutcomeRoll],
) -> dict[str, float]:
    """Diagnostic: empirical mean goals vs ``M`` moments (smoke only)."""
    if not rolls:
        return {"n": 0.0}
    gh = np.array([r.goals_home for r in rolls], dtype=float)
    ga = np.array([r.goals_away for r in rolls], dtype=float)
    return {
        "n": float(len(rolls)),
        "mean_gh": float(gh.mean()),
        "mean_ga": float(ga.mean()),
        "xg_home": float(probs.moments["xg_home"]),
        "xg_away": float(probs.moments["xg_away"]),
        "abs_err_h": float(abs(gh.mean() - probs.moments["xg_home"])),
        "abs_err_a": float(abs(ga.mean() - probs.moments["xg_away"])),
    }
