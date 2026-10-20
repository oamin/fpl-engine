"""Props shape goal shares; ``M`` sets the team goal level.

Live GW rule (PI 2026-10-20 — **row 1** / props-shape):

Anytime markets quote **P(to score)** = P(G≥1). Convert to a Poisson mean
``λ = −log(1−p)`` and use it as a **relative intensity**, not an absolute
``E[G]`` pin:

1. ``G_team = α_G × xG_M`` from AH/OU/BTTS (or a rolled ``k`` from ``M``)
   is the **only** level.
2. Weight ``w_i = λ(p_i)`` when player ``i`` is covered and ``m>0``;
   else ``w_i =`` baseline ``E[G]_i = s_i × G_team``.
3. ``π_i = w_i / Σ w``, then ``E[G]_i = π_i × G_team`` (mean path) or
   ``E[G_i | k] = π_i × k`` (short-horizon roll).

Assists: **no** prop overlay (locked). Lift-only / hard absolute replace are
parked — both fight a mean pot and disagree with scoreline rolls from ``M``.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


def anytime_p_to_lambda(p: float) -> float:
    """P(≥1 goal) → Poisson mean (unique μ with 1−e^{−μ}=p)."""
    p = float(min(1.0 - 1e-9, max(0.0, p)))
    if p <= 0.0:
        return 0.0
    return float(-math.log(1.0 - p))


def prop_goal_pi(
    sG: np.ndarray,
    *,
    G_team: float,
    prop_p: Mapping[int, float],
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build compositional goal shares ``π`` from props + baseline.

    Covered indices use ``λ(p)`` as weight; uncovered keep baseline
    ``s × G_team``. Returns ``π`` with ``Σ π = 1`` (or zeros if empty).
    """
    s = np.asarray(sG, dtype=float).copy()
    n = s.size
    empty = {
        "G_team": float(G_team),
        "covered": 0.0,
        "residual": float(G_team),
        "n_props": 0,
        "prop_scale": 1.0,
        "capped": False,
        "n_lifted": 0,
        "mode": "props_shape_m_level",
        "sum_w": 0.0,
        "sum_w_covered": 0.0,
        "sum_w_uncovered": 0.0,
    }
    if n == 0:
        return s, empty

    s = np.maximum(s, 0.0)
    if s.sum() <= eps:
        s[:] = 1.0 / n
    else:
        s /= s.sum()

    G = float(max(0.0, G_team))
    baseline_eg = s * G

    covered_idx = sorted(int(i) for i in prop_p if 0 <= int(i) < n)
    if not covered_idx or G <= eps:
        return s, {**empty, "G_team": G, "residual": G, "sum_w": float(baseline_eg.sum())}

    w = baseline_eg.copy()
    for i in covered_idx:
        w[i] = anytime_p_to_lambda(float(prop_p[i]))

    sum_w = float(w.sum())
    if sum_w <= eps:
        return s, {**empty, "G_team": G, "residual": G, "n_props": len(covered_idx)}

    pi = w / sum_w
    covered_mass = float((pi[covered_idx] * G).sum())
    uncovered = np.ones(n, dtype=bool)
    uncovered[covered_idx] = False
    sum_w_cov = float(w[covered_idx].sum())
    sum_w_unc = float(w[uncovered].sum()) if uncovered.any() else 0.0
    # "capped" only as a diagnostic: raw covered λ-mass exceeds the pot.
    capped = sum_w_cov > G + eps
    prop_scale = float(G / sum_w) if sum_w > eps else 1.0

    return pi, {
        "G_team": G,
        "covered": covered_mass,
        "residual": float(max(0.0, G - covered_mass)),
        "n_props": len(covered_idx),
        "prop_scale": prop_scale,
        "capped": capped,
        "n_lifted": 0,  # retired; kept for column compat
        "mode": "props_shape_m_level",
        "sum_w": sum_w,
        "sum_w_covered": sum_w_cov,
        "sum_w_uncovered": sum_w_unc,
    }


def rescale_goal_shares(
    sG: np.ndarray,
    *,
    G_team: float,
    prop_p: Mapping[int, float],
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Props shape ``π``; caller multiplies by ``G_team`` (or rolled ``k``).

    Name kept for live call sites. Semantics: **props_shape_m_level**
    (not lift-only).
    """
    return prop_goal_pi(sG, G_team=G_team, prop_p=prop_p, eps=eps)


def allocate_goals_with_pi(
    goals_for: int | float,
    pi: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Short-horizon / roll path: ``E[G_i | k] = k · π_i``.

    ``goals_for`` is the integer (or expected) team goals from a draw of ``M``.
    ``π`` must already be compositional (from ``prop_goal_pi`` or baseline shares).
    """
    pi_a = np.asarray(pi, dtype=float)
    if pi_a.size == 0:
        return pi_a
    k = float(max(0.0, goals_for))
    s = np.maximum(pi_a, 0.0)
    total = float(s.sum())
    if total <= eps:
        return np.zeros_like(s)
    s = s / total
    return s * k
