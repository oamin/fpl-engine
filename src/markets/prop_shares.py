"""Rescale team goal shares from player anytime-scorer props.

Live GW rule (PI 2026-10-16, lift-only):

Anytime markets quote **P(to score)** = P(G≥1), not E[G]. Poisson would
force E[G]=λ=−log(1−p), which can cut a star with baseline E[G]>1. So props
**lift** depressed baselines; they do not replace a higher share-based mean.

1. ``G_team = α_G × xG_M`` from AH/OU/BTTS is binding (Σ E[G] = G).
2. Baseline E[G]_i = s_i × G_team.
3. Covered players target ``max(baseline_i, λ(p_i))``. Extra mass is taken
   from **uncovered** teammates first (props precedence on the margin).
4. If lifts still cannot be funded (uncovered exhausted) and Σ would exceed G,
   scale all players by ``G / Σ``.

Assists unchanged.
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


def rescale_goal_shares(
    sG: np.ndarray,
    *,
    G_team: float,
    prop_p: Mapping[int, float],
    eps: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Lift-only prop overlay; returns shares summing to 1 with Σ E[G]=G_team."""
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
        "mode": "lift_only",
    }
    if n == 0:
        return s, empty
    s = np.maximum(s, 0.0)
    if s.sum() <= eps:
        s[:] = 1.0 / n
    else:
        s /= s.sum()

    G = float(max(0.0, G_team))
    eg = s * G
    covered_idx = sorted(int(i) for i in prop_p if 0 <= int(i) < n)
    if not covered_idx or G <= eps:
        return s, {**empty, "G_team": G, "residual": G}

    target = eg.copy()
    for i in covered_idx:
        lam = anytime_p_to_lambda(float(prop_p[i]))
        if lam > target[i]:
            target[i] = lam

    need = float(np.maximum(target - eg, 0.0).sum())
    n_lifted = int(np.sum(target > eg + eps))

    if need <= eps:
        # No lift — keep baseline (props did not raise anyone).
        return s, {
            "G_team": G,
            "covered": float(eg[covered_idx].sum()),
            "residual": float(G - eg[covered_idx].sum()),
            "n_props": len(covered_idx),
            "prop_scale": 1.0,
            "capped": False,
            "n_lifted": 0,
            "mode": "lift_only",
        }

    uncovered = np.ones(n, dtype=bool)
    uncovered[covered_idx] = False
    pool = float(eg[uncovered].sum()) if uncovered.any() else 0.0

    eg_out = eg.copy()
    prop_scale = 1.0
    capped = False

    if pool >= need - eps:
        # Fund lifts entirely from uncovered baseline mass.
        eg_out[covered_idx] = target[covered_idx]
        if pool > eps:
            shrink = (pool - need) / pool
            eg_out[uncovered] = eg[uncovered] * shrink
        else:
            eg_out[uncovered] = 0.0
    elif pool > eps:
        # Partial lifts: drain uncovered, allocate pool ∝ need.
        eg_out[uncovered] = 0.0
        eg_out[covered_idx] = eg[covered_idx]
        needs = np.maximum(target - eg, 0.0)
        eg_out += needs * (pool / float(needs.sum()))
    else:
        # No uncovered mass (all covered or zero baseline elsewhere): apply
        # targets then pot-cap so Σ ≤ G_team.
        eg_out = target.copy()
        raw = float(eg_out.sum())
        if raw > G + eps:
            prop_scale = G / raw
            eg_out *= prop_scale
            capped = True

    total = float(eg_out.sum())
    if total <= eps:
        return s, {
            "G_team": G,
            "covered": 0.0,
            "residual": G,
            "n_props": len(covered_idx),
            "prop_scale": prop_scale,
            "capped": capped,
            "n_lifted": n_lifted,
            "mode": "lift_only",
        }
    eg_out = eg_out * (G / total)
    out = eg_out / G
    covered_mass = float(eg_out[covered_idx].sum())
    return out, {
        "G_team": G,
        "covered": covered_mass,
        "residual": float(max(0.0, G - covered_mass)),
        "n_props": len(covered_idx),
        "prop_scale": float(prop_scale),
        "capped": capped,
        "n_lifted": n_lifted,
        "mode": "lift_only",
        "need": need,
        "uncovered_pool": pool,
    }
