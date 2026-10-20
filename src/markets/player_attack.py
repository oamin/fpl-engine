"""Player attack expectations from match ``M`` moments and compositional shares.

Second cut of the ``score_xp_v2`` decomposition (after ``match_outcome``):

- Team pots from the scoreline matrix: ``G_team = α_G · xG_M``,
  ``A_team = α_A · xG_M``.
- Player expectations: ``E[G_i] = sG_i · G_team``, ``E[A_i] = sA_i · A_team``.

Shares ``sG`` / ``sA`` are **not** built here — they come from same-season
appearance rates × minutes (``numagent_xp`` weights / share-SE shrink). This
module only multiplies shares by the M-derived pots.

FPL point conversion (MID goals = 5, assists = 3, …) is optional and sits
beside the expectations so callers can inspect raw ``E[G]`` / ``E[A]``
(player xG / xA in the score_xp_v2 sense) without folding in position points.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from src.rules.fpl_2026 import ASSIST_POINTS, GOAL_POINTS, normalize_position


@dataclass(frozen=True)
class TeamAttackPots:
    """FPL award pots for one club side, scaled from ``xG_M``."""

    xg: float
    alpha_g: float
    alpha_a: float
    G_team: float
    A_team: float
    d_G_team: float
    d_A_team: float
    rel_g: float
    rel_a: float

    def summary(self) -> dict[str, float]:
        return {
            "xg": self.xg,
            "alpha_g": self.alpha_g,
            "alpha_a": self.alpha_a,
            "G_team": self.G_team,
            "A_team": self.A_team,
            "d_G_team": self.d_G_team,
            "d_A_team": self.d_A_team,
            "rel_g": self.rel_g,
            "rel_a": self.rel_a,
        }


@dataclass(frozen=True)
class PlayerAttackExpectations:
    """Per-player ``E[G]`` / ``E[A]`` (and optional FPL attack xP)."""

    E_G: np.ndarray
    E_A: np.ndarray
    dE_G: np.ndarray
    dE_A: np.ndarray
    xp_goals: np.ndarray | None = None
    xp_assists: np.ndarray | None = None
    d_xp_goals: np.ndarray | None = None
    d_xp_assists: np.ndarray | None = None

    def summary(self) -> dict[str, float]:
        return {
            "sum_E_G": float(np.asarray(self.E_G).sum()),
            "sum_E_A": float(np.asarray(self.E_A).sum()),
            "n": float(np.asarray(self.E_G).size),
        }


def team_attack_pots(
    xg: float,
    *,
    alpha_g: float,
    alpha_a: float,
    d_alpha_g: float = 0.0,
    d_alpha_a: float = 0.0,
    d_xg: float = 0.0,
) -> TeamAttackPots:
    """``G_team = α_G · xG_M``, ``A_team = α_A · xG_M`` with relative SEs.

    Pot relative SE combines α uncertainty and optional ``d_xg`` from ``M``
    in quadrature: ``rel = sqrt((dα/α)² + (dxG/xG)²)``.
    """
    xg_f = float(xg)
    ag = float(alpha_g)
    aa = float(alpha_a)
    g_team = ag * xg_f
    a_team = aa * xg_f
    rel_alpha_g = abs(float(d_alpha_g) / ag) if ag else 0.0
    rel_alpha_a = abs(float(d_alpha_a) / aa) if aa else 0.0
    rel_xg = abs(float(d_xg) / xg_f) if xg_f else 0.0
    rel_g = math.sqrt(rel_alpha_g**2 + rel_xg**2)
    rel_a = math.sqrt(rel_alpha_a**2 + rel_xg**2)
    return TeamAttackPots(
        xg=xg_f,
        alpha_g=ag,
        alpha_a=aa,
        G_team=g_team,
        A_team=a_team,
        d_G_team=abs(g_team) * rel_g,
        d_A_team=abs(a_team) * rel_a,
        rel_g=rel_g,
        rel_a=rel_a,
    )


def player_attack_expectations(
    sG: np.ndarray | Sequence[float],
    sA: np.ndarray | Sequence[float],
    pots: TeamAttackPots,
    *,
    dsG: np.ndarray | Sequence[float] | None = None,
    dsA: np.ndarray | Sequence[float] | None = None,
) -> PlayerAttackExpectations:
    """``E[G_i] = sG_i · G_team``, ``E[A_i] = sA_i · A_team``.

    Uncertainty (method §7): ``ΔE/E = sqrt((Δs/s)² + (ΔG_team/G_team)²)``.
    """
    sG_a = np.asarray(sG, dtype=float)
    sA_a = np.asarray(sA, dtype=float)
    if sG_a.shape != sA_a.shape:
        raise ValueError(f"sG shape {sG_a.shape} != sA shape {sA_a.shape}")
    n = sG_a.size
    dsG_a = np.zeros(n) if dsG is None else np.asarray(dsG, dtype=float)
    dsA_a = np.zeros(n) if dsA is None else np.asarray(dsA, dtype=float)
    if dsG_a.shape != (n,) or dsA_a.shape != (n,):
        raise ValueError("dsG/dsA must match share length")

    eG = sG_a * float(pots.G_team)
    eA = sA_a * float(pots.A_team)
    dE_G = np.zeros(n)
    dE_A = np.zeros(n)
    for i in range(n):
        if eG[i] > 0 and sG_a[i] > 0:
            dE_G[i] = eG[i] * math.sqrt((dsG_a[i] / sG_a[i]) ** 2 + pots.rel_g**2)
        if eA[i] > 0 and sA_a[i] > 0:
            dE_A[i] = eA[i] * math.sqrt((dsA_a[i] / sA_a[i]) ** 2 + pots.rel_a**2)
    return PlayerAttackExpectations(E_G=eG, E_A=eA, dE_G=dE_G, dE_A=dE_A)


def attack_xp_from_expectations(
    expectations: PlayerAttackExpectations,
    positions: Sequence[object],
) -> PlayerAttackExpectations:
    """Map ``E[G]`` / ``E[A]`` to FPL goal / assist xP by position."""
    pos = [normalize_position("GKP" if str(p) == "GK" else str(p)) for p in positions]
    n = expectations.E_G.size
    if len(pos) != n:
        raise ValueError(f"positions length {len(pos)} != n={n}")
    goal_pts = np.array([float(GOAL_POINTS[p]) for p in pos], dtype=float)
    xp_goals = expectations.E_G * goal_pts
    d_xp_goals = expectations.dE_G * goal_pts
    xp_assists = expectations.E_A * float(ASSIST_POINTS)
    d_xp_assists = expectations.dE_A * float(ASSIST_POINTS)
    return PlayerAttackExpectations(
        E_G=expectations.E_G,
        E_A=expectations.E_A,
        dE_G=expectations.dE_G,
        dE_A=expectations.dE_A,
        xp_goals=xp_goals,
        xp_assists=xp_assists,
        d_xp_goals=d_xp_goals,
        d_xp_assists=d_xp_assists,
    )


def player_attack_from_m(
    sG: np.ndarray | Sequence[float],
    sA: np.ndarray | Sequence[float],
    *,
    xg: float,
    alpha_g: float,
    alpha_a: float,
    d_alpha_g: float = 0.0,
    d_alpha_a: float = 0.0,
    d_xg: float = 0.0,
    dsG: np.ndarray | Sequence[float] | None = None,
    dsA: np.ndarray | Sequence[float] | None = None,
    positions: Sequence[object] | None = None,
) -> tuple[TeamAttackPots, PlayerAttackExpectations]:
    """One-shot: ``xG_M`` + α + shares → pots and player attack expectations.

    When ``positions`` is set, also fills FPL ``xp_goals`` / ``xp_assists``.
    """
    pots = team_attack_pots(
        xg,
        alpha_g=alpha_g,
        alpha_a=alpha_a,
        d_alpha_g=d_alpha_g,
        d_alpha_a=d_alpha_a,
        d_xg=d_xg,
    )
    exp = player_attack_expectations(sG, sA, pots, dsG=dsG, dsA=dsA)
    if positions is not None:
        exp = attack_xp_from_expectations(exp, positions)
    return pots, exp
