"""Generative attack allocation on a rolled scoreline (short-horizon trial).

Locked stack (PI 2026-10-20):

1. ``k ∼ M`` (``match_outcome_roll``).
2. Each of ``k`` match goals: scorer ∼ ``π`` + ∅_G mass from ``α_G``.
3. If a player scored: assist ∼ ``ρ`` (scorer removed) + ∅_A mass from ``α_A``.

``π`` is props-shaped / baseline goal share; ``ρ`` is assist share only (no props).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from src.markets.lineup_roll import LineupWorld, gate_dice_weights
from src.markets.prop_shares import prop_goal_pi


EMPTY_G = "∅_G"
EMPTY_A = "∅_A"


@dataclass(frozen=True)
class GoalEvent:
    """One match goal and its FPL attribution draws."""

    scorer: str  # player name or EMPTY_G
    assister: str | None  # player name, EMPTY_A, or None if no FPL scorer


@dataclass(frozen=True)
class SideAttackRoll:
    """Attack draws for one club on one rolled scoreline."""

    team: str
    goals_for: int
    events: tuple[GoalEvent, ...]
    scorers: tuple[str, ...]  # FPL scorers only (no empties)
    assisters: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "team": self.team,
            "goals_for": self.goals_for,
            "events": [
                {"scorer": e.scorer, "assister": e.assister} for e in self.events
            ],
            "scorers": list(self.scorers),
            "assisters": list(self.assisters),
            "n_fpl_goals": len(self.scorers),
            "n_fpl_assists": len(self.assisters),
        }


@dataclass
class SideAttackDice:
    """Categorical weights for one club side."""

    names: list[str]
    pi: np.ndarray  # goal shares (compositional over players)
    rho: np.ndarray  # assist shares (compositional over players)
    alpha_g: float
    alpha_a: float
    prop_diag: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        rows = []
        for i, name in enumerate(self.names):
            rows.append(
                {
                    "player": name,
                    "pi": float(self.pi[i]),
                    "rho": float(self.rho[i]),
                }
            )
        rows.sort(key=lambda r: -r["pi"])
        return {
            "alpha_g": self.alpha_g,
            "alpha_a": self.alpha_a,
            "n": len(self.names),
            "top_pi": rows[:8],
            "prop": dict(self.prop_diag),
        }


def build_side_dice(
    names: Sequence[str],
    sG: np.ndarray | Sequence[float],
    sA: np.ndarray | Sequence[float],
    *,
    alpha_g: float,
    alpha_a: float,
    G_team: float,
    prop_p: Mapping[int, float] | None = None,
) -> SideAttackDice:
    """Build ``π`` (goals) and ``ρ`` (assists) for the dice."""
    names_l = [str(n) for n in names]
    sG_a = np.asarray(sG, dtype=float)
    sA_a = np.asarray(sA, dtype=float)
    if sG_a.size != len(names_l) or sA_a.size != len(names_l):
        raise ValueError("names / sG / sA length mismatch")
    prop_diag: dict[str, float] = {}
    if prop_p:
        pi, prop_diag = prop_goal_pi(sG_a, G_team=float(G_team), prop_p=prop_p)
    else:
        pi = np.maximum(sG_a, 0.0)
        z = float(pi.sum())
        pi = pi / z if z > 0 else np.full(len(names_l), 1.0 / max(len(names_l), 1))
    rho = np.maximum(sA_a, 0.0)
    zA = float(rho.sum())
    rho = rho / zA if zA > 0 else np.full(len(names_l), 1.0 / max(len(names_l), 1))
    return SideAttackDice(
        names=names_l,
        pi=np.asarray(pi, dtype=float),
        rho=np.asarray(rho, dtype=float),
        alpha_g=float(alpha_g),
        alpha_a=float(alpha_a),
        prop_diag=prop_diag,
    )


def _categorical(rng: Any, weights: np.ndarray) -> int:
    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.0)
    s = float(w.sum())
    if s <= 0:
        return int(rng.integers(0, len(w))) if hasattr(rng, "integers") else 0
    p = w / s
    if hasattr(rng, "random") and not hasattr(rng, "integers"):
        u = float(rng.random())
        return int(np.searchsorted(np.cumsum(p), u, side="right"))
    return int(np.asarray(rng.choice(len(p), p=p)))


def dice_for_lineup(dice: SideAttackDice, world: LineupWorld) -> SideAttackDice:
    """Renormalise ``π`` / ``ρ`` onto lineup-eligible players (m ≥ gate)."""
    pi = gate_dice_weights(dice.pi, world.eligible)
    rho = gate_dice_weights(dice.rho, world.eligible)
    return SideAttackDice(
        names=list(dice.names),
        pi=pi,
        rho=rho,
        alpha_g=dice.alpha_g,
        alpha_a=dice.alpha_a,
        prop_diag=dict(dice.prop_diag),
    )


def roll_side_attack(
    dice: SideAttackDice,
    goals_for: int,
    rng: Any,
    *,
    team: str = "",
    lineup: LineupWorld | None = None,
) -> SideAttackRoll:
    """Allocate ``goals_for`` match goals with α_G / α_A empty mass."""
    if lineup is not None:
        dice = dice_for_lineup(dice, lineup)
    k = int(max(0, goals_for))
    events: list[GoalEvent] = []
    scorers: list[str] = []
    assisters: list[str] = []
    ag = float(min(1.0, max(0.0, dice.alpha_g)))
    aa = float(min(1.0, max(0.0, dice.alpha_a)))
    n = len(dice.names)

    for _ in range(k):
        # Scorer dice: players × α_G + ∅_G × (1−α_G)
        w_g = np.zeros(n + 1)
        w_g[:n] = dice.pi * ag
        w_g[n] = 1.0 - ag
        idx = _categorical(rng, w_g)
        if idx >= n:
            events.append(GoalEvent(scorer=EMPTY_G, assister=None))
            continue
        scorer = dice.names[idx]
        scorers.append(scorer)

        # Assist dice: players × α_A (scorer out) + ∅_A × (1−α_A)
        w_a = np.zeros(n + 1)
        w_a[:n] = dice.rho * aa
        w_a[idx] = 0.0  # scorer cannot assist this goal
        w_a[n] = 1.0 - aa
        j = _categorical(rng, w_a)
        if j >= n:
            events.append(GoalEvent(scorer=scorer, assister=EMPTY_A))
        else:
            assister = dice.names[j]
            assisters.append(assister)
            events.append(GoalEvent(scorer=scorer, assister=assister))

    return SideAttackRoll(
        team=team,
        goals_for=k,
        events=tuple(events),
        scorers=tuple(scorers),
        assisters=tuple(assisters),
    )


def mean_side_from_dice(dice: SideAttackDice, goals_for: float) -> dict[str, float]:
    """E[FPL G/A] for a fixed ``k`` (diagnostic; no scorer≠assister coupling)."""
    k = float(max(0.0, goals_for))
    eg = {n: float(dice.pi[i] * dice.alpha_g * k) for i, n in enumerate(dice.names)}
    # Assist mean ignoring exclusion ≈ α_A · α_G · k · ρ (rough); exact needs the dice.
    ea = {n: float(dice.rho[i] * dice.alpha_a * dice.alpha_g * k) for i, n in enumerate(dice.names)}
    return {"E_G": eg, "E_A_approx": ea}
