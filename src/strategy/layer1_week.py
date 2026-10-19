"""Layer 1 — single-week joint MILP on absolute μ.

Optimises a legal 15 + starting XI + captain to maximise

    Σ_i μ_i · start_i  +  Σ_i μ_i · captain_i

No ownership tilt, SE, transfers, chips, or multi-week horizon.
Does not modify ``score_xp_v2``. Does not replace the published ``score_xp`` climb.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.rules.fpl_2026 import (
    BUDGET_TENTHS,
    MAX_DEF,
    MAX_FWD,
    MAX_MID,
    MAX_PER_CLUB,
    MIN_DEF,
    MIN_FWD,
    MIN_MID,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    normalize_position,
    squad_legal,
    xi_legal,
)

REQUIRED_COLS = ("player_id", "position", "club", "value", "mu")


@dataclass(frozen=True)
class WeekPlan:
    """Layer-1 solution for one deadline week."""

    squad: pd.DataFrame
    xi: pd.DataFrame
    captain_id: str
    formation: tuple[int, int, int]
    objective: float
    xi_mu: float
    captain_mu: float
    cost: int
    status: str

    def summary(self) -> dict[str, Any]:
        return {
            "n_squad": int(len(self.squad)),
            "n_xi": int(len(self.xi)),
            "captain_id": self.captain_id,
            "formation": f"1-{self.formation[0]}-{self.formation[1]}-{self.formation[2]}",
            "objective": float(self.objective),
            "xi_mu": float(self.xi_mu),
            "captain_mu": float(self.captain_mu),
            "cost": int(self.cost),
            "status": self.status,
        }


def _prepare_pool(pool: pd.DataFrame, *, mu_col: str) -> pd.DataFrame:
    df = pool.copy()
    if "player_id" not in df.columns:
        if "element" in df.columns:
            df["player_id"] = df["element"].astype(str)
        elif "name" in df.columns:
            df["player_id"] = df["name"].astype(str)
        else:
            raise ValueError("pool needs player_id, element, or name")
    df["player_id"] = df["player_id"].astype(str)
    if mu_col not in df.columns:
        raise ValueError(f"missing μ column {mu_col!r}")
    df["mu"] = pd.to_numeric(df[mu_col], errors="coerce")
    df = df.loc[np.isfinite(df["mu"])].copy()
    if "club" not in df.columns:
        for alt in ("team_norm", "team", "team_short"):
            if alt in df.columns:
                df["club"] = df[alt].astype(str)
                break
        else:
            raise ValueError("pool needs club / team_norm / team")
    else:
        df["club"] = df["club"].astype(str)
    if "value" not in df.columns:
        for alt in ("now_cost", "price"):
            if alt in df.columns:
                df["value"] = pd.to_numeric(df[alt], errors="coerce")
                break
        else:
            raise ValueError("pool needs value / now_cost / price (tenths of £m)")
    else:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["position"] = df["position"].map(normalize_position)
    df = df.drop_duplicates("player_id", keep="first").reset_index(drop=True)
    if len(df) < SQUAD_SIZE:
        raise RuntimeError(f"need ≥{SQUAD_SIZE} candidates with finite μ, got {len(df)}")
    for pos, need in SQUAD_QUOTA.items():
        if int((df["position"] == pos).sum()) < need:
            raise RuntimeError(f"pool has fewer than {need} {pos} candidates")
    return df


def solve_week(
    pool: pd.DataFrame,
    *,
    mu_col: str = "score_xp_v2",
    budget: int = BUDGET_TENTHS,
    max_per_club: int = MAX_PER_CLUB,
    time_limit: float = 30.0,
) -> WeekPlan:
    """Joint MILP: legal 15 + XI + captain maximising XI μ + captain μ."""
    df = _prepare_pool(pool, mu_col=mu_col)
    n = len(df)
    mu = df["mu"].to_numpy(float)
    value = df["value"].to_numpy(float)
    pos = df["position"].to_numpy(str)
    club = df["club"].to_numpy(str)

    # Decision vector z = [x | s | c] length 3n
    # x = in squad, s = start, c = captain
    nx, ns, nc = n, n, n

    def blk(kind: str, i: int) -> int:
        if kind == "x":
            return i
        if kind == "s":
            return nx + i
        return nx + ns + i

    # Objective: maximise Σ μ s + Σ μ c  → milp minimises, so negate
    c_obj = np.zeros(nx + ns + nc)
    for i in range(n):
        c_obj[blk("s", i)] = -mu[i]
        c_obj[blk("c", i)] = -mu[i]

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add_row(coeff: np.ndarray, low: float, high: float) -> None:
        rows.append(coeff)
        lo.append(low)
        hi.append(high)

    # Σ x = 15
    r = np.zeros(nx + ns + nc)
    for i in range(n):
        r[blk("x", i)] = 1.0
    add_row(r, float(SQUAD_SIZE), float(SQUAD_SIZE))

    # Σ s = 11
    r = np.zeros(nx + ns + nc)
    for i in range(n):
        r[blk("s", i)] = 1.0
    add_row(r, 11.0, 11.0)

    # Σ c = 1
    r = np.zeros(nx + ns + nc)
    for i in range(n):
        r[blk("c", i)] = 1.0
    add_row(r, 1.0, 1.0)

    # Position quotas on squad
    for p, need in SQUAD_QUOTA.items():
        r = np.zeros(nx + ns + nc)
        for i in range(n):
            if pos[i] == p:
                r[blk("x", i)] = 1.0
        add_row(r, float(need), float(need))

    # Formation on XI: 1 GKP; DEF/MID/FWD in official bounds
    r = np.zeros(nx + ns + nc)
    for i in range(n):
        if pos[i] == "GKP":
            r[blk("s", i)] = 1.0
    add_row(r, 1.0, 1.0)

    r = np.zeros(nx + ns + nc)
    for i in range(n):
        if pos[i] == "DEF":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_DEF), float(MAX_DEF))

    r = np.zeros(nx + ns + nc)
    for i in range(n):
        if pos[i] == "MID":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_MID), float(MAX_MID))

    r = np.zeros(nx + ns + nc)
    for i in range(n):
        if pos[i] == "FWD":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_FWD), float(MAX_FWD))

    # Club cap on squad
    for cl in sorted(set(club.tolist())):
        r = np.zeros(nx + ns + nc)
        for i in range(n):
            if club[i] == cl:
                r[blk("x", i)] = 1.0
        add_row(r, 0.0, float(max_per_club))

    # Budget on squad
    r = np.zeros(nx + ns + nc)
    for i in range(n):
        r[blk("x", i)] = value[i]
    add_row(r, -np.inf, float(budget))

    # Linking: s ≤ x, c ≤ s  →  s - x ≤ 0, c - s ≤ 0
    for i in range(n):
        r = np.zeros(nx + ns + nc)
        r[blk("s", i)] = 1.0
        r[blk("x", i)] = -1.0
        add_row(r, -np.inf, 0.0)
        r = np.zeros(nx + ns + nc)
        r[blk("c", i)] = 1.0
        r[blk("s", i)] = -1.0
        add_row(r, -np.inf, 0.0)

    A = np.vstack(rows)
    res = milp(
        c=c_obj,
        constraints=LinearConstraint(A, np.asarray(lo, float), np.asarray(hi, float)),
        bounds=Bounds(0, 1),
        integrality=np.ones(nx + ns + nc, dtype=int),
        options={"time_limit": float(time_limit)},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Layer-1 week MILP failed: {res.message}")

    x = res.x[:n]
    s = res.x[n : 2 * n]
    cap = res.x[2 * n :]
    squad_idx = np.where(x > 0.5)[0]
    xi_idx = np.where(s > 0.5)[0]
    cap_idx = np.where(cap > 0.5)[0]
    if len(squad_idx) != SQUAD_SIZE or len(xi_idx) != 11 or len(cap_idx) != 1:
        raise RuntimeError(
            f"Layer-1 bad support: |squad|={len(squad_idx)} |xi|={len(xi_idx)} |cap|={len(cap_idx)}"
        )

    squad = df.iloc[squad_idx].copy()
    xi = df.iloc[xi_idx].copy()
    captain_id = str(df.iloc[int(cap_idx[0])]["player_id"])
    if captain_id not in set(xi["player_id"].astype(str)):
        raise RuntimeError("captain not in XI")

    positions = squad["position"].astype(str).tolist()
    clubs = squad["club"].astype(str).tolist()
    values = [int(v) for v in squad["value"].tolist()]
    if not squad_legal(positions, clubs, values, budget=budget):
        raise RuntimeError("Layer-1 squad failed squad_legal")
    if not xi_legal(xi["position"].astype(str).tolist()):
        raise RuntimeError("Layer-1 XI failed xi_legal")

    n_def = int((xi["position"] == "DEF").sum())
    n_mid = int((xi["position"] == "MID").sum())
    n_fwd = int((xi["position"] == "FWD").sum())
    xi_mu = float(xi["mu"].sum())
    captain_mu = float(df.loc[df["player_id"] == captain_id, "mu"].iloc[0])
    objective = xi_mu + captain_mu
    cost = int(squad["value"].sum())

    squad = squad.assign(in_xi=squad["player_id"].isin(xi["player_id"]), is_captain=False)
    squad.loc[squad["player_id"] == captain_id, "is_captain"] = True
    xi = xi.copy()
    xi["is_captain"] = xi["player_id"] == captain_id

    return WeekPlan(
        squad=squad.reset_index(drop=True),
        xi=xi.reset_index(drop=True),
        captain_id=captain_id,
        formation=(n_def, n_mid, n_fwd),
        objective=objective,
        xi_mu=xi_mu,
        captain_mu=captain_mu,
        cost=cost,
        status=str(res.message),
    )
