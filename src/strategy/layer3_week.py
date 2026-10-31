"""Layer 3 — single-deadline transfers + hits from a held 15.

Maximises

    Σ_i μ_post · start_i  +  Σ_i μ_post · captain_i  −  HIT_COST · hits

subject to FPL 2026/27 squad, budget (bank + sell prices), and FT bank.
Hits are unconstrained aside from a solver bound of ``SQUAD_SIZE`` (full
rebuild); each hit still costs ``HIT_COST``. Chips and multi-week horizon
are out of scope.

``value_captain=False`` keeps a legal captain but drops the captain double
from the objective (transfer-rule / component isolation).

μ comes from the caller (normally Layer-2 ``mu_post`` ≡ ``score_xp_v2``
after 2026-10-16). No stacked ~0.5 haircut.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.rules.fpl_2026 import (
    HIT_COST,
    MAX_DEF,
    MAX_FWD,
    MAX_MID,
    MAX_PER_CLUB,
    MIN_DEF,
    MIN_FWD,
    MIN_MID,
    SQUAD_QUOTA,
    SQUAD_SIZE,
    advance_ft,
    hit_cost,
    sell_price,
    xi_legal,
)
from src.strategy.layer1_week import _prepare_pool

# Soft solver bound only — FPL has no hit cap. Objective still charges HIT_COST
# per hit. Cap at SQUAD_SIZE so a full rebuild is feasible; do not tighten.
DEFAULT_MAX_HITS = SQUAD_SIZE


@dataclass(frozen=True)
class HeldState:
    """Squad owned before this deadline."""

    player_ids: tuple[str, ...]
    purchase: Mapping[str, int]
    bank: int
    ft: int

    def __post_init__(self) -> None:
        ids = tuple(str(p) for p in self.player_ids)
        if len(ids) != SQUAD_SIZE:
            raise ValueError(f"held squad must have {SQUAD_SIZE} players, got {len(ids)}")
        if len(set(ids)) != SQUAD_SIZE:
            raise ValueError("held squad has duplicate player_id")
        missing = [p for p in ids if p not in self.purchase]
        if missing:
            raise ValueError(f"purchase missing for {missing[:3]}")
        if int(self.bank) < 0:
            raise ValueError("bank must be non-negative")
        if int(self.ft) < 0:
            raise ValueError("ft must be non-negative")
        object.__setattr__(self, "player_ids", ids)
        object.__setattr__(self, "bank", int(self.bank))
        object.__setattr__(self, "ft", int(self.ft))


@dataclass(frozen=True)
class TransferWeekPlan:
    """Layer-3 solution for one transfer deadline."""

    squad: pd.DataFrame
    xi: pd.DataFrame
    captain_id: str
    formation: tuple[int, int, int]
    objective: float
    xi_mu: float
    captain_mu: float
    hits: int
    n_transfers: int
    hit_points: int
    bank_after: int
    ft_after: int
    sold: tuple[str, ...]
    bought: tuple[str, ...]
    cost: int
    status: str
    mu_col: str

    def summary(self) -> dict[str, Any]:
        return {
            "n_squad": int(len(self.squad)),
            "n_xi": int(len(self.xi)),
            "captain_id": self.captain_id,
            "formation": f"1-{self.formation[0]}-{self.formation[1]}-{self.formation[2]}",
            "objective": float(self.objective),
            "xi_mu": float(self.xi_mu),
            "captain_mu": float(self.captain_mu),
            "n_transfers": int(self.n_transfers),
            "hits": int(self.hits),
            "hit_points": int(self.hit_points),
            "bank_after": int(self.bank_after),
            "ft_after": int(self.ft_after),
            "sold": list(self.sold),
            "bought": list(self.bought),
            "cost": int(self.cost),
            "mu_col": self.mu_col,
            "status": self.status,
        }


def held_from_frame(
    squad: pd.DataFrame,
    *,
    bank: int,
    ft: int,
    purchase_col: str = "purchase",
) -> HeldState:
    """Build HeldState from a 15-row frame."""
    if len(squad) != SQUAD_SIZE:
        raise ValueError(f"need {SQUAD_SIZE} rows, got {len(squad)}")
    ids = [str(p) for p in squad["player_id"].tolist()]
    if purchase_col in squad.columns:
        purchase = {
            str(r["player_id"]): int(r[purchase_col]) for _, r in squad.iterrows()
        }
    elif "value" in squad.columns:
        purchase = {str(r["player_id"]): int(r["value"]) for _, r in squad.iterrows()}
    else:
        raise ValueError("squad needs purchase or value column")
    return HeldState(tuple(ids), purchase, bank=bank, ft=ft)


def _ensure_held_in_pool(df: pd.DataFrame, held: HeldState) -> pd.DataFrame:
    have = set(df["player_id"].astype(str))
    missing = [p for p in held.player_ids if p not in have]
    if missing:
        raise ValueError(
            f"held players missing from pool (include them even under top-N): {missing[:5]}"
        )
    return df


def solve_week_transfers(
    pool: pd.DataFrame,
    held: HeldState,
    *,
    mu_col: str = "mu_post",
    max_per_club: int = MAX_PER_CLUB,
    max_hits: int = DEFAULT_MAX_HITS,
    time_limit: float = 30.0,
    value_captain: bool = True,
) -> TransferWeekPlan:
    """Joint MILP: transfers from held 15 + XI (+ optional captain μ) net of hits."""
    df = _prepare_pool(pool, mu_col=mu_col)
    df = _ensure_held_in_pool(df, held)

    n = len(df)
    mu = df["mu"].to_numpy(float)
    value = df["value"].to_numpy(int)
    pos = df["position"].to_numpy(str)
    club = df["club"].to_numpy(str)
    pids = df["player_id"].astype(str).to_numpy()

    owned_mask = np.array([pid in held.purchase for pid in pids], dtype=bool)
    # Charge sell_price for owned (kept or not) in affordability dual form;
    # charge now-cost for non-owned buys.
    sell = np.zeros(n, dtype=float)
    for i, pid in enumerate(pids):
        if owned_mask[i]:
            sell[i] = float(sell_price(int(held.purchase[pid]), int(value[i])))

    wealth = float(held.bank + sell[owned_mask].sum())
    max_transfers = int(held.ft) + int(max_hits)
    # Decision: x | s | c | extra   (extra = hit count beyond FT)
    nx = ns = nc = n
    n_extra = 1
    dim = nx + ns + nc + n_extra

    def blk(kind: str, i: int = 0) -> int:
        if kind == "x":
            return i
        if kind == "s":
            return nx + i
        if kind == "c":
            return nx + ns + i
        return nx + ns + nc  # extra

    # minimise −(Σ μ s [+ Σ μ c]) + HIT_COST * extra
    c_obj = np.zeros(dim)
    use_cap = bool(value_captain)
    for i in range(n):
        c_obj[blk("s", i)] = -mu[i]
        if use_cap:
            c_obj[blk("c", i)] = -mu[i]
    c_obj[blk("extra")] = float(HIT_COST)

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add_row(coeff: np.ndarray, low: float, high: float) -> None:
        rows.append(coeff)
        lo.append(low)
        hi.append(high)

    # Σ x = 15, Σ s = 11, Σ c = 1
    r = np.zeros(dim)
    for i in range(n):
        r[blk("x", i)] = 1.0
    add_row(r, float(SQUAD_SIZE), float(SQUAD_SIZE))

    r = np.zeros(dim)
    for i in range(n):
        r[blk("s", i)] = 1.0
    add_row(r, 11.0, 11.0)

    r = np.zeros(dim)
    for i in range(n):
        r[blk("c", i)] = 1.0
    add_row(r, 1.0, 1.0)

    for p, need in SQUAD_QUOTA.items():
        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == p:
                r[blk("x", i)] = 1.0
        add_row(r, float(need), float(need))

    r = np.zeros(dim)
    for i in range(n):
        if pos[i] == "GKP":
            r[blk("s", i)] = 1.0
    add_row(r, 1.0, 1.0)
    r = np.zeros(dim)
    for i in range(n):
        if pos[i] == "DEF":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_DEF), float(MAX_DEF))
    r = np.zeros(dim)
    for i in range(n):
        if pos[i] == "MID":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_MID), float(MAX_MID))
    r = np.zeros(dim)
    for i in range(n):
        if pos[i] == "FWD":
            r[blk("s", i)] = 1.0
    add_row(r, float(MIN_FWD), float(MAX_FWD))

    for cl in sorted(set(club.tolist())):
        r = np.zeros(dim)
        for i in range(n):
            if club[i] == cl:
                r[blk("x", i)] = 1.0
        add_row(r, 0.0, float(max_per_club))

    # Affordability: Σ_{buy} value·x + Σ_{owned} sell·x ≤ bank + Σ_{owned} sell
    r = np.zeros(dim)
    for i in range(n):
        if owned_mask[i]:
            r[blk("x", i)] = sell[i]
        else:
            r[blk("x", i)] = float(value[i])
    add_row(r, -np.inf, wealth)

    # Transfers: n_buys = Σ_{not owned} x ≤ max_transfers
    r = np.zeros(dim)
    for i in range(n):
        if not owned_mask[i]:
            r[blk("x", i)] = 1.0
    add_row(r, 0.0, float(max_transfers))

    # extra ≥ n_buys − ft  →  extra − Σ_{buy} x ≥ −ft
    r = np.zeros(dim)
    r[blk("extra")] = 1.0
    for i in range(n):
        if not owned_mask[i]:
            r[blk("x", i)] = -1.0
    add_row(r, float(-held.ft), np.inf)

    # Linking s ≤ x, c ≤ s
    for i in range(n):
        r = np.zeros(dim)
        r[blk("s", i)] = 1.0
        r[blk("x", i)] = -1.0
        add_row(r, -np.inf, 0.0)
        r = np.zeros(dim)
        r[blk("c", i)] = 1.0
        r[blk("s", i)] = -1.0
        add_row(r, -np.inf, 0.0)

    A = np.vstack(rows)
    # Bounds: binaries on x,s,c; extra in 0..max_hits
    lb = np.zeros(dim)
    ub = np.ones(dim)
    ub[blk("extra")] = float(max_hits)
    integrality = np.ones(dim, dtype=int)

    res = milp(
        c=c_obj,
        constraints=LinearConstraint(A, np.asarray(lo, float), np.asarray(hi, float)),
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"time_limit": float(time_limit)},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Layer-3 transfer MILP failed: {res.message}")

    x = res.x[:n]
    s = res.x[n : 2 * n]
    cap = res.x[2 * n : 3 * n]
    extra = int(round(float(res.x[3 * n])))
    squad_idx = np.where(x > 0.5)[0]
    xi_idx = np.where(s > 0.5)[0]
    cap_idx = np.where(cap > 0.5)[0]
    if len(squad_idx) != SQUAD_SIZE or len(xi_idx) != 11 or len(cap_idx) != 1:
        raise RuntimeError(
            f"Layer-3 bad support: |squad|={len(squad_idx)} |xi|={len(xi_idx)} |cap|={len(cap_idx)}"
        )

    squad = df.iloc[squad_idx].copy()
    xi = df.iloc[xi_idx].copy()
    captain_id = str(df.iloc[int(cap_idx[0])]["player_id"])
    new_ids = set(squad["player_id"].astype(str))
    old_ids = set(held.player_ids)
    bought = tuple(sorted(new_ids - old_ids))
    sold = tuple(sorted(old_ids - new_ids))
    n_transfers = len(bought)
    if n_transfers != len(sold):
        raise RuntimeError(f"buy/sell mismatch: bought={bought} sold={sold}")
    hp = hit_cost(held.ft, n_transfers)
    if hp != HIT_COST * extra and not (extra == max(0, n_transfers - held.ft)):
        # Prefer identity from rules on realised n_transfers
        pass
    hits = max(0, n_transfers - held.ft)
    hit_points = HIT_COST * hits

    # Bank after
    revenue = sum(
        sell_price(int(held.purchase[pid]), int(df.loc[df["player_id"] == pid, "value"].iloc[0]))
        for pid in sold
    )
    spend = sum(int(df.loc[df["player_id"] == pid, "value"].iloc[0]) for pid in bought)
    bank_after = int(held.bank + revenue - spend)
    if bank_after < 0:
        raise RuntimeError(f"Layer-3 negative bank_after={bank_after}")

    positions = squad["position"].astype(str).tolist()
    clubs = squad["club"].astype(str).tolist()
    # Mid-season now-cost can exceed £100m; enforce quota + club cap only.
    # Affordability was enforced via bank + sell prices in the MILP.
    if dict(Counter(positions)) != dict(SQUAD_QUOTA):
        raise RuntimeError(f"Layer-3 squad failed position quota: {Counter(positions)}")
    for cl, cnt in Counter(clubs).items():
        if int(cnt) > max_per_club:
            raise RuntimeError(f"Layer-3 club cap broken for {cl}: {cnt}")
    if not xi_legal(xi["position"].astype(str).tolist()):
        raise RuntimeError("Layer-3 XI failed xi_legal")
    if hit_cost(held.ft, n_transfers) != hit_points:
        raise RuntimeError(
            f"hit identity failed: rules={hit_cost(held.ft, n_transfers)} plan={hit_points}"
        )

    n_def = int((xi["position"] == "DEF").sum())
    n_mid = int((xi["position"] == "MID").sum())
    n_fwd = int((xi["position"] == "FWD").sum())
    xi_mu = float(xi["mu"].sum())
    captain_mu = float(df.loc[df["player_id"] == captain_id, "mu"].iloc[0])
    objective = xi_mu - float(hit_points)
    if use_cap:
        objective += captain_mu

    squad = squad.assign(in_xi=squad["player_id"].isin(xi["player_id"]), is_captain=False)
    squad.loc[squad["player_id"] == captain_id, "is_captain"] = True
    xi = xi.copy()
    xi["is_captain"] = xi["player_id"] == captain_id

    return TransferWeekPlan(
        squad=squad.reset_index(drop=True),
        xi=xi.reset_index(drop=True),
        captain_id=captain_id,
        formation=(n_def, n_mid, n_fwd),
        objective=objective,
        xi_mu=xi_mu,
        captain_mu=captain_mu if use_cap else 0.0,
        hits=hits,
        n_transfers=n_transfers,
        hit_points=hit_points,
        bank_after=bank_after,
        ft_after=advance_ft(held.ft, n_transfers),
        sold=sold,
        bought=bought,
        cost=int(squad["value"].sum()),
        status=str(res.message),
        mu_col=mu_col,
    )
