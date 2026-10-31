"""Layer 4 — receding-horizon transfers on μ_post.

Transfers + hits only at the **decision week** (t=0), same constraints as
Layer 3. For t = 1..H−1 the squad is fixed and only XI + captain are chosen
on that week's μ board. Objective:

    Σ_{t=0}^{H−1} γ^t (Σ μ_{t,i} start_{t,i} + Σ μ_{t,i} captain_{t,i})
    − HIT_COST · hits_0
    + γ^H · (τ_μ · Σ_i μ_{H−1,i} x_i + τ_v · Σ_i value_i x_i / V_scale)

Commit week-0 transfers only; re-solve next deadline.

``H=1`` with terminal weights 0 recovers ``solve_week_transfers``.

  python -m src.strategy.layer4_week   # smoke on toy (via tests)
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.models.season_climb_ft import GAMMA
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
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    TransferWeekPlan,
    _ensure_held_in_pool,
    solve_week_transfers,
)

DEFAULT_HORIZON = 4
DEFAULT_TERMINAL_MU_WEIGHT = 0.25
DEFAULT_TERMINAL_VALUE_WEIGHT = 0.0
VALUE_SCALE_TENTHS = 1000.0


@dataclass(frozen=True)
class HorizonPlan:
    """Layer-4 solution; commit ``week0`` only."""

    week0: TransferWeekPlan
    horizon: int
    gamma: float
    objective: float
    week_xi_mu: tuple[float, ...]
    week_captain_mu: tuple[float, ...]
    terminal: float
    status: str
    mu_col: str

    def summary(self) -> dict[str, Any]:
        w0 = self.week0.summary()
        return {
            **w0,
            "horizon": int(self.horizon),
            "gamma": float(self.gamma),
            "horizon_objective": float(self.objective),
            "week_xi_mu": [float(x) for x in self.week_xi_mu],
            "week_captain_mu": [float(x) for x in self.week_captain_mu],
            "terminal": float(self.terminal),
            "status": self.status,
        }


def align_horizon_pools(
    pools: Sequence[pd.DataFrame],
    held: HeldState,
    *,
    mu_col: str,
) -> list[pd.DataFrame]:
    """Align future μ onto the **decision-week** player universe.

    Buys must be available at t=0, so future-only players are never added.
    Missing future μ → 0 (still selectable in later XI only if owned).
    """
    if not pools:
        raise ValueError("need at least one pool")
    prepared = [_prepare_pool(p, mu_col=mu_col) for p in pools]
    prepared[0] = _ensure_held_in_pool(prepared[0], held)

    base = prepared[0].drop_duplicates("player_id", keep="first").reset_index(drop=True)
    pids = base["player_id"].astype(str).tolist()

    out: list[pd.DataFrame] = []
    for t, df in enumerate(prepared):
        by = df.drop_duplicates("player_id", keep="first").set_index(
            df.drop_duplicates("player_id", keep="first")["player_id"].astype(str)
        )
        rows = []
        for pid in pids:
            b = base.loc[base["player_id"].astype(str) == pid].iloc[0]
            if pid in by.index:
                r = by.loc[pid]
                if isinstance(r, pd.DataFrame):
                    r = r.iloc[0]
                mu_t = float(r["mu"])
            else:
                mu_t = 0.0
            rows.append(
                {
                    "player_id": pid,
                    "position": str(b["position"]),
                    "club": str(b["club"]),
                    "value": int(b["value"]),  # purchase prices = decision-week now-cost
                    "mu": mu_t,
                    "name": str(b["name"]) if "name" in b.index else pid,
                }
            )
        out.append(pd.DataFrame(rows))
    return out


def solve_horizon_transfers(
    pools: Sequence[pd.DataFrame],
    held: HeldState,
    *,
    mu_col: str = "mu_post",
    horizon: int | None = None,
    gamma: float = GAMMA,
    terminal_mu_weight: float = DEFAULT_TERMINAL_MU_WEIGHT,
    terminal_value_weight: float = DEFAULT_TERMINAL_VALUE_WEIGHT,
    terminal_by_player: Mapping[str, float] | None = None,
    value_scale: float = VALUE_SCALE_TENTHS,
    max_per_club: int = MAX_PER_CLUB,
    max_hits: int = DEFAULT_MAX_HITS,
    hit_penalty: float | None = None,
    time_limit: float = 60.0,
    value_captain: bool = True,
) -> HorizonPlan:
    """Multi-week horizon MILP; returns commit-week TransferWeekPlan in ``week0``.

    ``terminal_by_player`` (optional): extra per-player hold credit on squad ``x``,
    added as ``γ^H · credit[pid]`` — used for beyond-t+2 form without fill boards.

    ``hit_penalty`` (optional): objective cost per hit; defaults to ``HIT_COST``.
    Realised ``hit_points`` always use official ``HIT_COST`` (gate ≠ rescore).

    ``value_captain=False`` keeps a legal captain each week but drops the captain
    double from the objective (component / transfer-rule isolation).
    """
    H = int(horizon if horizon is not None else len(pools))
    if H < 1:
        raise ValueError("horizon must be >= 1")
    if len(pools) < H:
        raise ValueError(f"need {H} weekly pools, got {len(pools)}")
    pools = list(pools[:H])
    use_cap = bool(value_captain)

    # H=1 + zero terminal → exact L3
    if (
        H == 1
        and float(terminal_mu_weight) == 0.0
        and float(terminal_value_weight) == 0.0
        and not terminal_by_player
    ):
        w0 = solve_week_transfers(
            pools[0],
            held,
            mu_col=mu_col,
            max_per_club=max_per_club,
            max_hits=max_hits,
            time_limit=time_limit,
            value_captain=use_cap,
        )
        return HorizonPlan(
            week0=w0,
            horizon=1,
            gamma=float(gamma),
            objective=float(w0.objective),
            week_xi_mu=(float(w0.xi_mu),),
            week_captain_mu=(float(w0.captain_mu),),
            terminal=0.0,
            status=w0.status,
            mu_col=mu_col,
        )

    aligned = align_horizon_pools(pools, held, mu_col=mu_col)
    df0 = aligned[0]
    n = len(df0)
    mu = [a["mu"].to_numpy(float) for a in aligned]
    value = df0["value"].to_numpy(int)
    pos = df0["position"].to_numpy(str)
    club = df0["club"].to_numpy(str)
    pids = df0["player_id"].astype(str).to_numpy()

    owned_mask = np.array([pid in held.purchase for pid in pids], dtype=bool)
    sell = np.zeros(n, dtype=float)
    for i, pid in enumerate(pids):
        if owned_mask[i]:
            sell[i] = float(sell_price(int(held.purchase[pid]), int(value[i])))
    wealth = float(held.bank + sell[owned_mask].sum())
    max_transfers = int(held.ft) + int(max_hits)

    # Layout: x[n] | s[H*n] | c[H*n] | extra[1]
    nx = n
    ns = H * n
    nc = H * n
    n_extra = 1
    dim = nx + ns + nc + n_extra

    def ix_x(i: int) -> int:
        return i

    def ix_s(t: int, i: int) -> int:
        return nx + t * n + i

    def ix_c(t: int, i: int) -> int:
        return nx + ns + t * n + i

    def ix_extra() -> int:
        return nx + ns + nc

    g = float(gamma)
    c_obj = np.zeros(dim)
    for t in range(H):
        w = g**t
        for i in range(n):
            c_obj[ix_s(t, i)] = -w * mu[t][i]
            if use_cap:
                c_obj[ix_c(t, i)] = -w * mu[t][i]
    # Terminal on squad using last week's μ + mild value (+ optional by-player)
    wT = g**H
    tau_mu = float(terminal_mu_weight)
    tau_v = float(terminal_value_weight)
    tb = terminal_by_player or {}
    for i in range(n):
        term = tau_mu * mu[H - 1][i]
        if tau_v and value_scale:
            term += tau_v * (float(value[i]) / float(value_scale))
        term += float(tb.get(str(pids[i]), 0.0))
        c_obj[ix_x(i)] -= wT * term
    pen = float(HIT_COST if hit_penalty is None else hit_penalty)
    c_obj[ix_extra()] = pen

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add_row(coeff: np.ndarray, low: float, high: float) -> None:
        rows.append(coeff)
        lo.append(low)
        hi.append(high)

    # Squad size + quotas (once)
    r = np.zeros(dim)
    for i in range(n):
        r[ix_x(i)] = 1.0
    add_row(r, float(SQUAD_SIZE), float(SQUAD_SIZE))

    for p, need in SQUAD_QUOTA.items():
        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == p:
                r[ix_x(i)] = 1.0
        add_row(r, float(need), float(need))

    for cl in sorted(set(club.tolist())):
        r = np.zeros(dim)
        for i in range(n):
            if club[i] == cl:
                r[ix_x(i)] = 1.0
        add_row(r, 0.0, float(max_per_club))

    # Affordability + transfers (t=0 only) — same as L3
    r = np.zeros(dim)
    for i in range(n):
        r[ix_x(i)] = sell[i] if owned_mask[i] else float(value[i])
    add_row(r, -np.inf, wealth)

    r = np.zeros(dim)
    for i in range(n):
        if not owned_mask[i]:
            r[ix_x(i)] = 1.0
    add_row(r, 0.0, float(max_transfers))

    r = np.zeros(dim)
    r[ix_extra()] = 1.0
    for i in range(n):
        if not owned_mask[i]:
            r[ix_x(i)] = -1.0
    add_row(r, float(-held.ft), np.inf)

    # Per-week XI + captain + formation + linking
    for t in range(H):
        r = np.zeros(dim)
        for i in range(n):
            r[ix_s(t, i)] = 1.0
        add_row(r, 11.0, 11.0)
        r = np.zeros(dim)
        for i in range(n):
            r[ix_c(t, i)] = 1.0
        add_row(r, 1.0, 1.0)

        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == "GKP":
                r[ix_s(t, i)] = 1.0
        add_row(r, 1.0, 1.0)
        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == "DEF":
                r[ix_s(t, i)] = 1.0
        add_row(r, float(MIN_DEF), float(MAX_DEF))
        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == "MID":
                r[ix_s(t, i)] = 1.0
        add_row(r, float(MIN_MID), float(MAX_MID))
        r = np.zeros(dim)
        for i in range(n):
            if pos[i] == "FWD":
                r[ix_s(t, i)] = 1.0
        add_row(r, float(MIN_FWD), float(MAX_FWD))

        for i in range(n):
            r = np.zeros(dim)
            r[ix_s(t, i)] = 1.0
            r[ix_x(i)] = -1.0
            add_row(r, -np.inf, 0.0)
            r = np.zeros(dim)
            r[ix_c(t, i)] = 1.0
            r[ix_s(t, i)] = -1.0
            add_row(r, -np.inf, 0.0)

    A = np.vstack(rows)
    lb = np.zeros(dim)
    ub = np.ones(dim)
    ub[ix_extra()] = float(max_hits)
    integrality = np.ones(dim, dtype=int)

    res = milp(
        c=c_obj,
        constraints=LinearConstraint(A, np.asarray(lo, float), np.asarray(hi, float)),
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={"time_limit": float(time_limit)},
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"Layer-4 horizon MILP failed: {res.message}")

    x = res.x[:n]
    squad_idx = np.where(x > 0.5)[0]
    if len(squad_idx) != SQUAD_SIZE:
        raise RuntimeError(f"Layer-4 bad squad support: {len(squad_idx)}")

    # Week-0 XI / captain
    s0 = res.x[nx : nx + n]
    c0 = res.x[nx + ns : nx + ns + n]
    xi0_idx = np.where(s0 > 0.5)[0]
    cap0_idx = np.where(c0 > 0.5)[0]
    if len(xi0_idx) != 11 or len(cap0_idx) != 1:
        raise RuntimeError("Layer-4 bad week-0 XI/captain")

    squad = df0.iloc[squad_idx].copy()
    xi = df0.iloc[xi0_idx].copy()
    captain_id = str(df0.iloc[int(cap0_idx[0])]["player_id"])

    new_ids = set(squad["player_id"].astype(str))
    old_ids = set(held.player_ids)
    bought = tuple(sorted(new_ids - old_ids))
    sold = tuple(sorted(old_ids - new_ids))
    n_transfers = len(bought)
    if n_transfers != len(sold):
        raise RuntimeError(f"buy/sell mismatch: bought={bought} sold={sold}")
    hits = max(0, n_transfers - held.ft)
    hit_points = HIT_COST * hits
    if hit_cost(held.ft, n_transfers) != hit_points:
        raise RuntimeError("Layer-4 hit identity failed")

    revenue = sum(
        sell_price(
            int(held.purchase[pid]),
            int(df0.loc[df0["player_id"] == pid, "value"].iloc[0]),
        )
        for pid in sold
    )
    spend = sum(int(df0.loc[df0["player_id"] == pid, "value"].iloc[0]) for pid in bought)
    bank_after = int(held.bank + revenue - spend)
    if bank_after < 0:
        raise RuntimeError(f"Layer-4 negative bank_after={bank_after}")

    if dict(Counter(squad["position"].astype(str))) != dict(SQUAD_QUOTA):
        raise RuntimeError("Layer-4 squad quota failed")
    for cl, cnt in Counter(squad["club"].astype(str)).items():
        if int(cnt) > max_per_club:
            raise RuntimeError(f"Layer-4 club cap broken for {cl}")
    if not xi_legal(xi["position"].astype(str).tolist()):
        raise RuntimeError("Layer-4 week-0 XI illegal")

    week_xi_mu: list[float] = []
    week_cap_mu: list[float] = []
    for t in range(H):
        st = res.x[nx + t * n : nx + (t + 1) * n]
        ct = res.x[nx + ns + t * n : nx + ns + (t + 1) * n]
        xi_t = np.where(st > 0.5)[0]
        cap_t = np.where(ct > 0.5)[0]
        if len(xi_t) != 11 or len(cap_t) != 1:
            raise RuntimeError(f"Layer-4 bad XI/cap at t={t}")
        week_xi_mu.append(float(mu[t][xi_t].sum()))
        week_cap_mu.append(float(mu[t][int(cap_t[0])]))

    terminal = 0.0
    for i in squad_idx:
        terminal += tau_mu * float(mu[H - 1][i])
        if tau_v and value_scale:
            terminal += tau_v * (float(value[i]) / float(value_scale))
    terminal *= wT

    horizon_obj = (
        sum(
            (g**t) * (week_xi_mu[t] + (week_cap_mu[t] if use_cap else 0.0))
            for t in range(H)
        )
        - float(hit_points)
        + terminal
    )

    n_def = int((xi["position"] == "DEF").sum())
    n_mid = int((xi["position"] == "MID").sum())
    n_fwd = int((xi["position"] == "FWD").sum())
    xi_mu0 = week_xi_mu[0]
    captain_mu0 = week_cap_mu[0] if use_cap else 0.0
    # Week-0 objective piece (L3-style) for TransferWeekPlan.objective
    week0_obj = xi_mu0 + captain_mu0 - float(hit_points)

    squad = squad.assign(in_xi=squad["player_id"].isin(xi["player_id"]), is_captain=False)
    squad.loc[squad["player_id"] == captain_id, "is_captain"] = True
    xi = xi.copy()
    xi["is_captain"] = xi["player_id"] == captain_id
    # Attach week-0 mu column name expected by callers
    squad["mu"] = squad["player_id"].map(
        {str(p): float(m) for p, m in zip(pids, mu[0], strict=True)}
    )
    xi["mu"] = xi["player_id"].map(
        {str(p): float(m) for p, m in zip(pids, mu[0], strict=True)}
    )

    week0 = TransferWeekPlan(
        squad=squad.reset_index(drop=True),
        xi=xi.reset_index(drop=True),
        captain_id=captain_id,
        formation=(n_def, n_mid, n_fwd),
        objective=float(week0_obj),
        xi_mu=float(xi_mu0),
        captain_mu=float(captain_mu0),
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
    return HorizonPlan(
        week0=week0,
        horizon=H,
        gamma=g,
        objective=float(horizon_obj),
        week_xi_mu=tuple(week_xi_mu),
        week_captain_mu=tuple(week_cap_mu),
        terminal=float(terminal),
        status=str(res.message),
        mu_col=mu_col,
    )
