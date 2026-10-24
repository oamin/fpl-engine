"""Joint multi-GW transfer plan MILP for one cascade path strand.

Given per-GW player scores on a fully rolled strand, choose a legal 15 + XI
(+ optional captain) for **every** horizon GW and the transfers that link
them, subject to FT bank dynamics and −4/hit.

Modes:
- **free open** (default): opening GW is a wildcard-like rebuild; FT bank
  after open is 1; later weeks transfer from that 15.
- **from owned**: first GW transfers from a held 15 with a given FT bank
  (rolling deadline replan).

``value_captain=False`` isolates the **transfer rule**: objective is Σ XI
scores − hits; a legal captain is still picked but does not enter the
objective (no captaincy double).

**FT bank pot** is tracked as constrained state (``ft_before`` / ``ft_after``
per week) and returned as ``bank_pot`` — separate from the objective.
Default ``ft_pe_nu=0`` (no terminal bank PE in \(V\)).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp

from src.models.multistage_mc_milp import (
    DEFAULT_FT_PE_NU,
    DEFAULT_GAMMA,
    MAX_HITS_SNAPSHOT,
    ft_bank_potential,
)
from src.rules.fpl_2026 import (
    BUDGET_TENTHS as BUDGET,
    HIT_COST,
    MAX_DEF,
    MAX_FT,
    MAX_FWD,
    MAX_MID,
    MAX_PER_CLUB,
    MIN_DEF,
    MIN_FWD,
    MIN_MID,
    SQUAD_QUOTA,
    SQUAD_SIZE,
)

# Tractability: keep top-N by mean horizon score (per-position floors below).
DEFAULT_TOP_N = 48
POS_FLOOR = {"GKP": 3, "DEF": 8, "MID": 8, "FWD": 5}


def select_plan_candidates(
    pool: pd.DataFrame,
    scores_by_gw: Mapping[int, np.ndarray],
    *,
    gws: Sequence[int],
    buy_ids: set[str] | None,
    top_n: int = DEFAULT_TOP_N,
    force_ids: set[str] | None = None,
) -> pd.DataFrame:
    """Universe for the joint plan: buy filter ∩ top mean-score players."""
    df = pool.copy()
    if "player_id" not in df.columns:
        raise ValueError("pool needs player_id")
    df["player_id"] = df["player_id"].astype(str)
    if buy_ids is not None:
        keep = set(buy_ids)
        if force_ids:
            keep |= {str(x) for x in force_ids}
        df = df.loc[df["player_id"].isin(keep)].copy()
    if df.empty:
        raise RuntimeError("no candidates after buy filter")
    mean = np.zeros(len(df), dtype=float)
    pid_to_local = {pid: i for i, pid in enumerate(df["player_id"])}
    pool_pids = pool["player_id"].astype(str).to_numpy()
    pool_index = {pid: i for i, pid in enumerate(pool_pids)}
    for gw in gws:
        arr = np.asarray(scores_by_gw[int(gw)], dtype=float)
        for pid, li in pid_to_local.items():
            pi = pool_index.get(pid)
            if pi is None or pi >= len(arr):
                continue
            mean[li] += float(arr[pi])
    mean /= max(1, len(gws))
    df = df.reset_index(drop=True)
    df["_mean_s"] = mean
    kept_idx: list[int] = []
    force = {str(x) for x in force_ids} if force_ids else set()
    if force:
        kept_idx.extend(
            df.loc[df["player_id"].isin(force)].index.tolist()
        )
    for pos, floor in POS_FLOOR.items():
        sub = df.loc[df["position"].astype(str) == pos].sort_values(
            "_mean_s", ascending=False
        )
        kept_idx.extend(sub.head(int(floor)).index.tolist())
    rest = df.drop(index=list(set(kept_idx))).sort_values("_mean_s", ascending=False)
    for idx in rest.index.tolist():
        if len(set(kept_idx)) >= int(top_n):
            break
        kept_idx.append(int(idx))
    # Always keep forced owned ids even above top_n.
    if force:
        for idx in df.loc[df["player_id"].isin(force)].index.tolist():
            kept_idx.append(int(idx))
    out = df.loc[sorted(set(kept_idx))].drop(columns=["_mean_s"]).reset_index(drop=True)
    if len(out) < SQUAD_SIZE or out["position"].nunique() < 4:
        raise RuntimeError(
            f"plan candidate set too small: n={len(out)} pos={out['position'].nunique()}"
        )
    return out


def score_matrix_for_candidates(
    candidates: pd.DataFrame,
    pool: pd.DataFrame,
    scores_by_gw: Mapping[int, np.ndarray],
    gws: Sequence[int],
) -> np.ndarray:
    """Return S[t, i] for candidates i, weeks t."""
    pool_pids = pool["player_id"].astype(str).to_numpy()
    pool_index = {pid: i for i, pid in enumerate(pool_pids)}
    H = len(gws)
    n = len(candidates)
    S = np.zeros((H, n), dtype=float)
    for t, gw in enumerate(gws):
        arr = np.asarray(scores_by_gw[int(gw)], dtype=float)
        for i, pid in enumerate(candidates["player_id"].astype(str)):
            pi = pool_index.get(pid)
            if pi is not None and pi < len(arr):
                S[t, i] = float(arr[pi])
    return S


def pick_path_transfer_plan(
    candidates: pd.DataFrame,
    scores: np.ndarray,
    gws: Sequence[int],
    *,
    gamma: float = DEFAULT_GAMMA,
    max_hits: int = MAX_HITS_SNAPSHOT,
    ft_pe_nu: float = DEFAULT_FT_PE_NU,
    time_limit: float = 90.0,
    start_owned_ids: Sequence[str] | None = None,
    start_ft: int = 1,
    value_captain: bool = True,
) -> dict[str, Any]:
    """Joint path MILP over ``gws``.

    ``scores`` shape ``(H, n)`` aligned with ``gws`` and ``candidates``.

    When ``start_owned_ids`` is None, week 0 is a free rebuild and ``f_after[0]=1``.
    When set, week 0 transfers from that 15 with bank ``start_ft``.

    When ``ft_pe_nu > 0``, adds terminal PE ``U(f_H)=ν·f_H`` into the
    objective (diagnostic only; default ``ft_pe_nu=0``). The FT **bank pot**
    is always tracked via constraints and returned in ``bank_pot``.

    ``value_captain``: if False, captain double is omitted from the objective
    (transfer-rule isolation); a legal captain is still chosen in the XI.
    """
    H = int(len(gws))
    n = int(len(candidates))
    if scores.shape != (H, n):
        raise ValueError(f"scores shape {scores.shape} != ({H}, {n})")
    if H < 1:
        raise ValueError("need at least one GW")

    values = pd.to_numeric(candidates["value"], errors="coerce").fillna(50.0).to_numpy(float)
    positions = candidates["position"].astype(str).to_numpy()
    clubs = (
        candidates["team_norm"].astype(str).to_numpy()
        if "team_norm" in candidates.columns
        else candidates["team"].astype(str).to_numpy()
    )
    ids = candidates["player_id"].astype(str).to_numpy()
    names = (
        candidates["name"].astype(str).to_numpy()
        if "name" in candidates.columns
        else ids.copy()
    )

    from_owned = start_owned_ids is not None
    owned = np.zeros(n, dtype=float)
    if from_owned:
        want = {str(x) for x in start_owned_ids}
        if len(want) != SQUAD_SIZE:
            raise ValueError(f"start_owned_ids must have {SQUAD_SIZE} players")
        for i, pid in enumerate(ids):
            if pid in want:
                owned[i] = 1.0
        if int(round(owned.sum())) != SQUAD_SIZE:
            missing = want - set(ids.tolist())
            raise ValueError(f"owned players missing from candidates: {sorted(missing)[:5]}")
        ft0 = int(start_ft)
        if ft0 < 1 or ft0 > MAX_FT:
            raise ValueError(f"start_ft out of range: {ft0}")
    else:
        ft0 = 1

    nu = float(ft_pe_nu)
    use_pe = nu > 0.0

    # Transfer steps: from_owned → every week; free open → weeks 1..H-1.
    t_xfer = list(range(0, H)) if from_owned else list(range(1, H))
    n_step = len(t_xfer)
    step_of = {t: s for s, t in enumerate(t_xfer)}

    # Layout: x | y | c | buy | u | e | f | over | hit_flag
    nx = H * n
    ny = H * n
    nc = H * n
    n_buy = n_step * n
    n_u = n_step
    n_e = n_step
    n_f = H
    n_over = n_step
    n_hit = n_step
    dim = nx + ny + nc + n_buy + n_u + n_e + n_f + n_over + n_hit

    def ix_x(t: int, i: int) -> int:
        return t * n + i

    def ix_y(t: int, i: int) -> int:
        return nx + t * n + i

    def ix_c(t: int, i: int) -> int:
        return nx + ny + t * n + i

    def ix_buy(t: int, i: int) -> int:
        return nx + ny + nc + step_of[t] * n + i

    def ix_u(t: int) -> int:
        return nx + ny + nc + n_buy + step_of[t]

    def ix_e(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + step_of[t]

    def ix_f(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + t

    def ix_over(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + n_f + step_of[t]

    def ix_hit(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + n_f + n_over + step_of[t]

    g = float(gamma)
    use_cap = bool(value_captain)
    c_obj = np.zeros(dim)
    for t in range(H):
        w = g**t
        for i in range(n):
            c_obj[ix_y(t, i)] = -w * float(scores[t, i])
            if use_cap:
                c_obj[ix_c(t, i)] = -w * float(scores[t, i])
    for t in t_xfer:
        c_obj[ix_e(t)] = float(HIT_COST)
    if use_pe:
        c_obj[ix_f(H - 1)] = -nu

    rows: list[np.ndarray] = []
    lo: list[float] = []
    hi: list[float] = []

    def add(coeff: np.ndarray, low: float, high: float) -> None:
        rows.append(coeff)
        lo.append(low)
        hi.append(high)

    for t in range(H):
        r = np.zeros(dim)
        for i in range(n):
            r[ix_x(t, i)] = 1.0
        add(r, float(SQUAD_SIZE), float(SQUAD_SIZE))

        r = np.zeros(dim)
        for i in range(n):
            r[ix_x(t, i)] = float(values[i])
        add(r, -np.inf, float(BUDGET))

        for pos, need in SQUAD_QUOTA.items():
            r = np.zeros(dim)
            for i in range(n):
                if positions[i] == pos:
                    r[ix_x(t, i)] = 1.0
            add(r, float(need), float(need))

        for club in sorted(set(clubs.tolist())):
            r = np.zeros(dim)
            for i in range(n):
                if clubs[i] == club:
                    r[ix_x(t, i)] = 1.0
            add(r, -np.inf, float(MAX_PER_CLUB))

        r = np.zeros(dim)
        for i in range(n):
            r[ix_y(t, i)] = 1.0
        add(r, 11.0, 11.0)
        r = np.zeros(dim)
        for i in range(n):
            if positions[i] == "GKP":
                r[ix_y(t, i)] = 1.0
        add(r, 1.0, 1.0)
        for pos, plo, phi in (
            ("DEF", MIN_DEF, MAX_DEF),
            ("MID", MIN_MID, MAX_MID),
            ("FWD", MIN_FWD, MAX_FWD),
        ):
            r = np.zeros(dim)
            for i in range(n):
                if positions[i] == pos:
                    r[ix_y(t, i)] = 1.0
            add(r, float(plo), float(phi))

        r = np.zeros(dim)
        for i in range(n):
            r[ix_c(t, i)] = 1.0
        add(r, 1.0, 1.0)

        for i in range(n):
            r = np.zeros(dim)
            r[ix_y(t, i)] = 1.0
            r[ix_x(t, i)] = -1.0
            add(r, -np.inf, 0.0)
            r = np.zeros(dim)
            r[ix_c(t, i)] = 1.0
            r[ix_y(t, i)] = -1.0
            add(r, -np.inf, 0.0)

    k_max = min(int(SQUAD_SIZE), int(MAX_FT) + int(max_hits))

    if not from_owned:
        # Free open: f_after[0] = 1
        r = np.zeros(dim)
        r[ix_f(0)] = 1.0
        add(r, 1.0, 1.0)

    for t in t_xfer:
        # Previous ownership: owned vector (t=0 from_owned) or x[t-1].
        for i in range(n):
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            r[ix_x(t, i)] = -1.0
            if from_owned and t == 0:
                add(r, -float(owned[i]), np.inf)  # buy >= x - owned
            else:
                r[ix_x(t - 1, i)] = 1.0
                add(r, 0.0, np.inf)  # buy >= x_t - x_{t-1}
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            r[ix_x(t, i)] = -1.0
            add(r, -np.inf, 0.0)
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            if from_owned and t == 0:
                add(r, -np.inf, 1.0 - float(owned[i]))  # buy <= 1 - owned
            else:
                r[ix_x(t - 1, i)] = 1.0
                add(r, -np.inf, 1.0)

        # f_before for this step
        # free-open / later weeks: f[t-1]; from_owned week 0: constant ft0
        r = np.zeros(dim)
        r[ix_u(t)] = 1.0
        for i in range(n):
            r[ix_buy(t, i)] = -1.0
        add(r, -np.inf, 0.0)  # u <= k
        if from_owned and t == 0:
            r = np.zeros(dim)
            r[ix_u(t)] = 1.0
            add(r, -np.inf, float(ft0))  # u <= ft0
        else:
            r = np.zeros(dim)
            r[ix_u(t)] = 1.0
            r[ix_f(t - 1)] = -1.0
            add(r, -np.inf, 0.0)  # u <= f[t-1]

        r = np.zeros(dim)
        r[ix_e(t)] = 1.0
        r[ix_u(t)] = 1.0
        for i in range(n):
            r[ix_buy(t, i)] = -1.0
        add(r, 0.0, 0.0)  # e = k - u

        # Must-spend: e = max(0, k - f_before)
        if from_owned and t == 0:
            # f_before = ft0 constant
            r = np.zeros(dim)
            for i in range(n):
                r[ix_buy(t, i)] = 1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, -np.inf, float(ft0))  # k - ft0 <= k_max·a
            r = np.zeros(dim)
            for i in range(n):
                r[ix_buy(t, i)] = 1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, float(ft0 + 1 - k_max), np.inf)  # k - ft0 >= 1 - k_max(1-a)
            r = np.zeros(dim)
            r[ix_e(t)] = 1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, -np.inf, 0.0)
            r = np.zeros(dim)
            r[ix_e(t)] = 1.0
            for i in range(n):
                r[ix_buy(t, i)] = -1.0
            r[ix_hit(t)] = float(MAX_FT)
            add(r, -np.inf, float(MAX_FT) - float(ft0))  # e - k + ft0 + M a <= M
        else:
            r = np.zeros(dim)
            for i in range(n):
                r[ix_buy(t, i)] = 1.0
            r[ix_f(t - 1)] = -1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, -np.inf, 0.0)
            r = np.zeros(dim)
            for i in range(n):
                r[ix_buy(t, i)] = 1.0
            r[ix_f(t - 1)] = -1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, float(1 - k_max), np.inf)
            r = np.zeros(dim)
            r[ix_e(t)] = 1.0
            r[ix_hit(t)] = -float(k_max)
            add(r, -np.inf, 0.0)
            r = np.zeros(dim)
            r[ix_e(t)] = 1.0
            for i in range(n):
                r[ix_buy(t, i)] = -1.0
            r[ix_f(t - 1)] = 1.0
            r[ix_hit(t)] = float(MAX_FT)
            add(r, -np.inf, float(MAX_FT))

        r = np.zeros(dim)
        for i in range(n):
            r[ix_buy(t, i)] = 1.0
        add(r, 0.0, float(k_max))

        # f_after[t] = min(MAX_FT, f_before - u + 1)
        if from_owned and t == 0:
            r = np.zeros(dim)
            r[ix_f(t)] = 1.0
            r[ix_u(t)] = 1.0
            r[ix_over(t)] = 1.0
            add(r, float(ft0 + 1), float(ft0 + 1))  # f + u + over = ft0 + 1
            r = np.zeros(dim)
            r[ix_u(t)] = 1.0
            r[ix_over(t)] = 1.0
            add(r, float(ft0 + 1 - MAX_FT), np.inf)  # over >= ft0 - u + 1 - MAX
            r = np.zeros(dim)
            r[ix_over(t)] = float(MAX_FT)
            add(r, -np.inf, float(ft0))  # MAX_FT * over <= ft0
        else:
            r = np.zeros(dim)
            r[ix_f(t)] = 1.0
            r[ix_f(t - 1)] = -1.0
            r[ix_u(t)] = 1.0
            r[ix_over(t)] = 1.0
            add(r, 1.0, 1.0)
            r = np.zeros(dim)
            r[ix_f(t - 1)] = -1.0
            r[ix_u(t)] = 1.0
            r[ix_over(t)] = 1.0
            add(r, float(1 - MAX_FT), np.inf)
            r = np.zeros(dim)
            r[ix_over(t)] = float(MAX_FT)
            r[ix_f(t - 1)] = -1.0
            add(r, -np.inf, 0.0)

    lb = np.zeros(dim)
    ub = np.ones(dim)
    integrality = np.ones(dim, dtype=int)
    for t in range(H):
        lb[ix_f(t)] = 1.0 if (from_owned or t > 0) else 0.0
        ub[ix_f(t)] = float(MAX_FT)
    for t in t_xfer:
        ub[ix_u(t)] = float(MAX_FT)
        ub[ix_e(t)] = float(k_max)

    res = milp(
        c=c_obj,
        constraints=LinearConstraint(np.vstack(rows), np.asarray(lo), np.asarray(hi)),
        bounds=Bounds(lb, ub),
        integrality=integrality,
        options={
            "time_limit": float(time_limit),
            "mip_rel_gap": 1e-8,
        },
    )
    if not res.success or res.x is None:
        raise RuntimeError(f"path transfer plan MILP failed: {res.message}")

    x = res.x
    weeks: list[dict[str, Any]] = []
    for t, gw in enumerate(gws):
        squad_idx = [i for i in range(n) if x[ix_x(t, i)] > 0.5]
        xi_idx = [i for i in range(n) if x[ix_y(t, i)] > 0.5]
        cap_idx = [i for i in range(n) if x[ix_c(t, i)] > 0.5]
        if len(squad_idx) != 15 or len(xi_idx) != 11 or len(cap_idx) != 1:
            raise RuntimeError(
                f"bad plan at GW{gw}: squad={len(squad_idx)} xi={len(xi_idx)} cap={len(cap_idx)}"
            )
        if t not in step_of:
            n_tr, hits, ft_b, ft_a = 0, 0, 0, int(round(x[ix_f(0)]))
        else:
            buys = [i for i in range(n) if x[ix_buy(t, i)] > 0.5]
            n_tr = len(buys)
            hits = int(round(x[ix_e(t)])) * int(HIT_COST)
            if from_owned and t == 0:
                ft_b = int(ft0)
            else:
                ft_b = int(round(x[ix_f(t - 1)]))
            ft_a = int(round(x[ix_f(t)]))
        xi_score = float(sum(scores[t, i] for i in xi_idx))
        cap_score = float(scores[t, cap_idx[0]]) if use_cap else 0.0
        weeks.append(
            {
                "gw": int(gw),
                "player_ids": [str(ids[i]) for i in squad_idx],
                "names": [str(names[i]) for i in squad_idx],
                "xi_ids": [str(ids[i]) for i in xi_idx],
                "captain_id": str(ids[cap_idx[0]]),
                "n_transfers": int(n_tr),
                "hits": int(hits),
                "ft_before": int(ft_b),
                "ft_after": int(ft_a),
                "xi_score": xi_score,
                "captain_score": cap_score,
                "week_score": xi_score + cap_score,
                "value_captain": use_cap,
            }
        )

    obj = float(-res.fun) if res.fun is not None else float("nan")
    terminal_ft = int(weeks[-1]["ft_after"]) if weeks else 0
    bank_pot = {
        "in_objective": bool(use_pe),
        "ft_pe_nu": float(nu),
        "terminal_ft": terminal_ft,
        "by_gw": [
            {
                "gw": int(w["gw"]),
                "ft_before": int(w["ft_before"]),
                "ft_after": int(w["ft_after"]),
                "n_transfers": int(w["n_transfers"]),
                "hits": int(w["hits"]),
            }
            for w in weeks
        ],
    }
    return {
        "weeks": weeks,
        "objective": obj,
        "n_candidates": n,
        "horizon": H,
        "status": str(res.message),
        "ft_pe_nu": nu,
        "terminal_ft": terminal_ft,
        "terminal_pe": float(
            ft_bank_potential(terminal_ft, nu=nu) if weeks and use_pe else 0.0
        ),
        "bank_pot": bank_pot,
        "from_owned": bool(from_owned),
        "start_ft": int(ft0) if from_owned else None,
        "value_captain": use_cap,
    }
