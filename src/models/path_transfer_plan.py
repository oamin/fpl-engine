"""Joint multi-GW transfer plan MILP for one cascade path strand.

Given per-GW player scores on a fully rolled strand, choose a legal 15 + XI +
captain for **every** horizon GW and the transfers that link them, subject to
FT bank dynamics and −4/hit. Opening GW is a free rebuild (wildcard-like).

This is the path planner the cascade FT objective needs: the horizon is a
transfer *schedule*, not a myopic single-deadline solve on discounted μ.
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
# Per-buy friction so free FTs are not spent on zero-gain reshuffles.
# Large enough to beat HiGHS default MIP gap on flat boards.
TRANSFER_EPS = 1e-2


def select_plan_candidates(
    pool: pd.DataFrame,
    scores_by_gw: Mapping[int, np.ndarray],
    *,
    gws: Sequence[int],
    buy_ids: set[str] | None,
    top_n: int = DEFAULT_TOP_N,
) -> pd.DataFrame:
    """Universe for the joint plan: buy filter ∩ top mean-score players."""
    df = pool.copy()
    if "player_id" not in df.columns:
        raise ValueError("pool needs player_id")
    df["player_id"] = df["player_id"].astype(str)
    if buy_ids is not None:
        df = df.loc[df["player_id"].isin(buy_ids)].copy()
    if df.empty:
        raise RuntimeError("no candidates after buy filter")
    # Align score vectors to df rows via original pool index if needed.
    mean = np.zeros(len(df), dtype=float)
    pid_to_local = {pid: i for i, pid in enumerate(df["player_id"])}
    # scores_by_gw arrays must be aligned to ``pool`` row order.
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
) -> dict[str, Any]:
    """Joint path MILP: free open + FT/hit schedule over all ``gws``.

    ``scores`` shape ``(H, n)`` aligned with ``gws`` and ``candidates``.

    When ``ft_pe_nu > 0``, adds terminal PE ``U(f_H)=ν·f_H`` (default
    ``ν=HIT_COST``: +4 per stored FT, +20 at cap). For linear ``U`` this equals
    the telescoping sum of week-to-week ΔPE (``f_0`` fixed after open). Holding
    below the cap grows next-deadline bank by +1 → +ν; at five the weekly grant
    adds +0 PE unless a transfer is used first. No multi-week ``W`` scale.
    """
    H = int(len(gws))
    n = int(len(candidates))
    if scores.shape != (H, n):
        raise ValueError(f"scores shape {scores.shape} != ({H}, {n})")
    if H < 1:
        raise ValueError("need at least one GW")

    values = pd.to_numeric(candidates["value"], errors="coerce").fillna(50.0).to_numpy(float)
    positions = candidates["position"].astype(str).to_numpy()
    clubs = candidates["team_norm"].astype(str).to_numpy() if "team_norm" in candidates.columns else candidates["team"].astype(str).to_numpy()
    ids = candidates["player_id"].astype(str).to_numpy()
    names = (
        candidates["name"].astype(str).to_numpy()
        if "name" in candidates.columns
        else ids.copy()
    )

    nu = float(ft_pe_nu)
    use_pe = nu > 0.0

    # Layout:
    # x | y | c | buy | u | e | f | over | hit_flag (1 iff k > f this week)
    nx = H * n
    ny = H * n
    nc = H * n
    n_buy = max(0, H - 1) * n
    n_u = max(0, H - 1)
    n_e = max(0, H - 1)
    n_f = H
    n_over = max(0, H - 1)
    n_hit = max(0, H - 1)
    dim = nx + ny + nc + n_buy + n_u + n_e + n_f + n_over + n_hit

    def ix_x(t: int, i: int) -> int:
        return t * n + i

    def ix_y(t: int, i: int) -> int:
        return nx + t * n + i

    def ix_c(t: int, i: int) -> int:
        return nx + ny + t * n + i

    def ix_buy(t: int, i: int) -> int:
        # t is week index 1..H-1 mapped to 0..H-2
        return nx + ny + nc + (t - 1) * n + i

    def ix_u(t: int) -> int:
        return nx + ny + nc + n_buy + (t - 1)

    def ix_e(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + (t - 1)

    def ix_f(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + t

    def ix_over(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + n_f + (t - 1)

    def ix_hit(t: int) -> int:
        return nx + ny + nc + n_buy + n_u + n_e + n_f + n_over + (t - 1)

    g = float(gamma)
    c_obj = np.zeros(dim)
    for t in range(H):
        w = g**t
        for i in range(n):
            c_obj[ix_y(t, i)] = -w * float(scores[t, i])
            c_obj[ix_c(t, i)] = -w * float(scores[t, i])
    for t in range(1, H):
        c_obj[ix_e(t)] = float(HIT_COST)
        for i in range(n):
            c_obj[ix_buy(t, i)] = float(TRANSFER_EPS)
    # Terminal bank PE: max ν·f_H → minimise −ν·f_H.
    # (Sum of week-to-week ΔPE = ν·(f_H − f_0); f_0 fixed at 1 after open.)
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
        # Squad size / budget / quotas / clubs
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

        # XI
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

        # Captain
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

    # Free open: f_after[0] = 1
    r = np.zeros(dim)
    r[ix_f(0)] = 1.0
    add(r, 1.0, 1.0)

    k_max = int(SQUAD_SIZE)  # transfers in a week cannot exceed squad size
    k_max = min(k_max, int(MAX_FT) + int(max_hits))

    for t in range(1, H):
        # buy linking
        for i in range(n):
            # buy >= x_t - x_{t-1}
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            r[ix_x(t, i)] = -1.0
            r[ix_x(t - 1, i)] = 1.0
            add(r, 0.0, np.inf)
            # buy <= x_t
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            r[ix_x(t, i)] = -1.0
            add(r, -np.inf, 0.0)
            # buy <= 1 - x_{t-1}
            r = np.zeros(dim)
            r[ix_buy(t, i)] = 1.0
            r[ix_x(t - 1, i)] = 1.0
            add(r, -np.inf, 1.0)

        # u <= k, u <= f_before=f_after[t-1]; e = k - u
        # k = sum buy
        # u - sum buy <= 0
        r = np.zeros(dim)
        r[ix_u(t)] = 1.0
        for i in range(n):
            r[ix_buy(t, i)] = -1.0
        add(r, -np.inf, 0.0)
        # u <= f_after[t-1]
        r = np.zeros(dim)
        r[ix_u(t)] = 1.0
        r[ix_f(t - 1)] = -1.0
        add(r, -np.inf, 0.0)
        # e - sum buy + u = 0  → e = k - u
        r = np.zeros(dim)
        r[ix_e(t)] = 1.0
        r[ix_u(t)] = 1.0
        for i in range(n):
            r[ix_buy(t, i)] = -1.0
        add(r, 0.0, 0.0)
        # Must spend FT bank before hits: e = max(0, k - f).
        # hit_flag a ∈ {0,1} with a=1 iff k ≥ f+1 (integer). Then:
        #   a=0 → e=0; a=1 → e ≤ k−f.
        # With e = k−u and u ≤ f (⇒ e ≥ k−f) this forces u = min(k, f).
        # Do NOT add the always-on cut e ≤ k−f: when k < f that is infeasible
        # and blocks bank growth (holds).
        r = np.zeros(dim)
        for i in range(n):
            r[ix_buy(t, i)] = 1.0
        r[ix_f(t - 1)] = -1.0
        r[ix_hit(t)] = -float(k_max)
        add(r, -np.inf, 0.0)  # k - f <= k_max·a
        r = np.zeros(dim)
        for i in range(n):
            r[ix_buy(t, i)] = 1.0
        r[ix_f(t - 1)] = -1.0
        r[ix_hit(t)] = -float(k_max)
        add(r, float(1 - k_max), np.inf)  # k - f >= 1 - k_max·(1−a)
        r = np.zeros(dim)
        r[ix_e(t)] = 1.0
        r[ix_hit(t)] = -float(k_max)
        add(r, -np.inf, 0.0)  # e <= k_max·a  (⇒ e=0 when a=0)
        # a=1 ⇒ e ≤ k−f; a=0 ⇒ vacuous (e − k + f ≤ MAX_FT).
        r = np.zeros(dim)
        r[ix_e(t)] = 1.0
        for i in range(n):
            r[ix_buy(t, i)] = -1.0
        r[ix_f(t - 1)] = 1.0
        r[ix_hit(t)] = float(MAX_FT)
        add(r, -np.inf, float(MAX_FT))  # e - k + f + MAX_FT·a <= MAX_FT
        # k <= k_max
        r = np.zeros(dim)
        for i in range(n):
            r[ix_buy(t, i)] = 1.0
        add(r, 0.0, float(k_max))

        # f_after[t] = min(MAX_FT, f_after[t-1] - u + 1)
        # raw = f[t-1] - u + 1 ∈ [1, MAX_FT+1]; over ∈ {0,1} absorbs the cap.
        # Equality: f = raw - over
        r = np.zeros(dim)
        r[ix_f(t)] = 1.0
        r[ix_f(t - 1)] = -1.0
        r[ix_u(t)] = 1.0
        r[ix_over(t)] = 1.0
        add(r, 1.0, 1.0)
        # over >= raw - MAX_FT → -f[t-1] + u + over >= 1 - MAX_FT
        r = np.zeros(dim)
        r[ix_f(t - 1)] = -1.0
        r[ix_u(t)] = 1.0
        r[ix_over(t)] = 1.0
        add(r, float(1 - MAX_FT), np.inf)
        # over = 0 unless f[t-1] == MAX_FT: MAX_FT * over <= f[t-1]
        r = np.zeros(dim)
        r[ix_over(t)] = float(MAX_FT)
        r[ix_f(t - 1)] = -1.0
        add(r, -np.inf, 0.0)

    # Bounds / integrality
    lb = np.zeros(dim)
    ub = np.ones(dim)
    integrality = np.ones(dim, dtype=int)
    for t in range(H):
        lb[ix_f(t)] = 1.0 if t > 0 else 0.0
        ub[ix_f(t)] = float(MAX_FT)
    for t in range(1, H):
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
        if t == 0:
            n_tr, hits, ft_b, ft_a = 0, 0, 0, int(round(x[ix_f(0)]))
        else:
            buys = [i for i in range(n) if x[ix_buy(t, i)] > 0.5]
            n_tr = len(buys)
            hits = int(round(x[ix_e(t)])) * int(HIT_COST)
            ft_b = int(round(x[ix_f(t - 1)]))
            ft_a = int(round(x[ix_f(t)]))
        xi_score = float(sum(scores[t, i] for i in xi_idx))
        cap_score = float(scores[t, cap_idx[0]])
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
            }
        )

    obj = float(-res.fun) if res.fun is not None else float("nan")
    terminal_ft = int(weeks[-1]["ft_after"]) if weeks else 0
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
    }
