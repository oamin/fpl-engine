"""Glue: L4+Opta centreline boards → joint path transfer plan.

Full-board candidates (no top-48), ``xi_ft`` (``value_captain=False``),
``from_owned`` seeded 15. Optional Opta-far exponential decay toward 0.

  from src.strategy.centreline_path_plan import (
      apply_opta_far_decay,
      plan_centreline_path,
  )
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from src.models.path_transfer_plan import (
    DEFAULT_TOP_N,
    pick_path_transfer_plan,
    score_matrix_for_candidates,
    select_plan_candidates,
)
from src.models.multistage_mc_milp import DEFAULT_FT_PE_NU, MAX_HITS_SNAPSHOT
from src.rules.fpl_2026 import (
    HIT_COST,
    SQUAD_SIZE,
    advance_ft,
    hit_cost,
    sell_price,
    xi_legal,
)
from src.strategy.l3_centreline_horizon import NEAR_DEPTH
from src.strategy.layer3_week import HeldState, TransferWeekPlan
from src.strategy.layer4_mupost_priced import COMPONENT_MU_COLS

# Pre-registered: protocol.l4_opta_joint_path_climb
DEFAULT_DECAY_RHO = 0.85
DEFAULT_DECAY_TARGET = 0.0  # toward zero (not a positional mean)

_DECAY_COLS = (
    "mu_post",
    "score_xp_v2",
    "xp",
    "xp_se",
) + tuple(COMPONENT_MU_COLS)


def apply_opta_far_decay(
    pools: Sequence[pd.DataFrame],
    *,
    near_depth: int = NEAR_DEPTH,
    rho: float = DEFAULT_DECAY_RHO,
    target: float = DEFAULT_DECAY_TARGET,
) -> list[pd.DataFrame]:
    """Decay far-horizon xp toward ``target``: factor ``ρ^(t−near+1)`` for ``t≥near``.

    Near boards ``0 .. near-1`` are returned unchanged. Default target is 0
    (components shrink toward zero). A non-zero target uses
    ``target + (mu − target) * factor``.
    """
    if not pools:
        return []
    near = int(near_depth)
    if near < 0:
        raise ValueError(f"near_depth must be ≥0, got {near}")
    r = float(rho)
    if not (0.0 < r <= 1.0):
        raise ValueError(f"rho must be in (0, 1], got {r}")
    tgt = float(target)
    out: list[pd.DataFrame] = []
    for t, pool in enumerate(pools):
        if t < near:
            out.append(pool.copy())
            continue
        factor = r ** (t - near + 1)
        q = pool.copy()
        for col in _DECAY_COLS:
            if col not in q.columns:
                continue
            raw = pd.to_numeric(q[col], errors="coerce").fillna(0.0).astype(float)
            if tgt == 0.0:
                q[col] = raw * factor
            else:
                q[col] = tgt + (raw - tgt) * factor
        out.append(q)
    return out


def _ensure_team_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "team_norm" not in out.columns:
        if "club" in out.columns:
            out["team_norm"] = out["club"]
        elif "team" in out.columns:
            out["team_norm"] = out["team"]
        else:
            out["team_norm"] = "unknown"
    # Held-row patches sometimes leave NaN team_norm with a valid club.
    if "club" in out.columns:
        miss = out["team_norm"].isna() | (out["team_norm"].astype(str) == "nan")
        out.loc[miss, "team_norm"] = out.loc[miss, "club"]
    out["team_norm"] = out["team_norm"].fillna("unknown").astype(str)
    if "team" not in out.columns:
        out["team"] = out["team_norm"]
    else:
        tmiss = out["team"].isna() | (out["team"].astype(str) == "nan")
        out.loc[tmiss, "team"] = out.loc[tmiss, "team_norm"]
        out["team"] = out["team"].fillna(out["team_norm"]).astype(str)
    if "club" not in out.columns:
        out["club"] = out["team_norm"]
    else:
        cmiss = out["club"].isna() | (out["club"].astype(str) == "nan")
        out.loc[cmiss, "club"] = out.loc[cmiss, "team_norm"]
        out["club"] = out["club"].fillna(out["team_norm"]).astype(str)
    return out


def _full_board_candidates(
    week0: pd.DataFrame,
    scores_by_gw: dict[int, np.ndarray],
    gws: Sequence[int],
    *,
    force_ids: set[str],
) -> pd.DataFrame:
    """All week-0 players (bypass DEFAULT_TOP_N=48), forcing owned ids."""
    pool = _ensure_team_cols(week0)
    # top_n ≥ universe size keeps everyone after position floors.
    top_n = max(int(len(pool)), int(DEFAULT_TOP_N), int(SQUAD_SIZE))
    return select_plan_candidates(
        pool,
        scores_by_gw,
        gws=list(gws),
        buy_ids=None,
        top_n=top_n,
        force_ids=force_ids,
    )


def _scores_by_gw_from_pools(
    pools: Sequence[pd.DataFrame],
    gws: Sequence[int],
    *,
    mu_col: str = "mu_post",
) -> dict[int, np.ndarray]:
    """Map decision-week player order → per-GW μ arrays (missing → 0)."""
    if not pools:
        raise ValueError("need at least one pool")
    base = pools[0]
    base_ids = base["player_id"].astype(str).to_numpy()
    out: dict[int, np.ndarray] = {}
    for t, gw in enumerate(gws):
        if t >= len(pools):
            out[int(gw)] = np.zeros(len(base_ids), dtype=float)
            continue
        board = pools[t]
        if mu_col not in board.columns:
            raise RuntimeError(f"pool t={t} missing {mu_col!r}")
        by = board.set_index(board["player_id"].astype(str))[mu_col]
        by = pd.to_numeric(by, errors="coerce").fillna(0.0)
        out[int(gw)] = np.asarray(
            [float(by[pid]) if pid in by.index else 0.0 for pid in base_ids],
            dtype=float,
        )
    return out


def week0_to_transfer_plan(
    path_week: dict[str, Any],
    held: HeldState,
    week0_pool: pd.DataFrame,
    *,
    mu_col: str = "mu_post",
    status: str = "path_ok",
) -> TransferWeekPlan:
    """Commit path week-0 into a ``TransferWeekPlan`` for ``_step_held``.

    Path ``hits`` are hit **points**; this helper rebuilds count / points from
    buy/sell vs ``held.ft`` and computes ``bank_after`` from sell prices.
    """
    pool = _ensure_team_cols(week0_pool).copy()
    pool["player_id"] = pool["player_id"].astype(str)
    by = pool.drop_duplicates("player_id", keep="last").set_index("player_id", drop=True)
    new_ids = [str(p) for p in path_week["player_ids"]]
    xi_ids = [str(p) for p in path_week["xi_ids"]]
    captain_id = str(path_week["captain_id"])
    if len(new_ids) != SQUAD_SIZE:
        raise RuntimeError(f"path week0 squad size {len(new_ids)}")
    if len(xi_ids) != 11:
        raise RuntimeError(f"path week0 XI size {len(xi_ids)}")
    missing = [p for p in new_ids if p not in by.index]
    if missing:
        raise RuntimeError(f"path buys missing from week0 pool: {missing[:5]}")

    old = set(held.player_ids)
    new = set(new_ids)
    bought = tuple(sorted(new - old))
    sold = tuple(sorted(old - new))
    n_transfers = len(bought)
    if n_transfers != len(sold):
        raise RuntimeError(f"buy/sell mismatch: bought={bought} sold={sold}")
    hits = max(0, n_transfers - int(held.ft))
    hit_points = HIT_COST * hits
    if hit_cost(held.ft, n_transfers) != hit_points:
        raise RuntimeError("path week0 hit identity failed")

    revenue = sum(
        sell_price(int(held.purchase[pid]), int(by.loc[pid, "value"])) for pid in sold
    )
    spend = sum(int(by.loc[pid, "value"]) for pid in bought)
    # Path MILP budgets each GW on Σ market value ≤ BUDGET, not sell-price
    # cash. Clamp at 0 so the climb can still step held state.
    bank_after = max(0, int(held.bank + revenue - spend))

    squad = by.loc[list(new_ids)].reset_index()
    xi = by.loc[list(xi_ids)].reset_index()
    squad["player_id"] = squad["player_id"].astype(str)
    xi["player_id"] = xi["player_id"].astype(str)

    if mu_col in squad.columns:
        squad["mu"] = pd.to_numeric(squad[mu_col], errors="coerce").fillna(0.0)
        xi["mu"] = pd.to_numeric(xi[mu_col], errors="coerce").fillna(0.0)
    else:
        squad["mu"] = 0.0
        xi["mu"] = 0.0

    squad = squad.assign(
        in_xi=squad["player_id"].isin(xi_ids),
        is_captain=squad["player_id"] == captain_id,
    )
    xi = xi.copy()
    xi["is_captain"] = xi["player_id"] == captain_id

    if not xi_legal(xi["position"].astype(str).tolist()):
        raise RuntimeError("path week0 XI illegal")

    n_def = int((xi["position"] == "DEF").sum())
    n_mid = int((xi["position"] == "MID").sum())
    n_fwd = int((xi["position"] == "FWD").sum())
    xi_mu = float(xi["mu"].sum())
    # Captain μ not valued (xi_ft); keep 0 in the TransferWeekPlan objective piece.
    captain_mu = 0.0
    week0_obj = xi_mu - float(hit_points)

    return TransferWeekPlan(
        squad=squad.reset_index(drop=True),
        xi=xi.reset_index(drop=True),
        captain_id=captain_id,
        formation=(n_def, n_mid, n_fwd),
        objective=float(week0_obj),
        xi_mu=float(xi_mu),
        captain_mu=float(captain_mu),
        hits=int(hits),
        n_transfers=int(n_transfers),
        hit_points=int(hit_points),
        bank_after=int(bank_after),
        ft_after=int(advance_ft(held.ft, n_transfers)),
        sold=sold,
        bought=bought,
        cost=int(pd.to_numeric(squad["value"], errors="coerce").fillna(0).sum()),
        status=str(status),
        mu_col=mu_col,
    )


def plan_centreline_path(
    pools: Sequence[pd.DataFrame],
    held: HeldState,
    *,
    mu_col: str = "mu_post",
    decay: bool = False,
    near_depth: int = NEAR_DEPTH,
    rho: float = DEFAULT_DECAY_RHO,
    decay_target: float = DEFAULT_DECAY_TARGET,
    gamma: float = 1.0,
    max_hits: int = MAX_HITS_SNAPSHOT,
    ft_pe_nu: float = DEFAULT_FT_PE_NU,
    time_limit: float = 180.0,
    decision_gw: int | None = None,
) -> tuple[TransferWeekPlan, dict[str, Any]]:
    """Solve joint path on centreline pools; return week-0 plan + raw path dict.

    Candidates = full week-0 board. Ownership = ``held`` (``from_owned``).
    Objective = Σ XI − hits (``value_captain=False``).
    """
    if not pools:
        raise ValueError("need at least one pool")
    boards = list(pools)
    if decay:
        boards = apply_opta_far_decay(
            boards, near_depth=near_depth, rho=rho, target=decay_target
        )
    week0 = _ensure_team_cols(boards[0])
    if mu_col not in week0.columns:
        raise RuntimeError(f"week0 missing {mu_col!r}")

    gw0 = int(decision_gw) if decision_gw is not None else 0
    gws = [gw0 + t for t in range(len(boards))]
    scores_by_gw = _scores_by_gw_from_pools(boards, gws, mu_col=mu_col)
    owned = [str(p) for p in held.player_ids]
    cands = _full_board_candidates(
        week0, scores_by_gw, gws, force_ids=set(owned)
    )
    S = score_matrix_for_candidates(cands, week0, scores_by_gw, gws)
    raw = pick_path_transfer_plan(
        cands,
        S,
        gws,
        gamma=float(gamma),
        max_hits=int(max_hits),
        ft_pe_nu=float(ft_pe_nu),
        time_limit=float(time_limit),
        start_owned_ids=owned,
        start_ft=int(held.ft),
        value_captain=False,
    )
    if not raw.get("weeks"):
        raise RuntimeError("path plan returned no weeks")
    week0_plan = week0_to_transfer_plan(
        raw["weeks"][0],
        held,
        week0,
        mu_col=mu_col,
        status=str(raw.get("status", "path_ok")),
    )
    meta = {
        "objective": float(raw.get("objective", float("nan"))),
        "n_candidates": int(raw.get("n_candidates", len(cands))),
        "horizon": int(raw.get("horizon", len(boards))),
        "decay": bool(decay),
        "rho": float(rho) if decay else None,
        "near_depth": int(near_depth),
        "from_owned": True,
        "value_captain": False,
        "ft_pe_nu": float(ft_pe_nu),
        "status": str(raw.get("status", "")),
    }
    return week0_plan, {**raw, "meta": meta}
