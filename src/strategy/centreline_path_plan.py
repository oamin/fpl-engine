"""Glue: L4+Opta centreline boards → joint path transfer plan.

Full-board candidates (no top-48); default ``xi_ft`` (``value_captain=False``).
Optional ``value_captain=True`` for ``xi_plus_captain`` screens.
``from_owned`` seeded 15. Optional Opta-far exponential decay toward 0.

  from src.strategy.centreline_path_plan import (
      apply_opta_far_decay,
      plan_centreline_path,
  )
"""

from __future__ import annotations

from collections import Counter
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
# Pre-registered: protocol.l4_opta_joint_path_mu_band_roll
DEFAULT_MU_BAND_STRANDS = 8
DEFAULT_MU_BAND_A = 2.0
DEFAULT_MU_BAND_B = 2.0

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
    target: float | str = DEFAULT_DECAY_TARGET,
) -> list[pd.DataFrame]:
    """Decay far-horizon xp toward ``target``: factor ``ρ^(t−near+1)`` for ``t≥near``.

    Near boards ``0 .. near-1`` are returned unchanged.

    ``target`` modes:

    * ``0`` / ``"zero"`` — shrink toward 0 (``mu * factor``).
    * scalar float — ``target + (mu − target) * factor``.
    * ``"pos_mean"`` — per column, per row: target = mean of that column among
      same ``position`` on the board (falls back to board mean if position
      missing).
    """
    if not pools:
        return []
    near = int(near_depth)
    if near < 0:
        raise ValueError(f"near_depth must be ≥0, got {near}")
    r = float(rho)
    if not (0.0 < r <= 1.0):
        raise ValueError(f"rho must be in (0, 1], got {r}")

    mode: str
    scalar_tgt = 0.0
    if isinstance(target, str):
        mode = str(target).strip().lower()
        if mode in ("0", "zero", ""):
            mode = "zero"
        elif mode not in ("pos_mean", "zero"):
            raise ValueError(
                f"target str must be 'zero' or 'pos_mean', got {target!r}"
            )
    else:
        scalar_tgt = float(target)
        mode = "zero" if abs(scalar_tgt) < 1e-15 else "scalar"

    out: list[pd.DataFrame] = []
    for t, pool in enumerate(pools):
        if t < near:
            out.append(pool.copy())
            continue
        factor = r ** (t - near + 1)
        q = pool.copy()
        pos = None
        if mode == "pos_mean" and "position" in q.columns:
            pos = q["position"].astype(str)
        for col in _DECAY_COLS:
            if col not in q.columns:
                continue
            raw = pd.to_numeric(q[col], errors="coerce").fillna(0.0).astype(float)
            if mode == "zero":
                q[col] = raw * factor
            elif mode == "scalar":
                q[col] = scalar_tgt + (raw - scalar_tgt) * factor
            else:
                # pos_mean
                if pos is None:
                    tgt_series = pd.Series(float(raw.mean()), index=q.index)
                else:
                    board_mean = float(raw.mean()) if len(raw) else 0.0
                    means = raw.groupby(pos).transform("mean")
                    tgt_series = means.fillna(board_mean)
                q[col] = tgt_series.to_numpy(float) + (
                    raw.to_numpy(float) - tgt_series.to_numpy(float)
                ) * factor
        out.append(q)
    return out


def roll_mu_within_band(
    pools: Sequence[pd.DataFrame],
    rng: np.random.Generator,
    *,
    mu_col: str = "mu_post",
    se_col: str = "xp_se",
    band_a: float = DEFAULT_MU_BAND_A,
    band_b: float = DEFAULT_MU_BAND_B,
) -> list[pd.DataFrame]:
    """Independent Gaussian roll of ``mu_col`` inside ``[μ−aσ, μ+bσ]`` per row.

    ``σ`` is ``se_col`` (product ``xp_se``). Call **after** Opta-far decay so
    both μ and σ have already shrunk toward 0 on far boards.
    """
    a = float(band_a)
    b = float(band_b)
    if a < 0.0 or b < 0.0:
        raise ValueError(f"band_a/band_b must be ≥0, got {a}, {b}")
    out: list[pd.DataFrame] = []
    for pool in pools:
        q = pool.copy()
        if mu_col not in q.columns:
            raise RuntimeError(f"roll board missing {mu_col!r}")
        mu = pd.to_numeric(q[mu_col], errors="coerce").fillna(0.0).to_numpy(float)
        if se_col in q.columns:
            se = pd.to_numeric(q[se_col], errors="coerce").fillna(0.0).to_numpy(float)
        else:
            se = np.zeros(len(q), dtype=float)
        se = np.maximum(se, 0.0)
        z = rng.normal(0.0, 1.0, size=len(q))
        tilde = mu + se * z
        lo = mu - a * se
        hi = mu + b * se
        q[mu_col] = np.clip(tilde, lo, hi)
        out.append(q)
    return out


def _modal_week0_from_strands(
    strand_plans: Sequence[TransferWeekPlan],
    strand_raw: Sequence[dict[str, Any]],
    held: HeldState,
    week0_pool: pd.DataFrame,
    *,
    mu_col: str,
    value_captain: bool,
) -> tuple[TransferWeekPlan, dict[str, Any]]:
    """Commit week-0 by modal squad among strands (tie → highest path obj)."""
    if not strand_plans:
        raise ValueError("need at least one strand plan")
    squad_keys = [
        frozenset(str(p) for p in plan.squad["player_id"].astype(str).tolist())
        for plan in strand_plans
    ]
    counts = Counter(squad_keys)
    best_count = max(counts.values())
    contenders = [k for k, c in counts.items() if c == best_count]
    if len(contenders) == 1:
        chosen_key = contenders[0]
    else:
        # Tie: highest mean path objective among strands with that squad.
        best_obj = -float("inf")
        chosen_key = contenders[0]
        for key in contenders:
            objs = [
                float(raw.get("objective", float("nan")))
                for key_i, raw in zip(squad_keys, strand_raw)
                if key_i == key
            ]
            objs = [o for o in objs if o == o]
            mean_obj = float(np.mean(objs)) if objs else -float("inf")
            if mean_obj > best_obj:
                best_obj = mean_obj
                chosen_key = key

    idx = [i for i, k in enumerate(squad_keys) if k == chosen_key]
    # Modal XI among matching strands.
    xi_keys = [
        frozenset(
            str(p) for p in strand_plans[i].xi["player_id"].astype(str).tolist()
        )
        for i in idx
    ]
    xi_counts = Counter(xi_keys)
    xi_best = max(xi_counts.values())
    xi_contenders = [k for k, c in xi_counts.items() if c == xi_best]
    if len(xi_contenders) == 1:
        xi_key = xi_contenders[0]
    else:
        # Prefer XI from the highest-objective matching strand.
        best_i = max(idx, key=lambda i: float(strand_raw[i].get("objective", -1e30)))
        xi_key = frozenset(
            str(p) for p in strand_plans[best_i].xi["player_id"].astype(str).tolist()
        )
    xi_idx = [
        i
        for i in idx
        if frozenset(
            str(p) for p in strand_plans[i].xi["player_id"].astype(str).tolist()
        )
        == xi_key
    ]
    cap_counts = Counter(str(strand_plans[i].captain_id) for i in xi_idx)
    captain_id = cap_counts.most_common(1)[0][0]
    # Rebuild from a representative strand that matches squad+xi+cap when possible.
    rep = None
    for i in xi_idx:
        if str(strand_plans[i].captain_id) == captain_id:
            rep = i
            break
    if rep is None:
        rep = xi_idx[0]
    plan = strand_plans[rep]
    path_week = {
        "player_ids": [str(p) for p in plan.squad["player_id"].astype(str).tolist()],
        "xi_ids": [str(p) for p in plan.xi["player_id"].astype(str).tolist()],
        "captain_id": captain_id,
    }
    week0 = week0_to_transfer_plan(
        path_week,
        held,
        week0_pool,
        mu_col=mu_col,
        status="mu_band_modal",
        value_captain=value_captain,
    )
    meta = {
        "n_strands": int(len(strand_plans)),
        "modal_squad_count": int(best_count),
        "n_distinct_squads": int(len(counts)),
        "modal_xi_count": int(xi_best),
        "n_distinct_xi_among_modal_squad": int(len(xi_counts)),
        "strand_objectives": [
            float(r.get("objective", float("nan"))) for r in strand_raw
        ],
        "commit_strand": int(rep),
    }
    return week0, meta


def plan_centreline_path_mu_band_roll(
    pools: Sequence[pd.DataFrame],
    held: HeldState,
    *,
    mu_col: str = "mu_post",
    se_col: str = "xp_se",
    n_strands: int = DEFAULT_MU_BAND_STRANDS,
    band_a: float = DEFAULT_MU_BAND_A,
    band_b: float = DEFAULT_MU_BAND_B,
    seed: int = 0,
    decay: bool = True,
    near_depth: int = NEAR_DEPTH,
    rho: float = DEFAULT_DECAY_RHO,
    decay_target: float | str = DEFAULT_DECAY_TARGET,
    gamma: float = 1.0,
    max_hits: int = MAX_HITS_SNAPSHOT,
    ft_pe_nu: float = DEFAULT_FT_PE_NU,
    value_captain: bool = True,
    time_limit: float = 180.0,
    mip_rel_gap: float = 1e-3,
    decision_gw: int | None = None,
) -> tuple[TransferWeekPlan, dict[str, Any]]:
    """N independent μ-band path solves; commit modal week-0 squad.

    Decay μ and ``xp_se`` first (ρ→0), then roll inside bands, then solve each
    strand with ``decay=False`` (already applied).
    """
    n = int(n_strands)
    if n < 1:
        raise ValueError(f"n_strands must be ≥1, got {n}")
    boards = list(pools)
    if decay:
        boards = apply_opta_far_decay(
            boards, near_depth=near_depth, rho=rho, target=decay_target
        )
    week0_mean = _ensure_team_cols(boards[0])
    rng = np.random.default_rng(int(seed))
    strand_plans: list[TransferWeekPlan] = []
    strand_raw: list[dict[str, Any]] = []
    for s in range(n):
        # Independent stream per strand (stable if n changes upward).
        strand_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        rolled = roll_mu_within_band(
            boards,
            strand_rng,
            mu_col=mu_col,
            se_col=se_col,
            band_a=band_a,
            band_b=band_b,
        )
        plan, raw = plan_centreline_path(
            rolled,
            held,
            mu_col=mu_col,
            decay=False,
            near_depth=near_depth,
            rho=rho,
            decay_target=decay_target,
            gamma=gamma,
            max_hits=max_hits,
            ft_pe_nu=ft_pe_nu,
            value_captain=value_captain,
            time_limit=time_limit,
            mip_rel_gap=mip_rel_gap,
            decision_gw=decision_gw,
        )
        strand_plans.append(plan)
        strand_raw.append(raw)
    week0, commit_meta = _modal_week0_from_strands(
        strand_plans,
        strand_raw,
        held,
        week0_mean,
        mu_col=mu_col,
        value_captain=value_captain,
    )
    meta = {
        "objective": float(np.nanmean(commit_meta["strand_objectives"])),
        "n_candidates": int(strand_raw[0].get("n_candidates", 0)) if strand_raw else 0,
        "horizon": int(len(boards)),
        "decay": bool(decay),
        "rho": float(rho) if decay else None,
        "decay_target": decay_target if decay else None,
        "near_depth": int(near_depth),
        "from_owned": True,
        "value_captain": bool(value_captain),
        "ft_pe_nu": float(ft_pe_nu),
        "mu_band": {
            "n_strands": n,
            "band_a": float(band_a),
            "band_b": float(band_b),
            "se_col": se_col,
            "seed": int(seed),
            **commit_meta,
        },
        "status": "mu_band_modal",
    }
    return week0, {"meta": meta, "strand_raw": strand_raw}


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
    value_captain: bool = True,
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
    captain_mu = (
        float(xi.loc[xi["player_id"] == captain_id, "mu"].sum())
        if bool(value_captain)
        else 0.0
    )
    week0_obj = xi_mu + float(captain_mu) - float(hit_points)

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
    decay_target: float | str = DEFAULT_DECAY_TARGET,
    gamma: float = 1.0,
    max_hits: int = MAX_HITS_SNAPSHOT,
    ft_pe_nu: float = DEFAULT_FT_PE_NU,
    value_captain: bool = True,
    time_limit: float = 180.0,
    mip_rel_gap: float = 1e-3,
    decision_gw: int | None = None,
) -> tuple[TransferWeekPlan, dict[str, Any]]:
    """Solve joint path on centreline pools; return week-0 plan + raw path dict.

    Candidates = full week-0 board. Ownership = ``held`` (``from_owned``).
    Product default objective = Σ(XI+cap) − hits (``value_captain=True``;
    promoted ``l4_opta_joint_path_captain_climb`` 2026-10-31). Pass
    ``value_captain=False`` for transfer-rule isolation screens.
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
    use_cap = bool(value_captain)
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
        value_captain=use_cap,
        mip_rel_gap=float(mip_rel_gap),
    )
    if not raw.get("weeks"):
        raise RuntimeError("path plan returned no weeks")
    week0_plan = week0_to_transfer_plan(
        raw["weeks"][0],
        held,
        week0,
        mu_col=mu_col,
        status=str(raw.get("status", "path_ok")),
        value_captain=use_cap,
    )
    meta = {
        "objective": float(raw.get("objective", float("nan"))),
        "n_candidates": int(raw.get("n_candidates", len(cands))),
        "horizon": int(raw.get("horizon", len(boards))),
        "decay": bool(decay),
        "rho": float(rho) if decay else None,
        "decay_target": decay_target if decay else None,
        "near_depth": int(near_depth),
        "from_owned": True,
        "value_captain": use_cap,
        "ft_pe_nu": float(ft_pe_nu),
        "status": str(raw.get("status", "")),
    }
    return week0_plan, {**raw, "meta": meta}
