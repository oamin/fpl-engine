"""Layer 5 Step 4 — joint beam over chips + transfers on μ boards.

Receding horizon: at each deadline, beam over as-of boards (γ=1),
commit only the week-0 action, then re-solve next week.

**Transfers (locked 2026-10-16):** depth-0 none/TC/BB use the same
``solve_horizon_transfers`` MILP as L4 (transfers now; XI over remaining
boards). Later beam depths are XI-only on the committed 15 — matching L4's
"transfers at t only" law. Prior beam used myopic ``solve_week_transfers``
each depth and lost −441 vs L4 with chips locked
(``reports/layer5_nochips_diagnose.md``).

Beam width K is a search budget (not an FPL rule). Nested property: larger K
cannot worsen best **planned** Σμ on the same expansion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS
from src.strategy.layer2_week import solve_week_sigma
from src.strategy.layer3_climb import _ensure_held_rows
from src.strategy.layer3_week import (
    DEFAULT_MAX_HITS,
    HeldState,
    solve_week_transfers,
)
from src.strategy.layer4_week import solve_horizon_transfers
from src.strategy.layer5_week import (
    NO_CHIP,
    PlannerState,
    advance_planner_state,
    bench_from_squad,
    held_copy,
    held_from_squad_frame,
    score_commit_week,
)

# Locked 2026-10-16 after horizon-MILP K-grid vs L4 (reports/layer5_vs_l4_k_rescreen.md).
# Prior default was 2 (reports/layer5_beam_k.md). K=8≡16 pooled; PI lock.
DEFAULT_BEAM_K = 8


@dataclass
class BeamAction:
    chip: str
    commit_held: HeldState
    xi: pd.DataFrame
    captain_id: str
    bench: pd.DataFrame
    n_transfers: int
    hit_points: int
    week_mu: float


@dataclass
class BeamNode:
    state: PlannerState
    cum_mu: float
    depth: int
    first_action: BeamAction | None = None
    path_chips: tuple[str, ...] = ()


@dataclass
class BeamPlan:
    best: BeamNode
    planned_mu: float
    K: int
    horizon_gws: tuple[int, ...]
    n_expanded: int
    runner_up_mu: float | None
    margin_vs_runner: float | None


def _mu_col(df: pd.DataFrame) -> str:
    if "mu_post" in df.columns:
        return "mu_post"
    if "mu" in df.columns:
        return "mu"
    raise ValueError("pool needs mu_post or mu")


def _with_mu(df: pd.DataFrame, mu_col: str) -> pd.DataFrame:
    out = df.copy()
    if mu_col != "mu":
        out["mu"] = pd.to_numeric(out[mu_col], errors="coerce").fillna(0.0)
    return out


def _rebuild_held(
    pool: pd.DataFrame,
    *,
    moments: dict[str, Any] | None,
    ft: int,
    time_limit: float,
) -> HeldState:
    work = pool.copy()
    if "mu_post" not in work.columns:
        raise ValueError("rebuild pool needs mu_post")
    splan = solve_week_sigma(
        work,
        mu_col="score_xp_v2" if "score_xp_v2" in work.columns else "mu_post",
        se_col="xp_se" if "xp_se" in work.columns else "xp_se",
        mode="none",
        moments_by_pos=moments,
        budget=BUDGET_TENTHS,
        time_limit=time_limit,
    )
    squad = splan.plan.squad.copy()
    squad["purchase"] = squad["value"].astype(int)
    # Attach mu for scoring
    mu_col = _mu_col(work)
    meta = work.set_index(work["player_id"].astype(str))
    squad["mu"] = squad["player_id"].astype(str).map(meta[mu_col]).astype(float)
    return held_from_squad_frame(squad, ft=int(ft))


def _xi_from_held(
    pool: pd.DataFrame,
    held: HeldState,
    *,
    time_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame, str, pd.DataFrame]:
    """Hold squad; pick XI+captain on μ (0 transfers)."""
    work = _ensure_held_rows(pool, held, pool)
    work = _with_mu(work, _mu_col(work))
    frozen = HeldState(held.player_ids, held.purchase, bank=held.bank, ft=0)
    plan = solve_week_transfers(
        work, frozen, mu_col="mu", max_hits=0, time_limit=time_limit
    )
    squad = plan.squad.copy()
    if "mu" not in squad.columns:
        squad["mu"] = squad["player_id"].astype(str).map(
            work.set_index(work["player_id"].astype(str))["mu"]
        ).astype(float)
    xi = plan.xi.copy()
    if "mu" not in xi.columns:
        xi["mu"] = xi["player_id"].astype(str).map(
            work.set_index(work["player_id"].astype(str))["mu"]
        ).astype(float)
    bench = bench_from_squad(squad, xi)
    return squad, xi, plan.captain_id, bench


def _commit_from_week0(held: HeldState, week0: Any, pool0: pd.DataFrame) -> HeldState:
    """Post-transfer 15 with purchase map; FT left for ``advance_planner_state``."""
    purchase = dict(held.purchase)
    for pid in week0.sold:
        purchase.pop(str(pid), None)
    squad = week0.squad.set_index(week0.squad["player_id"].astype(str))
    for pid in week0.bought:
        pid = str(pid)
        if pid in squad.index:
            purchase[pid] = int(squad.loc[pid, "value"])
        else:
            row = pool0.loc[pool0["player_id"].astype(str) == pid]
            purchase[pid] = int(row.iloc[0]["value"]) if len(row) else 45
    return HeldState(
        tuple(str(p) for p in week0.squad["player_id"].tolist()),
        purchase,
        bank=int(week0.bank_after),
        ft=int(held.ft),
    )


def _horizon_transfer_then_score(
    pools: Sequence[pd.DataFrame],
    held: HeldState,
    *,
    chip: str,
    time_limit: float,
    max_hits: int = DEFAULT_MAX_HITS,
    gamma: float = 1.0,
) -> BeamAction:
    """L4-identical transfer solve on remaining boards; score commit week under ``chip``."""
    if not pools:
        raise ValueError("horizon pools empty")
    prepared: list[pd.DataFrame] = []
    for p in pools:
        w = _ensure_held_rows(p, held, p)
        if "mu_post" not in w.columns:
            w = w.copy()
            w["mu_post"] = pd.to_numeric(w[_mu_col(w)], errors="coerce").fillna(0.0)
        prepared.append(w)
    hplan = solve_horizon_transfers(
        prepared,
        held,
        mu_col="mu_post",
        horizon=len(prepared),
        gamma=float(gamma),
        terminal_mu_weight=0.0,
        terminal_value_weight=0.0,
        max_hits=max_hits,
        time_limit=time_limit,
    )
    week0 = hplan.week0
    xi = week0.xi.copy()
    if "mu" not in xi.columns:
        xi["mu"] = pd.to_numeric(xi.get("mu_post", 0.0), errors="coerce").fillna(0.0)
    squad = week0.squad.copy()
    if "mu" not in squad.columns:
        squad["mu"] = squad["player_id"].astype(str).map(
            xi.set_index(xi["player_id"].astype(str))["mu"]
        ).astype(float)
    bench = bench_from_squad(squad, xi)
    score = score_commit_week(
        xi=xi,
        captain_id=week0.captain_id,
        bench=bench,
        chip=chip,
        hit_points=float(week0.hit_points),
        mu_col="mu",
    )
    commit = _commit_from_week0(held, week0, prepared[0])
    return BeamAction(
        chip=chip,
        commit_held=commit,
        xi=xi,
        captain_id=str(week0.captain_id),
        bench=bench,
        n_transfers=int(week0.n_transfers),
        hit_points=int(week0.hit_points),
        week_mu=float(score.objective),
    )


def _hold_xi_then_score(
    pool: pd.DataFrame,
    held: HeldState,
    *,
    chip: str,
    time_limit: float,
) -> BeamAction:
    """No transfers — pick XI+captain on this week's board (look-ahead depths)."""
    squad, xi, cap, bench = _xi_from_held(pool, held, time_limit=time_limit)
    score = score_commit_week(
        xi=xi,
        captain_id=cap,
        bench=bench,
        chip=chip,
        hit_points=0.0,
        mu_col="mu",
    )
    return BeamAction(
        chip=chip,
        commit_held=held_copy(held),
        xi=xi,
        captain_id=str(cap),
        bench=bench,
        n_transfers=0,
        hit_points=0,
        week_mu=float(score.objective),
    )


def _rebuild_then_score(
    pool: pd.DataFrame,
    held: HeldState,
    *,
    chip: str,
    moments: dict[str, Any] | None,
    time_limit: float,
) -> BeamAction:
    assert chip in ("wildcard", "free_hit")
    work = pool.copy()
    if "xp_se" not in work.columns:
        work["xp_se"] = 0.0
    if "score_xp_v2" not in work.columns:
        work["score_xp_v2"] = work[_mu_col(work)]
    commit = _rebuild_held(work, moments=moments, ft=held.ft, time_limit=time_limit)
    # XI from rebuild via 0-transfer on commit
    pool2 = _ensure_held_rows(work, commit, work)
    pool2 = _with_mu(pool2, _mu_col(pool2))
    _, xi, cap, bench = _xi_from_held(pool2, commit, time_limit=time_limit)
    score = score_commit_week(
        xi=xi, captain_id=cap, bench=bench, chip=NO_CHIP, hit_points=0.0, mu_col="mu"
    )
    return BeamAction(
        chip=chip,
        commit_held=commit,
        xi=xi,
        captain_id=cap,
        bench=bench,
        n_transfers=0,
        hit_points=0,
        week_mu=float(score.objective),
    )


def expand_node(
    node: BeamNode,
    *,
    gw: int,
    pool: pd.DataFrame,
    remaining_pools: Sequence[pd.DataFrame],
    moments: dict[str, Any] | None,
    time_limit: float,
    allow_squad_transfers: bool,
    allow_transfer_chips: bool,
    allowed_chips: frozenset[str] | None = None,
) -> list[tuple[BeamAction, PlannerState, float]]:
    """Expand one node into (action, next_state, new_cum_mu) children.

    Depth 0: horizon MILP transfers (L4) for none/TC/BB; WC/FH rebuild.
    Later depths: XI-only on the committed 15 (no further transfers).

    ``allowed_chips`` — optional ablation filter. ``None`` = all wallet-legal.
    """
    actions_chips = list(node.state.decision().actions(gw))
    if allowed_chips is not None:
        allow = set(allowed_chips)
        if NO_CHIP not in allow:
            allow.add(NO_CHIP)
        actions_chips = [c for c in actions_chips if c in allow]
        if NO_CHIP not in actions_chips:
            actions_chips = [NO_CHIP] + actions_chips
    if not allow_transfer_chips:
        actions_chips = [c for c in actions_chips if c not in ("wildcard", "free_hit")]
        if NO_CHIP not in actions_chips:
            actions_chips = [NO_CHIP] + actions_chips

    scoring = [c for c in actions_chips if c not in ("wildcard", "free_hit")]
    out: list[tuple[BeamAction, PlannerState, float]] = []

    if scoring:
        if allow_squad_transfers:
            transfer_base = _horizon_transfer_then_score(
                remaining_pools,
                node.state.held,
                chip=NO_CHIP,
                time_limit=time_limit,
            )
        else:
            transfer_base = _hold_xi_then_score(
                pool, node.state.held, chip=NO_CHIP, time_limit=time_limit
            )
        for chip in scoring:
            if chip == NO_CHIP:
                act = transfer_base
            else:
                score = score_commit_week(
                    xi=transfer_base.xi,
                    captain_id=transfer_base.captain_id,
                    bench=transfer_base.bench,
                    chip=chip,
                    hit_points=float(transfer_base.hit_points),
                    mu_col="mu",
                )
                act = BeamAction(
                    chip=chip,
                    commit_held=transfer_base.commit_held,
                    xi=transfer_base.xi,
                    captain_id=transfer_base.captain_id,
                    bench=transfer_base.bench,
                    n_transfers=transfer_base.n_transfers,
                    hit_points=transfer_base.hit_points,
                    week_mu=float(score.objective),
                )
            trans = advance_planner_state(
                node.state,
                gw=gw,
                chip=chip,
                commit_held=act.commit_held,
                n_transfers=act.n_transfers,
            )
            out.append((act, trans.next_state, node.cum_mu + act.week_mu))

    if allow_squad_transfers:
        for chip in actions_chips:
            if chip not in ("wildcard", "free_hit"):
                continue
            act = _rebuild_then_score(
                pool, node.state.held, chip=chip, moments=moments, time_limit=time_limit
            )
            trans = advance_planner_state(
                node.state,
                gw=gw,
                chip=chip,
                commit_held=act.commit_held,
                n_transfers=0,
            )
            out.append((act, trans.next_state, node.cum_mu + act.week_mu))

    return out


def run_beam(
    state: PlannerState,
    *,
    horizon_gws: list[int],
    pools: dict[int, pd.DataFrame],
    moments_by_gw: dict[int, dict[str, Any] | None],
    K: int = DEFAULT_BEAM_K,
    time_limit: float = 12.0,
    allowed_chips: frozenset[str] | None = None,
) -> BeamPlan:
    """Forward beam over ``horizon_gws``. Returns best node (first_action = week-0 commit)."""
    width = int(K)
    if width < 1:
        raise ValueError("K must be >= 1")
    if not horizon_gws:
        raise ValueError("empty horizon")

    frontier: list[BeamNode] = [BeamNode(state=state, cum_mu=0.0, depth=0)]
    n_expanded = 0

    for d, gw in enumerate(horizon_gws):
        pool = pools[int(gw)]
        moments = moments_by_gw.get(int(gw))
        remaining = [pools[int(g)] for g in horizon_gws[d:]]
        children: list[BeamNode] = []
        for node in frontier:
            # Depth 0: horizon MILP + WC/FH. Later: XI-only (L4 transfer law).
            expanded = expand_node(
                node,
                gw=int(gw),
                pool=pool,
                remaining_pools=remaining,
                moments=moments,
                time_limit=time_limit,
                allow_squad_transfers=(d == 0),
                allow_transfer_chips=(d == 0),
                allowed_chips=allowed_chips,
            )
            n_expanded += len(expanded)
            for act, nxt, cum in expanded:
                children.append(
                    BeamNode(
                        state=nxt,
                        cum_mu=float(cum),
                        depth=d + 1,
                        first_action=act if node.first_action is None else node.first_action,
                        path_chips=node.path_chips + (act.chip,),
                    )
                )
        children.sort(key=lambda n: n.cum_mu, reverse=True)
        frontier = children[:width]
        if not frontier:
            raise RuntimeError(f"beam empty at GW{gw}")

    best = frontier[0]
    runner = frontier[1] if len(frontier) > 1 else None
    return BeamPlan(
        best=best,
        planned_mu=float(best.cum_mu),
        K=width,
        horizon_gws=tuple(int(g) for g in horizon_gws),
        n_expanded=n_expanded,
        runner_up_mu=float(runner.cum_mu) if runner else None,
        margin_vs_runner=(
            float(best.cum_mu - runner.cum_mu) if runner is not None else None
        ),
    )
