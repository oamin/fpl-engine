"""Layer 2 — single-week joint MILP on decision μ (column ``mu_post``).

PI 2026-10-16: default ``mode='none'`` — ``mu_post`` is identity with
``score_xp_v2`` (constant-K shrink retired for decision boards). Historical
``mode='constant'`` / ``se`` remain available for screens only.
``score_xp_v2`` is not modified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.rules.fpl_2026 import BUDGET_TENTHS, MAX_PER_CLUB
from src.strategy.decision_value import attach_mu_post
from src.strategy.layer1_week import WeekPlan, solve_week
from src.strategy.shrinkage import CohortMoments, moments_to_dict


@dataclass(frozen=True)
class SigmaWeekPlan:
    """Layer-2 solution on μ_post plus absolute-μ diagnostics."""

    plan: WeekPlan
    mode: str
    abs_xi_mu: float
    abs_captain_mu: float
    abs_objective: float
    post_objective: float
    mean_se_xi: float
    moments: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        base = self.plan.summary()
        base.update(
            {
                "mode": self.mode,
                "abs_xi_mu": float(self.abs_xi_mu),
                "abs_captain_mu": float(self.abs_captain_mu),
                "abs_objective": float(self.abs_objective),
                "post_objective": float(self.post_objective),
                "mean_se_xi": float(self.mean_se_xi),
                "moments": self.moments,
            }
        )
        return base


def solve_week_sigma(
    pool: pd.DataFrame,
    *,
    mu_col: str = "score_xp_v2",
    se_col: str = "xp_se",
    pos_col: str = "position",
    mode: str = "none",
    moments_by_pos: dict[str, CohortMoments] | None = None,
    budget: int = BUDGET_TENTHS,
    max_per_club: int = MAX_PER_CLUB,
    time_limit: float = 30.0,
) -> SigmaWeekPlan:
    """Joint MILP maximising Σ μ_post·start + Σ μ_post·captain.

    Default ``mode='none'`` (2026-10-16): μ_post ≡ score_xp_v2. Pass
    ``mode='constant'`` only to reproduce the frozen se_credibility screen.
    """
    if se_col not in pool.columns:
        raise ValueError(f"missing SE column {se_col!r}")
    work, moments = attach_mu_post(
        pool,
        mu_col=mu_col,
        se_col=se_col,
        pos_col=pos_col,
        out_col="mu_post",
        mode=mode,
        moments_by_pos=moments_by_pos,
    )
    work["mu_abs"] = pd.to_numeric(work[mu_col], errors="coerce")
    work["se_abs"] = pd.to_numeric(work[se_col], errors="coerce").fillna(0.0)
    plan = solve_week(
        work,
        mu_col="mu_post",
        budget=budget,
        max_per_club=max_per_club,
        time_limit=time_limit,
    )

    meta = work.set_index(work["player_id"].astype(str))
    abs_map = meta["mu_abs"].astype(float)
    se_map = meta["se_abs"].astype(float)
    post_map = meta["mu_post"].astype(float)

    def _enrich(frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        ids = out["player_id"].astype(str)
        out["mu_abs"] = ids.map(abs_map).astype(float)
        out["mu_post"] = ids.map(post_map).astype(float)
        out["xp_se"] = ids.map(se_map).astype(float)
        return out.reset_index(drop=True)

    squad = _enrich(plan.squad)
    xi = _enrich(plan.xi)
    plan = WeekPlan(
        squad=squad,
        xi=xi,
        captain_id=plan.captain_id,
        formation=plan.formation,
        objective=plan.objective,
        xi_mu=plan.xi_mu,
        captain_mu=plan.captain_mu,
        cost=plan.cost,
        status=plan.status,
    )
    abs_xi = float(xi["mu_abs"].sum())
    abs_cap = float(xi.loc[xi["player_id"] == plan.captain_id, "mu_abs"].iloc[0])
    return SigmaWeekPlan(
        plan=plan,
        mode=str(mode),
        abs_xi_mu=abs_xi,
        abs_captain_mu=abs_cap,
        abs_objective=abs_xi + abs_cap,
        post_objective=float(plan.objective),
        mean_se_xi=float(xi["xp_se"].mean()),
        moments=moments_to_dict(moments),
    )
