"""Competitive strategiser (decision layer).

``score_xp_v2`` stays the absolute scorer. This package turns μ into legal
squad actions.

Layer 1: single-week joint MILP on absolute μ.
Layer 2: shrink μ → μ_post with estimation SE (EB), then the same MILP on
μ_post. No η·SE penalty; ownership tilt scrapped.
Layer 3: transfers + hits from a held 15 on μ_post (single deadline).
Layer 4: receding horizon — transfers at t=0, XI+captain over H weeks.
"""

from src.strategy.decision_value import attach_mu_post
from src.strategy.layer1_week import WeekPlan, solve_week
from src.strategy.layer2_week import SigmaWeekPlan, solve_week_sigma
from src.strategy.layer3_week import HeldState, TransferWeekPlan, solve_week_transfers
from src.strategy.layer4_week import HorizonPlan, solve_horizon_transfers
from src.strategy.shrinkage import CohortMoments, shrink_mu

__all__ = [
    "WeekPlan",
    "SigmaWeekPlan",
    "TransferWeekPlan",
    "HorizonPlan",
    "HeldState",
    "CohortMoments",
    "solve_week",
    "solve_week_sigma",
    "solve_week_transfers",
    "solve_horizon_transfers",
    "attach_mu_post",
    "shrink_mu",
]
