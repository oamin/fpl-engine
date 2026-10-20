"""Decision values for Layer 2 — attach ``mu_post`` for the MILP.

PI 2026-10-16: default ``mode='none'`` so ``mu_post`` ≡ raw ``score_xp_v2``.
Constant-K / SE shrink remain callable for historical screens only
(``se_credibility``, ``shrink_climb``). An η·SE penalty stays retired.
"""

from __future__ import annotations

import pandas as pd

from src.strategy.shrinkage import (
    CohortMoments,
    estimate_cohort_moments,
    shrink_frame,
    shrink_mu,
)

# Decision-board default (column name ``mu_post`` kept for solver APIs).
DECISION_SHRINK_MODE = "none"

# Re-exports for callers
__all__ = [
    "CohortMoments",
    "DECISION_SHRINK_MODE",
    "attach_mu_post",
    "estimate_cohort_moments",
    "shrink_mu",
]


def attach_mu_post(
    pool: pd.DataFrame,
    *,
    mu_col: str = "score_xp_v2",
    se_col: str = "xp_se",
    pos_col: str = "position",
    out_col: str = "mu_post",
    mode: str = DECISION_SHRINK_MODE,
    moments_by_pos: dict[str, CohortMoments] | None = None,
    use_sigma_model_in_k: bool = False,
) -> tuple[pd.DataFrame, dict[str, CohortMoments]]:
    """Attach decision means. ``mode`` in {none, constant, se}; default none."""
    return shrink_frame(
        pool,
        mu_col=mu_col,
        se_col=se_col,
        pos_col=pos_col,
        out_col=out_col,
        mode=mode,
        moments_by_pos=moments_by_pos,
        use_sigma_model_in_k=use_sigma_model_in_k,
    )
