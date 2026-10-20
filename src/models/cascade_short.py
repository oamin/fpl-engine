"""Cascade short branch — priced horizon t : t+1.

Mean layer matches ``cascade_long`` (``score_xp_v2`` / ``score_team_side``).
Scenario strands on priced weeks use generative attack
(``short_horizon_generative``: ``k ∼ M`` → lineup cov → scorer/assist dice).

Match inputs:

- **Historical screens:** football-data AH+OU Dixon–Coles λ (± band scenarios).
- **Live:** book prices (Betfair / live pick snapshots; not this module's pull).

Buy universe: shared ``player_pool`` (FFScout firm ∩ expanding minutes).

Implementation entry: ``src.models.short_horizon_cascade``. L4 path boards use
``scored_weeks_to_l4_pools`` → ``layer4_mu_inject``. See ``reports/CASCADE_PLAN.md``.
"""

from __future__ import annotations

from src.models.player_pool import POOL_JSON, filter_board_to_pool, load_player_pool_ids
from src.models.short_horizon_cascade import (
    SCORER,
    attach_prices_and_cap,
    run_decision,
    scored_weeks_to_l4_pools,
)

BRANCH = "cascade_short"
HORIZON = "t:t+1"
MATCH_SOURCE = "priced_fd_or_live_odds"

__all__ = [
    "BRANCH",
    "HORIZON",
    "MATCH_SOURCE",
    "POOL_JSON",
    "SCORER",
    "attach_prices_and_cap",
    "filter_board_to_pool",
    "load_player_pool_ids",
    "run_decision",
    "scored_weeks_to_l4_pools",
]
