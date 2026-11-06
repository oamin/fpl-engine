"""Cascade long branch — Opta-derived ``M`` for t+2 … t+n.

Same generative player stack as ``cascade_short`` (lineup → attack dice →
CS/GC from rolled GA → soft priors), **without player props** (unavailable
beyond priced weeks). Match inputs:

- Opta predicted ranks → season GF/GA (±MAE from closed-season place
  history) → match λ → Dixon–Coles ``M`` → ``k ∼ M`` → ClubPath update.

Buy universe: shared ``player_pool`` (FFScout firm ∩ expanding minutes).

Implementation entry: ``src.models.multistage_mc_milp``. See
``reports/CASCADE_PLAN.md`` / ``protocol.cascade_long_match_outcome``.
"""

from __future__ import annotations

from src.markets.cascade_score_v2 import SCORER, build_frozen_pool, score_pool_on_lam_pots
from src.models.multistage_mc_milp import (
    OBJECTIVES,
    ClubPath,
    run_multistage_trial,
)
from src.models.player_pool import POOL_JSON, load_player_pool_ids

BRANCH = "cascade_long"
HORIZON = "t+2:t+n"
MATCH_SOURCE = "opta_rank_gf_ga"

__all__ = [
    "BRANCH",
    "HORIZON",
    "MATCH_SOURCE",
    "OBJECTIVES",
    "POOL_JSON",
    "SCORER",
    "ClubPath",
    "build_frozen_pool",
    "load_player_pool_ids",
    "run_multistage_trial",
    "score_pool_on_lam_pots",
]
