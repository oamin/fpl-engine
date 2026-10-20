"""Cascade long branch — Opta-derived match λ for t+2 … t+n.

Same scoring principle as ``cascade_short`` (``score_xp_v2`` via
``cascade_score_v2`` / ``score_team_side``), different match inputs:

- Opta predicted ranks → season GF/GA (±MAE) → Dixon–Coles λ → ClubPath roll.

Buy universe: shared ``player_pool`` (FFScout firm ∩ expanding minutes).

Implementation entry: ``src.models.multistage_mc_milp``. See
``reports/CASCADE_PLAN.md``.
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
