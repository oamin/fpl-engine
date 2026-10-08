"""Compatibility shim. Prefer ``src.models.forecast_xp``.

The look-ahead API was renamed to ``forecast_xp`` on 2026-10-08. This module
re-exports the same symbols so historical callers keep working.
"""

from __future__ import annotations

from src.models.forecast_xp import (  # noqa: F401
    BASE_LAMBDA,
    MIN_OPENING_COVER,
    NEUTRAL_TOTAL,
    OUTRIGHT_SHRINK,
    SEASON_CODE,
    attach_forecast_xp,
    attach_opening_horizon,
    blend_pots,
    compute_player_forecast,
    deadline_rate_history,
    fixture_calendar,
    make_forecast_steps,
    make_horizon_scores,
    match_count_calendar,
    neutral_pot,
    opening_pots_by_team_gw,
    opening_pots_for_sheet,
    prior_pots,
    project_player,
    season_key,
    side_pot,
    single_fixture_calendar,
    strength_match_pots,
    xp_on_pot,
)
