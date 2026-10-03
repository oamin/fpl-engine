"""Captain, bench, and formation for one live squad.

The score column is an input. This module does not refit λ or replace
``score_xp`` on a historical season.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.models.season_climb import ordered_bench, pick_xi
from src.rules.fpl_2026 import OFFICIAL_FORMATIONS, captain_extra_points


def live_xi(squad: pd.DataFrame, score_col: str) -> dict[str, Any]:
    """Official formations, including 5-2-3. Captain and bench follow ``score_col``."""
    xi, form = pick_xi(squad, score_col, formations=list(OFFICIAL_FORMATIONS))
    score = pd.to_numeric(xi[score_col], errors="coerce")
    order = score.sort_values(ascending=False, kind="mergesort")
    captain = str(xi.loc[order.index[0], "player_id"])
    vice = str(xi.loc[order.index[1], "player_id"]) if len(order) > 1 else captain
    bench = ordered_bench(squad, xi, score_col)
    return {
        "formation": form,
        "xi": xi,
        "captain": captain,
        "vice": vice,
        "bench": bench,
        "xi_score": float(score.sum()),
        "bench_score": float(pd.to_numeric(bench[score_col], errors="coerce").fillna(0.0).sum()),
    }


def captain_extra(
    captain_points: float,
    vice_points: float,
    *,
    captain_played: bool,
    vice_played: bool,
    chip: str | None = None,
) -> float:
    """Delegate to the rules module. Triple Captain uses the same fall-through."""
    return captain_extra_points(
        captain_points,
        vice_points,
        captain_played=captain_played,
        vice_played=vice_played,
        chip=chip,
    )
