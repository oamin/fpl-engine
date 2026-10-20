"""Defence points from a rolled match scoreline (cascade_short).

On a generative path the scoreline ``(g_home, g_away)`` is already drawn from
``M``. Clean sheets and goals-conceded deductions are deterministic given that
draw and the player's minutes / position — no separate xCS / xGC moments.

    CS  = 1 if goals_against == 0 else 0   (needs ≥60' and position CS pts)
    GC  = ⌊goals_against / 2⌋             (−1 per bin for GKP/DEF who played)

Firm XI / minutes still come from the lineup world (or mean m). Mean-layer
boards keep moment-based ``xcs`` / ``xgc`` via ``score_team_side``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from src.rules.fpl_2026 import (
    clean_sheet_points,
    goals_conceded_points,
    normalize_position,
)


@dataclass(frozen=True)
class DefenceRoll:
    """Club defence outcome implied by one rolled scoreline."""

    goals_against: int
    clean_sheet: bool
    gc_bins: int  # ⌊GA / 2⌋

    def as_dict(self) -> dict[str, Any]:
        return {
            "goals_against": self.goals_against,
            "clean_sheet": self.clean_sheet,
            "gc_bins": self.gc_bins,
        }


def defence_from_scoreline(goals_against: int) -> DefenceRoll:
    """Map rolled GA → CS flag and GC deduction bins."""
    ga = int(max(0, goals_against))
    return DefenceRoll(
        goals_against=ga,
        clean_sheet=ga == 0,
        gc_bins=ga // 2,
    )


def xp_defence_on_roll(
    positions: Sequence[str],
    minutes: np.ndarray | Sequence[float],
    goals_against: int,
) -> tuple[np.ndarray, np.ndarray, DefenceRoll]:
    """Per-player CS / GC xP from a rolled scoreline + minutes world.

    Uses ``clean_sheet_points`` / ``goals_conceded_points`` from ``fpl_2026``
    (CS needs ≥60'; GKP/DEF GC for any minutes > 0).
    """
    defence = defence_from_scoreline(goals_against)
    m = np.asarray(minutes, dtype=float)
    n = m.size
    if len(positions) != n:
        raise ValueError("positions / minutes length mismatch")
    xp_cs = np.zeros(n, dtype=float)
    xp_gc = np.zeros(n, dtype=float)
    for i, raw_pos in enumerate(positions):
        pos = normalize_position("GKP" if str(raw_pos) == "GK" else str(raw_pos))
        mi = int(round(float(m[i])))
        xp_cs[i] = float(
            clean_sheet_points(pos, mi, defence.clean_sheet)
        )
        xp_gc[i] = float(
            goals_conceded_points(pos, mi, defence.goals_against)
        )
    return xp_cs, xp_gc, defence
