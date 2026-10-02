"""Live chip policy locked with Gemini on 2026-10-02.

Search expected points of the current half only. Candidate weeks are the
current week, blanks, and doubles. A chip is played this week only when
this week is the best legal slot for it. Historical seasons are not searched.
Realised points are not an input.

Thresholds are the locked live policy, not the historical +34 climb bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rules.fpl_2026 import validate_chip_map

# Free Hit must beat the no-chip XI by this many expected points.
FH_MARGIN = 12.0
# Wildcard must beat the constrained path by this many expected points.
WC_MARGIN = 16.0


@dataclass(frozen=True)
class WeekOutlook:
    """Expected points for one future week. All figures are forecasts."""

    gw: int
    is_blank: bool
    is_double: bool
    best_xi: float
    bench_xp: float
    best_player_xp: float
    fh_xi: float | None = None
    wc_delta: float | None = None


def recommend_chip(
    current_gw: int,
    weeks: list[WeekOutlook],
    played: dict[int, str] | None = None,
) -> str | None:
    """Chip to play in ``current_gw``, or None.

    Raises if ``played`` is an illegal map. Does not look at results.
    """
    already = validate_chip_map(played)
    gains: dict[str, float] = {}

    doubles = [row for row in weeks if row.is_double]
    if doubles and _legal(already, current_gw, "bench_boost"):
        best = max(doubles, key=lambda row: (row.bench_xp, -row.gw))
        if best.gw == current_gw and best.bench_xp > 0:
            gains["bench_boost"] = best.bench_xp

    if doubles and _legal(already, current_gw, "triple_captain"):
        best = max(doubles, key=lambda row: (row.best_player_xp, -row.gw))
        if best.gw == current_gw and best.best_player_xp > 0:
            # Triple Captain adds one extra copy of the captain versus the normal double.
            gains["triple_captain"] = best.best_player_xp

    blanks = [row for row in weeks if row.is_blank and row.fh_xi is not None]
    if blanks and _legal(already, current_gw, "free_hit"):
        best = max(blanks, key=lambda row: ((row.fh_xi or 0) - row.best_xi, -row.gw))
        margin = (best.fh_xi or 0) - best.best_xi
        if best.gw == current_gw and margin >= FH_MARGIN:
            gains["free_hit"] = margin

    current = next((row for row in weeks if row.gw == current_gw), None)
    if (
        current is not None
        and current.wc_delta is not None
        and current.wc_delta >= WC_MARGIN
        and _legal(already, current_gw, "wildcard")
    ):
        gains["wildcard"] = current.wc_delta

    if not gains:
        return None
    ranked = sorted(gains.items(), key=lambda item: (-item[1], item[0]))
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def _legal(played: dict[int, str], gw: int, chip: str) -> bool:
    trial = dict(played)
    trial[int(gw)] = chip
    try:
        validate_chip_map(trial)
    except ValueError:
        return False
    return True
